"""Pipeline RAG cho BCTC: parse PDF (bảng -> Markdown) -> chunk -> hybrid search (BM25 + dense, RRF)."""
import csv, glob, hashlib, io, json, os, re
from dataclasses import dataclass, asdict

import numpy as np
import pymupdf
from rank_bm25 import BM25Okapi

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")  # có thể đổi sang BAAI/bge-m3
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


@dataclass
class Chunk:
    id: int
    doc: str
    page: int
    kind: str  # "text" | "table"
    text: str


# ---------- 1. Parse PDF (có OCR cho trang scan) ----------
OCR_LANG = os.getenv("OCR_LANG", "vie+eng")
OCR_DPI = int(os.getenv("OCR_DPI", "300"))  # cao hon = nhan dien chu/so chinh xac hon, nhung cham hon
MAX_OCR_PAGES = int(os.getenv("MAX_OCR_PAGES", "80"))  # giới hạn số trang OCR mỗi file để khỏi chạy quá lâu


def _setup_tesseract():
    """Trả về ngôn ngữ OCR dùng được (vie+eng, eng) hoặc None nếu chưa cài tesseract."""
    if not os.environ.get("TESSDATA_PREFIX"):
        for d in glob.glob("/usr/share/tesseract-ocr/*/tessdata") + glob.glob("/usr/share/tessdata"):
            if os.path.isdir(d):
                os.environ["TESSDATA_PREFIX"] = d
                break
    d = os.environ.get("TESSDATA_PREFIX", "")
    if not d or not os.path.isdir(d):
        return None
    have = {f.split(".")[0] for f in os.listdir(d) if f.endswith(".traineddata")}
    langs = [l for l in OCR_LANG.split("+") if l in have]
    return "+".join(langs) or None


def _needs_ocr(text, tables):
    n = len(text) + sum(len(t) for t in tables)
    if n < 80:
        return True
    bad = sum(1 for ch in text if ch == "\ufffd" or 0xE000 <= ord(ch) <= 0xF8FF)  # font mã hoá lỗi
    return bad > 0.15 * max(len(text), 1)


def _ocr_cache_path(file_hash, page_no):
    d = os.path.join(CACHE_DIR, "ocr")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{file_hash}_{page_no}_{OCR_LANG}_{OCR_DPI}.txt")


def _ocr_page(page, lang, cache_key=None):
    if cache_key:
        p = _ocr_cache_path(*cache_key)
        if os.path.exists(p):
            return open(p, encoding="utf-8").read()
    tp = page.get_textpage_ocr(language=lang, dpi=OCR_DPI, full=True)
    text = page.get_text("text", textpage=tp, sort=True).strip()
    if cache_key:
        try:
            open(_ocr_cache_path(*cache_key), "w", encoding="utf-8").write(text)
        except OSError:
            pass
    return text


def parse_pdf(data: bytes, name: str, progress=None):
    """Trả về (pages, info). pages: list {page, text, tables[markdown], ocr}. Bảng được tách riêng để không vỡ hàng/cột."""
    doc = pymupdf.open(stream=data, filetype="pdf")
    pages, info = [], {"ocr_pages": 0, "ocr_missing": False, "ocr_skipped": 0}
    lang = None
    file_hash = hashlib.md5(data).hexdigest()[:16]
    for i, page in enumerate(doc, 1):
        if progress:
            progress(i, len(doc))
        tables = []
        try:
            for t in page.find_tables().tables:
                md = t.to_markdown().strip()
                if md:
                    tables.append((pymupdf.Rect(t.bbox), md))
        except Exception:
            pass
        blocks = []
        for b in page.get_text("blocks", sort=True):
            if b[6] != 0 or not b[4].strip():
                continue
            r = pymupdf.Rect(b[:4])
            if any((r & tr).get_area() > 0.5 * max(r.get_area(), 1e-6) for tr, _ in tables):
                continue  # block này đã nằm trong bảng
            blocks.append(b[4].strip())
        text, tmd, ocr = "\n".join(blocks), [m for _, m in tables], False
        if _needs_ocr(text, tmd):
            if info["ocr_pages"] >= MAX_OCR_PAGES:
                info["ocr_skipped"] += 1
            else:
                if lang is None:
                    lang = _setup_tesseract() or ""
                if not lang:
                    info["ocr_missing"] = True
                else:
                    try:
                        o = _ocr_page(page, lang, cache_key=(file_hash, i))
                        if len(o) > len(text):
                            text, tmd, ocr = o, [], True
                            info["ocr_pages"] += 1
                    except Exception:
                        info["ocr_missing"] = True
        pages.append({"page": i, "text": text, "tables": tmd, "ocr": ocr})
    return pages, info


# ---------- 2. Chunking ----------
def _split_text(text, size=900, overlap=150):
    paras = [p.strip() for p in re.split(r"\n{1,}", text) if p.strip()]
    out, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > size:
            out.append(cur)
            cur = cur[-overlap:] + "\n" + p
        else:
            cur = (cur + "\n" + p) if cur else p
    if cur:
        out.append(cur)
    return out


def _split_table(md, max_chars=2500):
    if len(md) <= max_chars:
        return [md]
    lines = md.split("\n")
    header, rows = lines[:2], lines[2:]  # giữ header cho mỗi mảnh
    out, cur = [], list(header)
    for r in rows:
        if sum(len(x) for x in cur) + len(r) > max_chars and len(cur) > 2:
            out.append("\n".join(cur))
            cur = list(header)
        cur.append(r)
    out.append("\n".join(cur))
    return out


def make_chunks(pages, doc_name, start_id=0):
    chunks, cid = [], start_id
    for p in pages:
        for t in _split_text(p["text"]):
            if len(t) > 30:
                chunks.append(Chunk(cid, doc_name, p["page"], "text", t)); cid += 1
        for tb in p["tables"]:
            for part in _split_table(tb):
                chunks.append(Chunk(cid, doc_name, p["page"], "table", part)); cid += 1
    return chunks


# ---------- 3. Index & Hybrid retrieval ----------
def tokenize(text):
    toks = re.findall(r"\w+", text.lower())
    nums = [re.sub(r"\D", "", t) for t in re.findall(r"\d[\d.,]*\d", text)]  # 1.234.567 -> 1234567
    bigrams = [f"{a}_{b}" for a, b in zip(toks, toks[1:])]  # bắt cụm từ tiếng Việt: "lợi_nhuận"
    return toks + bigrams + [n for n in nums if len(n) > 2]


class Index:
    def __init__(self, chunks, use_dense=True):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None
        self.emb, self.model = None, None
        if use_dense and chunks:
            self._build_dense()

    def _prefix(self, kind):
        return {"query": "query: ", "passage": "passage: "}[kind] if "e5" in EMBED_MODEL.lower() else ""

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(EMBED_MODEL)
        return self.model

    def _build_dense(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        key = hashlib.md5((EMBED_MODEL + "".join(c.text for c in self.chunks)).encode()).hexdigest()
        path = os.path.join(CACHE_DIR, f"{key}.npy")
        try:
            if os.path.exists(path):
                self.emb = np.load(path)
            else:
                m = self._load_model()
                self.emb = m.encode([self._prefix("passage") + c.text for c in self.chunks],
                                    normalize_embeddings=True, batch_size=32, show_progress_bar=False)
                np.save(path, self.emb)
        except Exception as e:  # không tải được model -> chỉ dùng BM25
            print(f"[WARN] Dense embedding không khả dụng ({e.__class__.__name__}); chỉ dùng BM25.")
            self.emb = None

    def search(self, query, top_k=6, alpha=0.5, pool=30):
        """RRF giữa BM25 và dense. alpha = trọng số dense (0..1)."""
        if not self.chunks:
            return []
        n = len(self.chunks)
        bm = self.bm25.get_scores(tokenize(query))
        scores = {}
        has_pos = bool((bm > 0).any())
        for r, i in enumerate(np.argsort(-bm)[:pool]):
            if bm[i] > 0 or not has_pos:  # kho quá nhỏ -> BM25 có thể toàn 0, vẫn xếp theo thứ tự
                scores[int(i)] = scores.get(int(i), 0) + (1 - alpha if self.emb is not None else 1) / (60 + r)
        if self.emb is not None:
            q = self._load_model().encode([self._prefix("query") + query], normalize_embeddings=True)[0]
            sim = self.emb @ q
            for r, i in enumerate(np.argsort(-sim)[:pool]):
                scores[int(i)] = scores.get(int(i), 0) + alpha / (60 + r)
        top = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [(self.chunks[i], scores[i]) for i in top]


def parse_docx(data: bytes, name: str):
    """Word: mỗi heading cấp 1 coi là 1 'trang' ảo để còn dẫn nguồn được; bảng Word -> Markdown."""
    import docx
    d = docx.Document(io.BytesIO(data))
    pages, buf, pno = [], [], 1

    def flush():
        nonlocal buf, pno
        if buf:
            pages.append({"page": pno, "text": "\n".join(buf).strip(), "tables": [], "ocr": False})
            buf, pno = [], pno + 1

    for el in d.element.body:
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "p":
            from docx.text.paragraph import Paragraph
            para = Paragraph(el, d)
            if para.style and para.style.name and para.style.name.lower().startswith("heading") and buf:
                flush()
            if para.text.strip():
                buf.append(para.text.strip())
        elif tag == "tbl":
            from docx.table import Table
            t = Table(el, d)
            rows = [[c.text.strip() for c in r.cells] for r in t.rows]
            if rows:
                md = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
                md += ["| " + " | ".join(r) + " |" for r in rows[1:]]
                pages.append({"page": pno, "text": "", "tables": ["\n".join(md)], "ocr": False})
                pno += 1
    flush()
    if not pages:
        pages = [{"page": 1, "text": "", "tables": [], "ocr": False}]
    return pages, {"ocr_pages": 0, "ocr_missing": False, "ocr_skipped": 0}


def parse_csv(data: bytes, name: str):
    """CSV: toàn bộ thành 1 bảng Markdown (coi là 1 trang)."""
    text = data.decode("utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:2000], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.reader(io.StringIO(text), dialect))
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return [{"page": 1, "text": "", "tables": [], "ocr": False}], {"ocr_pages": 0, "ocr_missing": False, "ocr_skipped": 0}
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    md = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width] + ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return [{"page": 1, "text": "", "tables": ["\n".join(md)], "ocr": False}], {"ocr_pages": 0, "ocr_missing": False, "ocr_skipped": 0}


def parse_xlsx(data: bytes, name: str):
    """Excel: mỗi sheet là 1 'trang' (số thứ tự), bảng -> Markdown; ô trống ở cuối dòng/cột được cắt bớt."""
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True, read_only=True)
    pages = []
    for i, ws in enumerate(wb.worksheets, 1):
        rows = []
        for row in ws.iter_rows(values_only=True):
            if any(v not in (None, "") for v in row):
                rows.append(["" if v is None else str(v) for v in row])
        if not rows:
            continue
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        md = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width] + ["| " + " | ".join(r) + " |" for r in rows[1:]]
        pages.append({"page": i, "text": f"[Sheet: {ws.title}]", "tables": ["\n".join(md)], "ocr": False})
    if not pages:
        pages = [{"page": 1, "text": "(file Excel rỗng)", "tables": [], "ocr": False}]
    return pages, {"ocr_pages": 0, "ocr_missing": False, "ocr_skipped": 0}


def parse_txt(data: bytes, name: str):
    """File .txt: coi là văn bản đã được OCR/làm sạch từ nơi khác. Trang ngăn bằng dòng \f hoặc '---PAGE n---'."""
    text = data.decode("utf-8", errors="replace")
    parts = re.split(r"\f|^-{0,3}\s*PAGE\s+\d+\s*-{0,3}\s*$", text, flags=re.MULTILINE | re.IGNORECASE)
    parts = [p for p in parts if p.strip()] or [text]
    return [{"page": i, "text": p.strip(), "tables": [], "ocr": False} for i, p in enumerate(parts, 1)], \
           {"ocr_pages": 0, "ocr_missing": False, "ocr_skipped": 0}


SUPPORTED_EXT = (".pdf", ".docx", ".csv", ".xlsx", ".xls", ".txt")
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "40"))

_PARSERS = {
    ".pdf": lambda data, name, cb: parse_pdf(data, name, cb),
    ".docx": lambda data, name, cb: parse_docx(data, name),
    ".csv": lambda data, name, cb: parse_csv(data, name),
    ".xlsx": lambda data, name, cb: parse_xlsx(data, name),
    ".xls": lambda data, name, cb: parse_xlsx(data, name),
    ".txt": lambda data, name, cb: parse_txt(data, name),
}
_NO_OCR_INFO = {"ocr_pages": 0, "ocr_missing": False, "ocr_skipped": 0}


def build_index(files: dict, use_dense=True, progress=None):
    """files: {tên_file: bytes}. progress(msg, fraction) tuỳ chọn. Trả về (Index, cảnh báo, thống kê mỗi file).
    Tự bỏ qua file lỗi/không hỗ trợ/quá lớn thay vì làm hỏng cả mẻ; mỗi file được báo riêng trong `warns`/`stats`."""
    chunks, warns, stats = [], [], []
    names = list(files)
    seen_hash = {}
    for k, name in enumerate(names):
        ext = os.path.splitext(name)[1].lower()
        data = files[name]
        size_mb = len(data) / 1_000_000
        cb = (lambda i, n, k=k, name=name: progress(f"{name}: trang {i}/{n}", (k + i / n) / len(names))) if progress else None

        if ext not in SUPPORTED_EXT:
            warns.append(f"{name}: bỏ qua — định dạng {ext or '(không rõ)'} chưa hỗ trợ (chỉ nhận PDF, DOCX, CSV, XLSX, TXT).")
            continue
        if size_mb > MAX_FILE_MB:
            warns.append(f"{name}: bỏ qua — file {size_mb:.0f} MB vượt giới hạn {MAX_FILE_MB} MB.")
            continue
        h = hashlib.md5(data).hexdigest()
        if h in seen_hash:
            warns.append(f"{name}: bỏ qua — nội dung trùng với file '{seen_hash[h]}' đã nạp.")
            continue
        seen_hash[h] = name

        try:
            pages, info = _PARSERS[ext](data, name, cb)
        except Exception as e:
            warns.append(f"{name}: lỗi khi đọc file ({e.__class__.__name__}: {e}) — đã bỏ qua, các file khác vẫn được xử lý.")
            continue

        if info["ocr_pages"]:
            warns.append(f"{name}: {info['ocr_pages']} trang là ảnh scan, đã dùng OCR. OCR có thể đọc sai chữ/số — hãy đối chiếu số liệu với file gốc.")
        if info["ocr_missing"]:
            warns.append(f"{name}: có trang scan nhưng chưa cài Tesseract nên chưa đọc được (cần packages.txt: tesseract-ocr, tesseract-ocr-vie).")
        if info["ocr_skipped"]:
            warns.append(f"{name}: bỏ qua {info['ocr_skipped']} trang scan vượt giới hạn OCR ({MAX_OCR_PAGES} trang/file).")
        total_chars = sum(len(p["text"]) + sum(map(len, p["tables"])) for p in pages)
        # Chỉ cảnh báo khi thực sự gần như trống (file lỗi/mã hoá lạ/ảnh quá mờ), không áp cho
        # file ngắn hợp lệ (ghi chú 1 dòng, 1 trang ít chữ...) — dùng ngưỡng tuyệt đối, không nhân theo số trang.
        if ext == ".pdf" and total_chars < 25 and not info["ocr_pages"]:
            warns.append(f"{name}: gần như không có nội dung đọc được — file có thể bị lỗi, mã hoá lạ, hoặc là PDF scan cần OCR.")
        elif ext != ".pdf" and total_chars < 10:
            warns.append(f"{name}: gần như không có nội dung đọc được — file có thể bị lỗi hoặc rỗng.")

        start = len(chunks)
        file_chunks = make_chunks(pages, name, start_id=start)
        chunks += file_chunks
        stats.append({"name": name, "pages": len(pages), "chunks": len(file_chunks),
                      "tables": sum(c.kind == "table" for c in file_chunks), "ocr_pages": info["ocr_pages"]})
    if progress:
        progress("Đang tạo index...", 1.0)
    return Index(chunks, use_dense), warns, stats
