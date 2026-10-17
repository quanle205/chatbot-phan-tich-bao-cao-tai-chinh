"""Pipeline RAG cho BCTC: parse PDF (bảng -> Markdown) -> chunk -> hybrid search (BM25 + dense, RRF)."""
import glob, hashlib, json, os, re
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
OCR_DPI = int(os.getenv("OCR_DPI", "200"))
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


def _ocr_page(page, lang):
    tp = page.get_textpage_ocr(language=lang, dpi=OCR_DPI, full=True)
    return page.get_text("text", textpage=tp, sort=True).strip()


def parse_pdf(data: bytes, name: str, progress=None):
    """Trả về (pages, info). pages: list {page, text, tables[markdown], ocr}. Bảng được tách riêng để không vỡ hàng/cột."""
    doc = pymupdf.open(stream=data, filetype="pdf")
    pages, info = [], {"ocr_pages": 0, "ocr_missing": False, "ocr_skipped": 0}
    lang = None
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
                        o = _ocr_page(page, lang)
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


def build_index(files: dict, use_dense=True, progress=None):
    """files: {tên_file: bytes}. progress(msg, fraction) tuỳ chọn. Trả về (Index, cảnh báo)."""
    chunks, warns = [], []
    names = list(files)
    for k, name in enumerate(names):
        cb = (lambda i, n, k=k, name=name: progress(f"{name}: trang {i}/{n}", (k + i / n) / len(names))) if progress else None
        pages, info = parse_pdf(files[name], name, cb)
        if info["ocr_pages"]:
            warns.append(f"{name}: {info['ocr_pages']} trang là ảnh scan, đã dùng OCR. OCR có thể đọc sai chữ/số — hãy đối chiếu số liệu với file gốc.")
        if info["ocr_missing"]:
            warns.append(f"{name}: có trang scan nhưng chưa cài Tesseract nên chưa đọc được (cần packages.txt: tesseract-ocr, tesseract-ocr-vie).")
        if info["ocr_skipped"]:
            warns.append(f"{name}: bỏ qua {info['ocr_skipped']} trang scan vượt giới hạn OCR ({MAX_OCR_PAGES} trang/file).")
        if sum(len(p["text"]) + sum(map(len, p["tables"])) for p in pages) < 40 * max(len(pages), 1):
            warns.append(f"{name}: vẫn gần như không có text sau khi OCR — file có thể bị lỗi hoặc ảnh quá mờ.")
        chunks += make_chunks(pages, name, start_id=len(chunks))
    if progress:
        progress("Đang tạo index...", 1.0)
    return Index(chunks, use_dense), warns
