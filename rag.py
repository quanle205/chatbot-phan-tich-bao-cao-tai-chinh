"""Pipeline RAG cho BCTC: parse PDF (bảng -> Markdown) -> chunk -> hybrid search (BM25 + dense, RRF)."""
import hashlib, json, os, re
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


# ---------- 1. Parse PDF ----------
def parse_pdf(data: bytes, name: str):
    """Trả về list page: {page, text, tables[markdown]}. Bảng được tách riêng để không vỡ hàng/cột."""
    doc = pymupdf.open(stream=data, filetype="pdf")
    pages = []
    for i, page in enumerate(doc, 1):
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
        pages.append({"page": i, "text": "\n".join(blocks), "tables": [m for _, m in tables]})
    return pages


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


def build_index(files: dict, use_dense=True):
    """files: {tên_file: bytes}. Trả về (Index, cảnh báo)."""
    chunks, warns = [], []
    for name, data in files.items():
        pages = parse_pdf(data, name)
        if sum(len(p["text"]) + sum(map(len, p["tables"])) for p in pages) < 40 * max(len(pages), 1):
            warns.append(f"{name}: gần như không có text — có thể là PDF scan, cần OCR trước.")
        chunks += make_chunks(pages, name, start_id=len(chunks))
    return Index(chunks, use_dense), warns
