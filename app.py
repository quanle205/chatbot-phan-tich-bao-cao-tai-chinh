import glob, os, re

import streamlit as st

# ---- Cấu hình LLM từ Streamlit Secrets (phải đặt TRƯỚC khi import llm) ----
def _secret(name, default=""):
    try:
        return str(st.secrets[name])
    except Exception:
        return os.environ.get(name, default)

_provider = _secret("LLM_PROVIDER", "openai")  # "openai" (dùng cho Gemini) hoặc "anthropic"
os.environ["LLM_PROVIDER"] = _provider
os.environ["LLM_MODEL"] = _secret("LLM_MODEL", "gemini-3.8-flash")
if _provider == "anthropic":
    os.environ["ANTHROPIC_API_KEY"] = _secret("LLM_API_KEY")
else:
    os.environ["OPENAI_API_KEY"] = _secret("LLM_API_KEY")
    os.environ["OPENAI_BASE_URL"] = _secret("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

import pymupdf
from rag import build_index
import llm

st.set_page_config(page_title="Chatbot phân tích BCTC", page_icon="📊", layout="wide")
st.title("📊 Chatbot hỏi đáp Báo cáo tài chính (RAG)")

# ---- (Tuỳ chọn) khoá bằng mật khẩu để người lạ không dùng hết hạn mức API ----
_pwd = _secret("APP_PASSWORD")
if _pwd and st.text_input("Mật khẩu", type="password") != _pwd:
    st.info("Nhập mật khẩu để sử dụng.")
    st.stop()

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SAMPLES = ["Doanh thu thuần quý I là bao nhiêu?", "Lợi nhuận sau thuế của công ty mẹ là bao nhiêu?",
           "Tổng tài sản tại ngày kết thúc kỳ là bao nhiêu?", "Tính tăng trưởng doanh thu so với cùng kỳ năm trước",
           "Biên lợi nhuận gộp là bao nhiêu phần trăm?", "Dòng tiền thuần từ hoạt động kinh doanh là bao nhiêu?"]


@st.cache_resource(show_spinner="Đang đọc PDF, tách bảng, tạo index...")
def load_default(use_dense):
    files = {os.path.basename(p): open(p, "rb").read() for p in sorted(glob.glob(os.path.join(DATA, "*.pdf")))}
    if not files:
        return None, {}, []
    idx, warns = build_index(files, use_dense)
    return idx, files, warns


with st.sidebar:
    st.header("Dữ liệu")
    use_dense = st.checkbox("Dùng embedding (cần nhiều RAM)", os.getenv("USE_DENSE", "0") == "1",
                            help="Trên Streamlit Cloud (1 GB RAM) nên để tắt → chỉ dùng BM25")
    top_k = st.slider("Số đoạn truy xuất (Top-K)", 3, 12, 6)
    alpha = st.slider("Trọng số dense vs BM25", 0.0, 1.0, 0.5, 0.1, disabled=not use_dense)
    st.caption(f"Model: {llm.PROVIDER} / {llm.MODEL}")
    ups = st.file_uploader("Hoặc tải PDF khác để thử", type="pdf", accept_multiple_files=True)
    if ups and st.button("Xây dựng index từ file tải lên", type="primary"):
        files = {u.name: u.getvalue() for u in ups}
        with st.spinner("Đang xử lý..."):
            idx, warns = build_index(files, use_dense)
        st.session_state.update(index=idx, files=files, history=[], chat=[], warns=warns)

if "index" not in st.session_state:
    idx, files, warns = load_default(use_dense)
    if idx is not None:
        st.session_state.update(index=idx, files=files, history=[], chat=[], warns=warns)

if "index" not in st.session_state:
    st.info("Chưa có dữ liệu: đặt file PDF BCTC vào thư mục `data/` của repo, hoặc tải file lên ở thanh bên.")
    st.stop()

with st.sidebar:
    for w in st.session_state.get("warns", []):
        st.warning(w)
    ch = st.session_state.index.chunks
    st.success(f"Đã index {len(ch)} đoạn ({sum(c.kind == 'table' for c in ch)} bảng) từ {len(st.session_state.files)} file")
    st.caption("Tìm kiếm: " + ("hybrid (BM25 + embedding)" if st.session_state.index.emb is not None else "BM25"))
    st.header("Câu hỏi mẫu")
    pick = None
    for s in SAMPLES:
        if st.button(s, use_container_width=True):
            pick = s


def render_sources(hits, text, key):
    cited = sorted({int(n) for n in re.findall(r"\[#(\d+)\]", text) if 0 < int(n) <= len(hits)})
    for n in cited or range(1, min(3, len(hits)) + 1):
        c, _ = hits[n - 1]
        with st.expander(f"[#{n}] {c.doc} — trang {c.page} ({'bảng' if c.kind == 'table' else 'văn bản'})"):
            st.markdown(c.text)
            if st.checkbox("Xem trang PDF gốc", key=f"pg{key}_{n}"):
                pdf = pymupdf.open(stream=st.session_state.files[c.doc], filetype="pdf")
                st.image(pdf[c.page - 1].get_pixmap(dpi=110).tobytes("png"))


for i, m in enumerate(st.session_state.chat):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("hits"):
            render_sources(m["hits"], m["content"], i)

q = st.chat_input("Hỏi về báo cáo tài chính...") or pick
if q:
    st.session_state.chat.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        h = st.session_state.history
        search_q = q if len(q.split()) > 8 or not h else h[-2]["content"].split("CÂU HỎI:")[-1].strip() + " " + q
        hits = st.session_state.index.search(search_q, top_k=top_k, alpha=alpha)
        try:
            with st.spinner("Đang phân tích..."):
                text, calcs = llm.answer(q, hits, h[-6:])
        except Exception as e:
            text, calcs = f"Lỗi gọi LLM: {e}\n\nKiểm tra `LLM_API_KEY` và `LLM_MODEL` trong Secrets.", []
        st.markdown(text)
        if calcs:
            with st.expander("🧮 Phép tính đã thực hiện"):
                for e, r in calcs:
                    st.code(f"{e} = {r}")
        render_sources(hits, text, len(st.session_state.chat))
    h.append({"role": "user", "content": f"CÂU HỎI: {q}"})
    h.append({"role": "assistant", "content": text})
    st.session_state.chat.append({"role": "assistant", "content": text, "hits": hits})
