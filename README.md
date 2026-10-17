# Chatbot hỏi đáp BCTC (RAG) — bản triển khai Streamlit Community Cloud

1. Đưa thư mục này lên một repo GitHub **public** (nhớ bỏ PDF BCTC vào `data/`).
2. Vào share.streamlit.io → New app → chọn repo, branch, main file `app.py`.
3. Advanced settings → Secrets, dán:
   ```
   LLM_API_KEY = "khoa-gemini-cua-ban"
   # LLM_MODEL = "gemini-3.8-flash"
   # APP_PASSWORD = "mat-khau-neu-muon-khoa"
   ```
4. Deploy. Không bao giờ commit API key vào repo.
