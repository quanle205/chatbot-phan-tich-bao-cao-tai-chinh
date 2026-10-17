"""Sinh câu trả lời có trích dẫn + tool tính toán an toàn (tránh LLM tự nhẩm sai số lớn)."""
import ast, json, operator, os

PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # anthropic | openai (OpenAI-compatible)
MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5" if PROVIDER == "anthropic" else "gpt-4o-mini")

SYSTEM = """Bạn là trợ lý phân tích báo cáo tài chính doanh nghiệp (tiếng Việt).
QUY TẮC BẮT BUỘC:
1. Chỉ trả lời dựa trên các đoạn NGỮ CẢNH được cung cấp. Không dùng kiến thức bên ngoài để bịa số liệu.
2. Sau mỗi số liệu/khẳng định phải ghi nguồn dạng [#n] (n là số thứ tự đoạn ngữ cảnh).
3. Nếu ngữ cảnh không đủ để trả lời, nói rõ: "Tài liệu được cung cấp không có thông tin này." và không đoán.
4. Giữ nguyên đơn vị (VND, triệu đồng, tỷ đồng...) và kỳ báo cáo như trong tài liệu; nêu rõ kỳ/ngày của số liệu.
5. KHÔNG tự tính nhẩm. Mọi phép tính (tăng trưởng %, biên lợi nhuận, tỷ số, tổng/hiệu) phải gọi công cụ `calculate`
   với số liệu lấy từ ngữ cảnh (viết số thuần, không dấu phân cách, ví dụ 1234567.8), rồi trình bày công thức + kết quả.
6. Trả lời ngắn gọn, rõ ràng; dùng bảng markdown khi so sánh nhiều chỉ tiêu."""

TOOL_DESC = "Tính biểu thức số học (+ - * / ** và ngoặc). Dùng cho mọi phép tính số liệu tài chính."


def safe_calc(expr: str) -> str:
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Mod: operator.mod}

    def ev(n):
        if isinstance(n, ast.Expression): return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            if isinstance(n.op, ast.Pow) and abs(ev(n.right)) > 10: raise ValueError("số mũ quá lớn")
            return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in ops: return ops[type(n.op)](ev(n.operand))
        raise ValueError("biểu thức không hợp lệ")
    try:
        return repr(round(ev(ast.parse(expr.strip(), mode="eval")), 6))
    except Exception as e:
        return f"Lỗi: {e}"


def format_context(hits):
    return "\n\n".join(f"[#{i}] (file: {c.doc}, trang {c.page}, {'BẢNG' if c.kind == 'table' else 'văn bản'})\n{c.text}"
                       for i, (c, _) in enumerate(hits, 1))


def answer(question, hits, history=None, max_steps=5):
    """Trả về (text, calc_log). history: list {role, content} các lượt trước (chỉ text)."""
    user = f"NGỮ CẢNH:\n{format_context(hits)}\n\nCÂU HỎI: {question}"
    msgs = list(history or []) + [{"role": "user", "content": user}]
    return (_anthropic if PROVIDER == "anthropic" else _openai)(msgs, max_steps)


def _anthropic(msgs, max_steps):
    import anthropic
    client, log = anthropic.Anthropic(), []
    tools = [{"name": "calculate", "description": TOOL_DESC,
              "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}]
    for _ in range(max_steps):
        r = client.messages.create(model=MODEL, max_tokens=1500, system=SYSTEM, tools=tools, messages=msgs)
        if r.stop_reason != "tool_use":
            return "".join(b.text for b in r.content if b.type == "text"), log
        msgs.append({"role": "assistant", "content": r.content})
        res = []
        for b in r.content:
            if b.type == "tool_use":
                out = safe_calc(b.input.get("expression", ""))
                log.append((b.input.get("expression", ""), out))
                res.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
        msgs.append({"role": "user", "content": res})
    return "Không hoàn tất được phép tính, vui lòng thử lại.", log


def _openai(msgs, max_steps):
    from openai import OpenAI
    client, log = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None), []
    tools = [{"type": "function", "function": {"name": "calculate", "description": TOOL_DESC,
              "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}]
    m = [{"role": "system", "content": SYSTEM}] + msgs
    for _ in range(max_steps):
        r = client.chat.completions.create(model=MODEL, messages=m, tools=tools, max_tokens=1500)
        msg = r.choices[0].message
        if not msg.tool_calls:
            return msg.content or "", log
        m.append(msg)
        for tc in msg.tool_calls:
            expr = json.loads(tc.function.arguments).get("expression", "")
            out = safe_calc(expr); log.append((expr, out))
            m.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    return "Không hoàn tất được phép tính, vui lòng thử lại.", log
