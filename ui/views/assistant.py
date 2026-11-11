from __future__ import annotations

import json
from html import escape

import streamlit as st

from intellihire.agent.agent import RecruiterAgent
from intellihire.rag.llm import llm_status
from intellihire.services.insights import clear_chat, load_chat, save_chat
from ui.common import blind_mode, current_user, job_picker

QUICK_ACTIONS = [
    "Who are the top 5 candidates and why?",
    "Compare the top 2 candidates",
    "What are the skill gaps of #1?",
    "Write interview questions for #1",
]


def _redact_history(job_id: int, history: list[dict]) -> list[dict]:
    """Blind mode: strip candidate names and contact details from stored chat before display or LLM use."""
    from intellihire.rag.guard import redact_pii
    from intellihire.rag.retriever import redact_name
    from intellihire.services.pipeline import candidate_options, get_candidate

    names = [(get_candidate(cid) or {}).get("name") or "" for cid, _ in candidate_options(job_id)]

    def clean(text: str) -> str:
        text = redact_pii(text or "")
        for n in names:
            text = redact_name(text, n)
        return text

    out = []
    for m in history:
        m = dict(m)
        m["content"] = clean(m.get("content", ""))
        m.pop("trace", None)  # tool results contain names
        out.append(m)
    return out


def render() -> None:
    st.title("AI assistant")
    job = job_picker("assist_job")
    if not job:
        return
    user = current_user()
    status = llm_status()
    st.caption(f"Model: **{status}** · grounded in this job's resumes and transcripts · "
               + ("blind mode on" if blind_mode() else "blind mode off"))
    if status.startswith("offline"):
        st.info("No LLM API key configured, so the assistant runs its deterministic agent: it uses the same tools "
                "(ranking, skill-gap, comparison, semantic evidence search) and shows results as tables. "
                "Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in `.env` for free-form answers.")

    history = load_chat(user["id"], job["id"])
    if blind_mode():
        history = _redact_history(job["id"], history)
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("trace"):
                _trace(msg["trace"])

    cols = st.columns(len(QUICK_ACTIONS) + 1)
    clicked = None
    for col, action in zip(cols, QUICK_ACTIONS):
        if col.button(action, key=f"qa_{action}"):
            clicked = action
    if cols[-1].button("🗑 Clear chat"):
        clear_chat(user["id"], job["id"])
        st.rerun()

    prompt = st.chat_input("Ask about candidates, e.g. 'Who has Kubernetes and more than 3 years experience?'")
    query = clicked or prompt
    if not query:
        return
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Thinking - retrieving evidence and calling tools…"):
            result = RecruiterAgent(job["id"], blind=blind_mode()).run(query, history)
        st.markdown(result.text)
        _trace(result.trace)
    save_chat(user["id"], job["id"], "user", query)
    save_chat(user["id"], job["id"], "assistant", result.text, result.trace)


def _trace(trace: list[dict]) -> None:
    if not trace:
        return
    with st.expander(f"Agent steps ({len(trace)} tool call{'s' if len(trace) != 1 else ''})"):
        for i, step in enumerate(trace, start=1):
            args = json.dumps(step.get("args", {}), ensure_ascii=False)
            st.markdown(f"**{i}. `{step['tool']}`** <span class='ih-muted'>{escape(args)}</span>", unsafe_allow_html=True)
            try:
                st.json(json.loads(step.get("result", "{}")), expanded=False)
            except json.JSONDecodeError:
                st.code(step.get("result", "")[:2000])
