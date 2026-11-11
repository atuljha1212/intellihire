from __future__ import annotations

import pandas as pd
import streamlit as st

from intellihire.services.pipeline import EmbeddingModelUnavailable, rebuild_vector_index
from intellihire.services.system import component_statuses
from intellihire.services.users import (
    UserError,
    change_password,
    create_user,
    list_users,
    recent_audit,
    set_user_active,
)
from ui.common import current_user, is_admin


def render() -> None:
    _msg = st.session_state.pop("_flash_admin", None)
    if _msg:
        st.toast(_msg)
    if not is_admin():
        st.error("Admins only.")
        return
    st.title("Admin")
    tabs = st.tabs(["System status", "Users", "Maintenance", "Audit log"])
    with tabs[0]:
        _status()
    with tabs[1]:
        _users()
    with tabs[2]:
        _maintenance()
    with tabs[3]:
        rows = recent_audit()
        if rows:
            df = pd.DataFrame(rows)
            df["detail"] = df["detail"].astype(str)
            st.dataframe(df, hide_index=True)
        else:
            st.caption("No activity yet.")


def _status() -> None:
    for c in component_statuses():
        icon = "🟢" if c.ok and not c.degraded else ("🟠" if c.ok else "🔴")
        st.markdown(f"{icon} **{c.name}** - {c.detail}")
    st.caption("🟠 = running in a fallback mode. See README › Quick start (local) to enable the full stack.")


def _users() -> None:
    me = current_user()
    users = list_users()
    st.dataframe(pd.DataFrame(users)[["id", "username", "full_name", "role", "is_active", "last_login"]], hide_index=True)
    c1, c2 = st.columns(2)
    with c1, st.form("new_user", clear_on_submit=True):
        st.markdown("**Create user**")
        username = st.text_input("Username")
        full_name = st.text_input("Full name")
        email = st.text_input("Email")
        role = st.selectbox("Role", ["recruiter", "admin"])
        password = st.text_input("Temporary password", type="password")
        if st.form_submit_button("Create", type="primary"):
            try:
                create_user(me["id"], username, password, role, full_name, email)
                st.session_state["_flash_admin"] = f"Created {username}"
                st.rerun()
            except UserError as exc:
                st.error(str(exc))
    with c2:
        st.markdown("**Manage user**")
        by_id = {u["id"]: u for u in users}
        uid = st.selectbox("User", list(by_id), format_func=lambda i: by_id[i]["username"])
        target = by_id[uid]
        if st.button("Deactivate" if target["is_active"] else "Activate"):
            try:
                set_user_active(me["id"], uid, not target["is_active"])
                st.rerun()
            except UserError as exc:
                st.error(str(exc))
        new_pw = st.text_input("New password", type="password", key="reset_pw")
        if st.button("Set password"):
            try:
                change_password(me["id"], uid, new_pw)
                # rerun so the navigation is rebuilt (unlocks the app after replacing the default password)
                st.session_state["_flash_admin"] = "Password updated"
                st.rerun()
            except UserError as exc:
                st.error(str(exc))


def _maintenance() -> None:
    st.markdown("**Vector index** - rebuild FAISS from the chunk embeddings stored in the database "
                "(use after restoring a database backup or if the index file was deleted).")
    if st.button("Rebuild vector index"):
        try:
            with st.spinner("Rebuilding…"):
                n = rebuild_vector_index(reembed=False)
            st.success(f"Indexed {n} chunks.")
        except EmbeddingModelUnavailable as exc:
            st.error(str(exc))
    st.markdown("**Re-embed everything** - required after changing `EMBEDDING_MODEL` (or once the real "
                "Sentence Transformer becomes available after running in fallback mode). Re-scores every job.")
    if st.button("Re-embed all documents"):
        bar = st.progress(0.0)
        try:
            n = rebuild_vector_index(reembed=True, progress=lambda i, total, _: bar.progress(min(1.0, i / max(total, 1))))
            st.success(f"Re-embedded {n} chunks and re-scored all jobs.")
        except EmbeddingModelUnavailable as exc:
            st.error(str(exc))
