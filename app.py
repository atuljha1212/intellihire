"""IntelliHire - Explainable AI for Recruitment Decision Support.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import logging

import streamlit as st

st.set_page_config(page_title="IntelliHire", page_icon="🧭", layout="wide", initial_sidebar_state="expanded")

from intellihire.db.session import init_db  # noqa: E402
from intellihire.services.users import uses_default_password  # noqa: E402
from ui.common import current_user, inject_css, is_admin  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@st.cache_resource(show_spinner="Starting IntelliHire: preparing database and models…")
def bootstrap() -> bool:
    init_db()
    # Warm the heavy singletons once per server process.
    from intellihire.embeddings.encoder import get_encoder
    from intellihire.nlp.spacy_pipeline import get_nlp
    from intellihire.rag.vector_store import get_vector_store

    get_encoder()
    get_nlp()
    get_vector_store()
    return True


bootstrap()
inject_css()

from ui.pages import page  # noqa: E402

user = current_user()
if user is None:
    nav = st.navigation([page("login")], position="hidden")
else:
    with st.sidebar:
        st.markdown(f"**{user['full_name']}**  \n<span class='ih-muted'>{user['role']}</span>", unsafe_allow_html=True)
        st.toggle("Blind screening", key="blind_mode",
                  help="Hide names and contact details in rankings, profiles and the assistant to reduce bias.")
        if st.button("Log out"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    if is_admin() and uses_default_password(user["id"]):
        # Bootstrap password still in use: only the Admin page is reachable until it is changed.
        st.warning("**Security:** this admin account still uses the default password. "
                   "Set a new password for it under **Admin › Users › Manage user** to unlock the rest of the app.",
                   icon="⚠️")
        sections = {"Administration": [page("admin")]}
    else:
        sections = {
            "Overview": [page("dashboard")],
            "Hiring": [page("jobs"), page("upload"), page("rankings"), page("candidate")],
            "Insights": [page("assistant"), page("analytics")],
        }
        if is_admin():
            sections["Administration"] = [page("admin")]
    nav = st.navigation(sections)

nav.run()
