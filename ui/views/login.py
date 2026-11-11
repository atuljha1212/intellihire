from __future__ import annotations

import ipaddress
import time

import streamlit as st

from intellihire.config import get_settings
from intellihire.services.users import authenticate, login_lockout_seconds

MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 60


def client_ip_from_forwarded(header, hops: int) -> str | None:
    """Client IP from an X-Forwarded-For value (a string, or a list of repeated headers)
    when `hops` trusted proxies sit in front of the app: each proxy appends the address
    it saw, so the client is the entry `hops` places from the right. Entries further left
    are client-supplied and never used. None when hops <= 0, the header has fewer
    entries than hops, or that entry is not an IP address."""
    if hops <= 0 or not header:
        return None
    if not isinstance(header, str):
        header = ",".join(str(h) for h in header)
    entries = [e.strip() for e in header.split(",") if e.strip()]
    if len(entries) < hops:
        return None
    try:
        return str(ipaddress.ip_address(entries[-hops]))
    except ValueError:
        return None


def _client_ip() -> str | None:
    """The client's IP: from X-Forwarded-For when TRUSTED_PROXY_HOPS > 0 (falling back to
    the connection peer), otherwise the browser connection's IP. None on localhost or
    when Streamlit cannot tell."""
    try:
        ctx = getattr(st, "context", None)
        hops = get_settings().trusted_proxy_hops
        if hops > 0:
            headers = getattr(ctx, "headers", None)
            if headers is not None:
                get_all = getattr(headers, "get_all", None)
                xff = get_all("X-Forwarded-For") if callable(get_all) else headers.get("X-Forwarded-For")
                ip = client_ip_from_forwarded(xff, hops)
                if ip:
                    return ip
        ip = getattr(ctx, "ip_address", None)
    except Exception:
        return None
    return str(ip) if ip else None


def render() -> None:
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown("## 🧭 IntelliHire")
        st.markdown("<span class='ih-muted'>Explainable AI for recruitment decision support</span>",
                    unsafe_allow_html=True)
        st.write("")
        locked_until = st.session_state.get("locked_until", 0)
        if time.time() < locked_until:
            st.error(f"Too many failed attempts. Try again in {int(locked_until - time.time())} s.")
            return
        with st.form("login"):
            username = st.text_input("Username", autocomplete="username")
            password = st.text_input("Password", type="password", autocomplete="current-password")
            submitted = st.form_submit_button("Sign in", type="primary")
        if submitted:
            # Server-side throttle (per username+IP, per IP and per username, from the audit
            # log) - survives new browser sessions. The message never says which limit hit.
            ip = _client_ip()
            wait = login_lockout_seconds(username, ip) if username else 0
            if wait:
                st.error(f"Too many sign-in attempts. Try again in {wait} s.")
                return
            user = authenticate(username, password, ip) if username and password else None
            if user:
                st.session_state["user"] = user
                st.session_state.pop("failed_logins", None)
                st.rerun()
            failures = st.session_state.get("failed_logins", 0) + 1
            st.session_state["failed_logins"] = failures
            if failures >= MAX_ATTEMPTS:
                st.session_state["locked_until"] = time.time() + LOCKOUT_SECONDS
                st.session_state["failed_logins"] = 0
            st.error("Invalid username or password.")
        st.caption("First run? Sign in with the admin account configured in your `.env` "
                   "(`ADMIN_USERNAME` / `ADMIN_PASSWORD`).")
