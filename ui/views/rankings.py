from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from intellihire.services.insights import latest_ranker, ranking_rows
from intellihire.services.pipeline import STATUSES, set_status
from ui.common import blind_mode, current_user, display_name, job_picker, score_color

RANK_MODES = {
    "Transparent formula": "final",
    "Learned from my decisions (ML)": "ml",
    "Blend (50/50)": "blend",
}


def _table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["display"] = [display_name(i, n) for i, n in zip(df["id"], df["name"])]
    df["req"] = [f"{m}/{t}" if t else "—" for m, t in zip(df["matched"], df["required_total"])]
    df["missing_txt"] = [", ".join(m[:5]) + ("…" if len(m) > 5 else "") for m in df["missing"]]
    df["flag"] = ["⚠" if f else "" for f in df["flags"]]
    df["video_icon"] = ["🎥" if v else "" for v in df["has_video"]]
    df["blend"] = [(f + (m if m is not None else f)) / 2 for f, m in zip(df["final"], df["ml"])]
    return df


def render() -> None:
    _msg = st.session_state.pop("_flash_rankings", None)
    if _msg:
        st.toast(_msg)
    st.title("Rankings")
    job = job_picker("rank_job")
    if not job:
        return
    rows = ranking_rows(job["id"])
    if not rows:
        st.info("No candidates yet - upload resumes on the **Upload candidates** page.")
        return
    df = _table(rows)

    model = latest_ranker(job["id"])
    c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1.2])
    modes = list(RANK_MODES) if model else list(RANK_MODES)[:1]
    mode_label = c1.selectbox("Rank by", modes, help="Train the learned ranker on the Analytics page to enable ML ranking.")
    min_score = c2.slider("Min. score", 0, 100, 0, 5)
    status_filter = c3.multiselect("Status", STATUSES, default=[s for s in STATUSES if s != "rejected"])
    skill_filter = c4.multiselect("Must have skills", sorted({s for r in rows for s in r["skills"]}))
    c5, c6 = st.columns([1, 3])
    video_only = c5.checkbox("Has video")
    flagged_only = c6.checkbox("Only flagged resumes")

    sort_key = RANK_MODES[mode_label]
    view = df.copy()
    view["sort_value"] = view[sort_key].fillna(view["final"])
    view = view[(view["sort_value"] >= min_score) & (view["status"].isin(status_filter or STATUSES))]
    if skill_filter:
        view = view[view["skills"].apply(lambda s: set(skill_filter).issubset(s))]
    if video_only:
        view = view[view["has_video"]]
    if flagged_only:
        view = view[view["flags"] > 0]
    view = view.sort_values("sort_value", ascending=False)
    view["position"] = range(1, len(view) + 1)

    _top_cards(view.head(5), sort_key)
    st.caption(f"Showing {len(view)} of {len(df)} candidates. Edit the **Status** column to shortlist or reject; "
               "changes are saved when you click *Save status changes*.")

    shown = view[["id", "position", "display", "sort_value", "final", "ml", "semantic", "skill", "experience", "video",
                  "communication", "years", "req", "missing_txt", "video_icon", "flag", "status"]].set_index("id")
    edited = st.data_editor(
        shown,
        hide_index=True,
        disabled=[c for c in shown.columns if c != "status"],
        column_config={
            "position": st.column_config.NumberColumn("#", width="small"),
            "display": st.column_config.TextColumn("Candidate", width="medium"),
            "sort_value": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
            "final": st.column_config.NumberColumn("Formula", format="%.1f"),
            "ml": st.column_config.NumberColumn("ML", format="%.0f"),
            "semantic": st.column_config.NumberColumn("Semantic", format="%.0f"),
            "skill": st.column_config.NumberColumn("Skills", format="%.0f"),
            "experience": st.column_config.NumberColumn("Exp. fit", format="%.0f"),
            "video": st.column_config.NumberColumn("Video", format="%.0f"),
            "communication": st.column_config.NumberColumn("Comm.", format="%.0f"),
            "years": st.column_config.NumberColumn("Years", format="%.1f"),
            "req": st.column_config.TextColumn("Req. skills"),
            "missing_txt": st.column_config.TextColumn("Missing required", width="medium"),
            "video_icon": st.column_config.TextColumn("🎥", width="small"),
            "flag": st.column_config.TextColumn("⚠", width="small", help="Hidden or instruction-like text detected"),
            "status": st.column_config.SelectboxColumn("Status", options=STATUSES, required=True),
        },
        key=f"rank_editor_{job['id']}",
    )

    changes = edited["status"].compare(shown["status"]) if len(edited) else pd.DataFrame()
    b1, b2, b3, b4 = st.columns([1.3, 1.2, 1, 2])
    if b1.button("Save status changes", type="primary", disabled=changes.empty):
        for cid, row in changes.iterrows():
            set_status(current_user()["id"], [int(cid)], row["self"])
        st.session_state["_flash_rankings"] = f"Updated {len(changes)} candidate(s)."
        st.rerun()
    labels = {int(i): f"#{p}  {d}" for i, p, d in zip(view["id"], view["position"], view["display"])}
    if labels:
        chosen = b2.selectbox("Open profile", list(labels), format_func=lambda i: labels[i], label_visibility="collapsed")
        if b3.button("Open profile"):
            from ui.pages import page  # local import avoids a circular import

            st.session_state["candidate_id"] = chosen
            st.switch_page(page("candidate"))
    export = view.drop(columns=["skills", "missing", "sort_value"]).copy()
    if blind_mode():
        export = export.drop(columns=["name", "email"])
    b4.download_button("Export CSV", export.to_csv(index=False).encode(), f"rankings_job_{job['id']}.csv", "text/csv")


def _top_cards(top: pd.DataFrame, sort_key: str) -> None:
    if top.empty:
        return
    st.markdown("#### Top candidates")
    cols = st.columns(len(top))
    for col, (_, r) in zip(cols, top.iterrows()):
        score = r["sort_value"]
        missing = r["missing"][:3]
        missing_html = ("<div class='ih-muted'>Missing: " + escape(", ".join(missing)) + "</div>") if missing else \
            "<div class='ih-muted'>All required skills ✓</div>"
        col.markdown(
            f"<div class='ih-card'><div class='rank'>#{r['position']} {r['video_icon']} {r['flag']}</div>"
            f"<div class='name'>{escape(r['display'])}</div>"
            f"<div class='score' style='color:{score_color(score)}'>{score:.1f}</div>"
            f"<div class='ih-muted'>{r['req']} req. skills · {r['years']:g} yrs</div>{missing_html}</div>",
            unsafe_allow_html=True,
        )
