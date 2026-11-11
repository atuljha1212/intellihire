"""Controlled agentic workflows for the recruiter assistant.

With an LLM configured, the model plans multi-step requests itself through
tool calling (bounded by AGENT_MAX_STEPS; tools are read-only and job-scoped).
Without one, a deterministic intent router runs the same tools and renders
the results as markdown - so the assistant is useful offline and its
behaviour is fully reproducible for testing.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from intellihire.agent.tools import RecruiterTools
from intellihire.config import get_settings
from intellihire.nlp.skills import extract_skills
from intellihire.rag.llm import LLMError, get_llm
from intellihire.rag.prompts import QUESTIONS_INSTRUCTIONS, SUMMARY_INSTRUCTIONS, SYSTEM_PROMPT, evidence_block
from intellihire.rag.retriever import retrieve
from intellihire.services.insights import ranking_rows

log = logging.getLogger(__name__)
_NAME_NOISE = {"candidate", "unknown", "resume", "the", "and", "for", "with", "mr.", "ms.", "dr."}
_NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "ten": 10, "2": 2, "3": 3, "4": 4, "5": 5, "10": 10}


@dataclass
class AgentResult:
    text: str
    trace: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    mode: str = "offline"


class RecruiterAgent:
    def __init__(self, job_id: int, blind: bool = False):
        self.job_id = job_id
        self.blind = blind
        self.tools = RecruiterTools(job_id, blind)

    # ------------------------------------------------------------ public API
    def run(self, query: str, history: list[dict] | None = None) -> AgentResult:
        llm = get_llm()
        if llm is not None:
            try:
                messages = self._history(history) + [{"role": "user", "content": query}]
                text, trace = llm.run_tools(SYSTEM_PROMPT + self._job_context(), messages, self.tools.specs(),
                                            self.tools.execute, get_settings().agent_max_steps)
                return AgentResult(text, trace, self.tools.evidence, f"{llm.provider}:{llm.model}")
            except LLMError as exc:
                log.warning("LLM failed, using offline mode: %s", exc)
                result = self._offline(query)
                result.text = f"> LLM unavailable ({exc}). Showing the offline analysis instead.\n\n" + result.text
                return result
        return self._offline(query)

    def _job_context(self) -> str:
        job = self.tools.get_job_requirements()
        mode = "BLIND MODE is on: refer to candidates only as 'Candidate C<id>'." if self.blind else ""
        return f"\n\nCurrent job: {job.get('title')} (job id {self.job_id}). {mode}"

    @staticmethod
    def _history(history: list[dict] | None) -> list[dict]:
        msgs = [{"role": m["role"], "content": m["content"]} for m in (history or [])[-8:]
                if m.get("role") in {"user", "assistant"} and m.get("content")]
        while msgs and msgs[0]["role"] != "user":
            msgs.pop(0)
        # keep strict user/assistant alternation (required by some providers)
        cleaned: list[dict] = []
        for m in msgs:
            if cleaned and cleaned[-1]["role"] == m["role"]:
                cleaned[-1]["content"] += "\n\n" + m["content"]
            else:
                cleaned.append(m)
        if cleaned and cleaned[-1]["role"] == "user":
            cleaned.pop()  # the new query is appended by the caller
        return cleaned

    # ------------------------------------------------------- offline router
    def _call(self, trace: list, name: str, **args) -> dict:
        out = self.tools.execute(name, args)
        trace.append({"tool": name, "args": args, "result": out[:4000]})
        return json.loads(out)

    def _mentioned(self, query: str) -> list[int]:
        ids: list[int] = []
        for ref in re.findall(r"#\d+|\bC\d+\b", query):
            cid = self.tools.resolve(ref)
            if cid and cid not in ids:
                ids.append(cid)
        rows = ranking_rows(self.job_id)
        lowered = query.lower()
        for r in rows:
            tokens = [t for t in re.split(r"\s+", r["name"].lower()) if len(t) > 2 and t not in _NAME_NOISE]
            if (r["name"].lower() in lowered or any(re.search(rf"\b{re.escape(t)}\b", lowered) for t in tokens)) \
                    and r["id"] not in ids:
                ids.append(r["id"])
        return ids

    @staticmethod
    def _top_n(query: str, default: int) -> int:
        m = re.search(r"\btop\s+(two|three|four|five|ten|\d+)\b", query.lower())
        if not m:
            return default
        return _NUMBER_WORDS.get(m.group(1), int(m.group(1)) if m.group(1).isdigit() else default)

    def _offline(self, query: str) -> AgentResult:
        q = query.lower()
        trace: list[dict] = []
        mentioned = self._mentioned(query)
        top_id = next((r["id"] for r in ranking_rows(self.job_id)), None)
        target = str(mentioned[0]) if mentioned else (str(top_id) if top_id else None)
        if top_id is None:
            return AgentResult("No candidates have been processed for this job yet. Upload resumes first.", trace)

        if "compare" in q or " vs " in q or "versus" in q:
            ranked = [r["id"] for r in ranking_rows(self.job_id)]
            if len(mentioned) >= 2:
                ids = mentioned
            elif mentioned:  # "compare Priya with the leader"
                ids = mentioned + [i for i in ranked if i not in mentioned][:1]
            else:
                ids = ranked[: max(2, min(self._top_n(query, 2), 4))]
            data = self._call(trace, "compare_candidates", candidates=[str(i) for i in ids[:4]])
            text = _render_comparison(data)
        elif re.search(r"interview|question", q):
            data = self._call(trace, "interview_question_plan", candidate=target)
            text = _render_questions(data)
        elif re.search(r"\bgap|missing|lack|weak", q):
            data = self._call(trace, "get_skill_gap", candidate=target)
            text = _render_gap(data)
        elif re.search(r"requirement|what does the job|job need|criteria", q):
            text = _render_requirements(self._call(trace, "get_job_requirements"))
        elif re.search(r"\bwho\b.*\b(know|has|have|with)|candidates? with|experience (in|with)|skilled in", q) \
                and extract_skills(query):
            skill = next(iter(extract_skills(query)))
            text = _render_skill_search(self._call(trace, "find_candidates_by_skill", skill=skill))
        elif re.search(r"\btop\b|\bbest\b|\brank|shortlist|strongest|leaderboard", q) and not mentioned:
            text = _render_top(self._call(trace, "list_top_candidates", limit=self._top_n(query, 5)))
        elif mentioned and re.search(r"summar|profile|tell me about|who is|overview|why", q):
            text = _render_profile(self._call(trace, "get_candidate_profile", candidate=str(mentioned[0])))
        else:
            data = self._call(trace, "search_evidence", query=query,
                              candidate=str(mentioned[0]) if mentioned else None, k=5)
            text = _render_evidence(data)
        return AgentResult(text, trace, self.tools.evidence, "offline")

    # ------------------------------------------------ grounded generation
    def summarize(self, candidate_id: int) -> AgentResult:
        trace: list[dict] = []
        profile = self._call(trace, "get_candidate_profile", candidate=str(candidate_id))
        if "error" in profile:
            return AgentResult(profile["error"], trace)
        gap = self._call(trace, "get_skill_gap", candidate=str(candidate_id))
        hits = retrieve(self.job_id, "key achievements, projects, impact and technical depth", [candidate_id],
                        k=6, blind=self.blind)
        llm = get_llm()
        if llm is None:
            return AgentResult(_render_profile(profile) + "\n\n" + _render_gap(gap), trace, hits)
        prompt = (f"{SUMMARY_INSTRUCTIONS}\n\nJOB REQUIREMENTS:\n{json.dumps(self.tools.get_job_requirements(), default=str)}"
                  f"\n\nSCORE BREAKDOWN & PROFILE (from IntelliHire):\n{json.dumps(profile, default=str)}"
                  f"\n\nSKILL GAP:\n{json.dumps(gap, default=str)}\n\nEVIDENCE:\n{evidence_block(hits)}")
        try:
            return AgentResult(llm.complete(SYSTEM_PROMPT, prompt), trace, hits, f"{llm.provider}:{llm.model}")
        except LLMError as exc:
            return AgentResult(f"> LLM unavailable ({exc}).\n\n" + _render_profile(profile), trace, hits)

    def interview_questions(self, candidate_id: int, focus: str | None = None) -> AgentResult:
        trace: list[dict] = []
        plan = self._call(trace, "interview_question_plan", candidate=str(candidate_id), focus=focus)
        if "error" in plan:
            return AgentResult(plan["error"], trace)
        llm = get_llm()
        if llm is None:
            return AgentResult(_render_questions(plan), trace, self.tools.evidence)
        prompt = f"{QUESTIONS_INSTRUCTIONS}\n\nPLAN DATA:\n{json.dumps(plan, default=str)}"
        try:
            return AgentResult(llm.complete(SYSTEM_PROMPT, prompt), trace, self.tools.evidence,
                               f"{llm.provider}:{llm.model}")
        except LLMError as exc:
            return AgentResult(f"> LLM unavailable ({exc}).\n\n" + _render_questions(plan), trace, self.tools.evidence)


# =================================================================== renderers
def _fmt(v) -> str:
    return "-" if v is None else (f"{v:.0f}" if isinstance(v, (int, float)) else str(v))


def _render_top(data: dict) -> str:
    rows = data.get("candidates", [])
    if not rows:
        return "No candidates match."
    lines = [f"**Top {len(rows)} of {data.get('total_candidates')} candidates** (score out of 100)\n",
             "| # | Candidate | Score | Semantic | Skills | Exp. | Video | Req. skills | Status |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        flag = " ⚠" if r["flags"] else ""
        lines.append(f"| {r['rank']} | {r['candidate']}{flag} | **{r['final_score']:.1f}** | {_fmt(r['semantic'])} | "
                     f"{_fmt(r['skill'])} | {_fmt(r['experience'])} | {_fmt(r['video'])} | "
                     f"{r['required_skills_matched']} | {r['status']} |")
    best = rows[0]
    lines.append(f"\n**Why #{best['rank']} leads:** {best['required_skills_matched']} required skills, "
                 f"{best['years']:g} yrs experience, semantic match {_fmt(best['semantic'])}/100.")
    if best["missing_required"]:
        lines.append(f"Still missing: {', '.join(best['missing_required'])}.")
    if any(r["flags"] for r in rows):
        lines.append("\n⚠ = resume contains text flagged by the prompt-injection guard - review before deciding.")
    return "\n".join(lines)


def _render_profile(p: dict) -> str:
    if "error" in p:
        return p["error"]
    s = p["sub_scores_pct"]
    lines = [f"### {p['candidate']} - rank #{p['rank']}, score {p['final_score']:.1f}/100",
             f"- **Sub-scores:** semantic {_fmt(s['semantic'])}, skills {_fmt(s['skill'])}, experience "
             f"{_fmt(s['experience'])}, video {_fmt(s['video'])}, communication {_fmt(s['communication'])}",
             f"- **Experience:** {p['experience_years']:g} years (from {p['experience_basis']})",
             f"- **Skills:** {', '.join(p['skills'][:20]) or 'none detected'}"]
    if p["education"]:
        lines.append(f"- **Education:** {p['education'][0]}")
    if p["certifications"]:
        lines.append(f"- **Certifications:** {'; '.join(p['certifications'][:4])}")
    if p.get("ml_score") is not None:
        lines.append(f"- **Learned-ranker score:** {p['ml_score']:.0f}/100")
    lines.append("\n**How the score was built:**")
    lines += [f"- {r}" for r in p.get("score_reasons") or []]
    if p["security_flags"]:
        lines.append(f"\n⚠ **Security flags:** {', '.join(p['security_flags'])}")
    return "\n".join(lines)


def _render_gap(g: dict) -> str:
    if "error" in g:
        return g["error"]
    lines = [f"### Skill gap - {g['candidate']}"]
    lines.append(f"- ✅ **Required, present:** {', '.join(g['matched_required']) or 'none'}")
    lines.append(f"- ❌ **Required, missing:** {', '.join(g['missing_required']) or 'none'}")
    lines.append(f"- ✅ **Preferred, present:** {', '.join(g['matched_preferred']) or 'none'}")
    lines.append(f"- ➖ **Preferred, missing:** {', '.join(g['missing_preferred']) or 'none'}")
    if g["mentioned_only_in_video"]:
        lines.append(f"- 🎥 **Only mentioned in video:** {', '.join(g['mentioned_only_in_video'])}")
    shown = list(g["evidence_lines"].items())[:4]
    if shown:
        lines.append("\n**Evidence from the resume:**")
        lines += [f"- *{skill}*: “{lines_[0]}”" for skill, lines_ in shown if lines_]
    return "\n".join(lines)


def _render_comparison(d: dict) -> str:
    ps = d.get("candidates", [])
    if len(ps) < 2:
        return ("I need at least two candidates to compare. "
                + (f"Not found: {d.get('not_found')}" if d.get("not_found") else "")
                + (" " + " ".join(d["errors"]) if d.get("errors") else ""))
    head = "| Metric | " + " | ".join(p["candidate"] for p in ps) + " |"
    sep = "|---" * (len(ps) + 1) + "|"
    rows = [("Rank", lambda p: _fmt(p["rank"])), ("Final score", lambda p: f"{p['final_score']:.1f}"),
            ("Semantic", lambda p: _fmt(p["sub_scores_pct"]["semantic"])),
            ("Skills", lambda p: _fmt(p["sub_scores_pct"]["skill"])),
            ("Experience fit", lambda p: _fmt(p["sub_scores_pct"]["experience"])),
            ("Years", lambda p: f"{p['experience_years']:g}"),
            ("Video", lambda p: _fmt(p["sub_scores_pct"]["video"])),
            ("Communication", lambda p: _fmt(p["sub_scores_pct"]["communication"])),
            ("Flags", lambda p: str(len(p["security_flags"])))]
    lines = ["### Candidate comparison", head, sep]
    lines += [f"| {label} | " + " | ".join(fn(p) for p in ps) + " |" for label, fn in rows]
    if d.get("shared_skills"):
        lines.append(f"\n**Shared skills:** {', '.join(d['shared_skills'][:15])}")
    for name, skills in (d.get("unique_skills") or {}).items():
        if skills:
            lines.append(f"- **Only {name}:** {', '.join(skills[:12])}")
    leader = max(ps, key=lambda p: p["final_score"] or 0)
    lines.append(f"\n**Edge:** {leader['candidate']} on overall score - check the gaps above before deciding.")
    return "\n".join(lines)


def _render_questions(plan: dict) -> str:
    if "error" in plan:
        return plan["error"]
    qs = []
    for skill in plan.get("missing_required", [])[:3]:
        qs.append((f"The role needs **{skill}**, which isn't evident in your resume. What exposure have you had to it, "
                   f"and how would you get productive with it in your first month?",
                   "Honest self-assessment and a concrete learning plan."))
    seen: set[str] = set()
    for ev in plan.get("claims_to_verify", [])[:3]:
        snippet = _claim_line(ev["text"])
        if snippet and snippet not in seen:
            seen.add(snippet)
            qs.append((f"You wrote: “{snippet}” [{ev['ref']}]. What exactly was your contribution, and what "
                       f"was the hardest technical decision?", "Ownership, depth and trade-off reasoning."))
    deep_skills = [s for s in plan.get("matched_required", []) if s not in _GENERIC_SKILLS]
    for skill in deep_skills[:2]:
        qs.append((f"Describe a production problem you solved with **{skill}**. How did you measure success?",
                   "Real-world depth beyond keyword familiarity."))
    req, have = plan.get("required_years"), plan.get("experience_years") or 0
    if req and have < req:
        qs.append((f"The role asks for {req:g}+ years; your resume shows about {have:g}. Which experiences best prepare you "
                   f"for the scope of this role?", "Transferable experience and self-awareness."))
    resp = [r.strip() for r in (plan.get("responsibilities_excerpt") or "").split("\n") if len(r.strip()) > 20]
    task = next((r for r in resp if r.startswith("-")), resp[0] if resp else "").strip("- ").strip()
    if task:
        qs.append((f"Scenario: “{_clip(task, 130)}” - walk us through how you would approach this in your first 90 days.",
                   "Structured thinking applied to the actual job."))
    lines = [f"### Interview questions - {plan['candidate']} ({plan['job_title']})"]
    for i, (q, why) in enumerate(qs[:8], start=1):
        lines.append(f"{i}. {q}\n   - *Strong answer shows:* {why}")
    return "\n".join(lines)


_GENERIC_SKILLS = {"Git", "SQL", "Python", "Linux", "Excel", "Agile", "Communication", "Teamwork", "Leadership",
                   "Problem Solving", "HTML", "CSS"}


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def _claim_line(chunk: str) -> str:
    """Most claim-like line of an evidence chunk: substantive bullets first, never the contact header."""
    lines = [l.strip() for l in chunk.split("\n") if "[removed" not in l and "[email]" not in l and "[phone]" not in l]
    substantive = [l for l in lines if len(l.split()) >= 6]
    bullets = [l for l in substantive if l.startswith("-")]
    best = (bullets or substantive or [""])[0]
    return _clip(best.lstrip("- "), 140)


def _render_requirements(j: dict) -> str:
    exp = "not specified" if j.get("min_years") is None else (
        f"{j['min_years']:g}-{j['max_years']:g} years" if j.get("max_years") else f"{j['min_years']:g}+ years")
    return (f"### {j['title']}\n- **Required skills:** {', '.join(j['required_skills']) or '-'}\n"
            f"- **Preferred skills:** {', '.join(j['preferred_skills']) or '-'}\n- **Experience:** {exp}\n"
            f"- **Scoring weights:** " + ", ".join(f"{k} {v:.2f}" for k, v in (j.get('weights') or {}).items()))


def _render_skill_search(d: dict) -> str:
    matches = d.get("matches", [])
    if not matches:
        return f"No candidate shows **{d['skill']}** in their resume or video."
    lines = [f"### Candidates with {d['skill']} ({len(matches)})"]
    for m in matches[:15]:
        if "final_score" in m:
            lines.append(f"- #{m['rank']} {m['candidate']} - score {m['final_score']:.1f}")
        else:
            lines.append(f"- {m['candidate']} ({m['source']}): “{m['snippet']}”")
    return "\n".join(lines)


def _render_evidence(d: dict) -> str:
    ev = d.get("evidence", [])
    if not ev:
        return d.get("error") or "I couldn't find relevant evidence in the candidates' documents."
    lines = [f"**Most relevant evidence for:** _{d['query']}_ (offline mode shows retrieved passages verbatim)\n"]
    for e in ev:
        where = f"{e['source']}" + (f", {e['time']}" if e.get("time") else "")
        text = e["text"].replace("\n", " ")
        lines.append(f"- **[{e['ref']}] {e['candidate']}** ({where}, relevance {e['relevance']:.2f}): {text[:300]}")
    return "\n".join(lines)
