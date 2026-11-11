"""Tools the recruiter agent can call. Every tool is read-only and scoped to a
single job, so the agent can analyse but never modify hiring data.

All candidate-originated text returned by tools is sanitised (instruction-like
sentences removed, contact details redacted)."""
from __future__ import annotations

import json
import re

from intellihire.nlp.skills import extract_skills
from intellihire.rag.guard import sanitize_for_llm
from intellihire.rag.llm import ToolSpec
from intellihire.rag.retriever import candidate_label, redact_name, retrieve
from intellihire.services.insights import ranking_rows
from intellihire.services.pipeline import find_candidate_ids, get_candidate, get_job

_CANDIDATE_REF = {"type": "string", "description": "Candidate name, id (e.g. '12' or 'C12') or rank (e.g. '#1')."}


class RecruiterTools:
    def __init__(self, job_id: int, blind: bool = False):
        self.job_id = job_id
        self.blind = blind
        self.evidence: list[dict] = []  # every evidence chunk shown to the model, numbered E1..En

    # ---------------------------------------------------------------- specs
    @staticmethod
    def specs() -> list[ToolSpec]:
        return [
            ToolSpec("get_job_requirements", "Get the job's title, required/preferred skills, experience range and scoring weights.",
                     {"type": "object", "properties": {}}),
            ToolSpec("list_top_candidates", "List the highest-ranked candidates with their scores and sub-scores.",
                     {"type": "object", "properties": {
                         "limit": {"type": "integer", "description": "How many candidates (default 5, max 25)."},
                         "status": {"type": "string", "description": "Optional status filter: new, shortlisted, interview, rejected, hired."}}}),
            ToolSpec("get_candidate_profile", "Structured profile and full score breakdown of one candidate.",
                     {"type": "object", "properties": {"candidate": _CANDIDATE_REF}, "required": ["candidate"]}),
            ToolSpec("get_skill_gap", "Matched and missing required/preferred skills for one candidate, with evidence lines.",
                     {"type": "object", "properties": {"candidate": _CANDIDATE_REF}, "required": ["candidate"]}),
            ToolSpec("compare_candidates", "Side-by-side comparison of 2-4 candidates on every scoring feature and skill.",
                     {"type": "object", "properties": {"candidates": {"type": "array", "items": _CANDIDATE_REF, "minItems": 2, "maxItems": 4}},
                      "required": ["candidates"]}),
            ToolSpec("search_evidence", "Semantic search over resumes and video transcripts. Returns cited evidence chunks [E#].",
                     {"type": "object", "properties": {
                         "query": {"type": "string", "description": "What to look for, in natural language."},
                         "candidate": {"type": "string", "description": "Optional: restrict to one candidate (name, id or #rank)."},
                         "k": {"type": "integer", "description": "Number of chunks (default 5, max 10)."}},
                      "required": ["query"]}),
            ToolSpec("find_candidates_by_skill", "Find candidates whose resume (or video) shows a given skill.",
                     {"type": "object", "properties": {"skill": {"type": "string"}}, "required": ["skill"]}),
            ToolSpec("interview_question_plan", "Facts needed to write tailored interview questions: gaps, claims to verify, evidence.",
                     {"type": "object", "properties": {"candidate": _CANDIDATE_REF,
                                                       "focus": {"type": "string", "description": "Optional topic to focus on."}},
                      "required": ["candidate"]}),
        ]

    # ------------------------------------------------------------- helpers
    def _label(self, cid: int, name: str) -> str:
        return candidate_label(cid, name, self.blind)

    def _clean(self, text: str, name: str) -> str:
        """Sanitise candidate text; in blind mode also strip the candidate's name."""
        text = sanitize_for_llm(text)
        return redact_name(text, name) if self.blind else text

    def lookup(self, ref) -> tuple[int | None, str | None]:
        """Resolve a candidate reference to one id, or return an error message.

        An empty reference is rejected (it is a substring of every name), and a
        reference matching several candidates is reported as ambiguous instead of
        silently picking one."""
        ref = str(ref or "").strip()
        if not ref:
            return None, "No candidate given - pass a name, id (e.g. 'C12') or rank (e.g. '#1')."
        if re.fullmatch(r"[Cc]\d+", ref):
            ref = ref[1:]
        ids = list(dict.fromkeys(find_candidate_ids(self.job_id, ref)))
        if not ids:
            return None, f"No candidate matching '{ref}' in this job."
        if len(ids) > 1:
            options = []
            for cid in ids[:6]:
                c = get_candidate(cid)
                options.append(self._label(cid, c["name"]) if c else f"C{cid}")
            return None, (f"'{ref}' is ambiguous - it matches {len(ids)} candidates: {', '.join(options)}. "
                          "Use the candidate id (e.g. 'C12') instead.")
        return ids[0], None

    def resolve(self, ref: str) -> int | None:
        return self.lookup(ref)[0]

    def _add_evidence(self, items: list[dict]) -> list[dict]:
        out = []
        for e in items:
            e = dict(e)
            e["ref"] = f"E{len(self.evidence) + 1}"
            self.evidence.append(e)
            out.append({"ref": e["ref"], "candidate": e["candidate"], "source": e["source"],
                        "relevance": e["score"], "text": e["text"],
                        **({"time": f'{e["start_sec"]:.0f}s-{e["end_sec"]:.0f}s'} if e.get("start_sec") is not None else {})})
        return out

    @staticmethod
    def _pct(value) -> float | None:
        return None if value is None else round(value * 100, 1)

    def _profile(self, cid: int) -> dict | None:
        c = get_candidate(cid)
        if c is None or c["job_id"] != self.job_id:
            return None
        sc = c["score"] or {}
        details = sc.get("details", {}) or {}
        video = c["video"]
        return {
            "candidate": self._label(c["id"], c["name"]),
            "rank": sc.get("rank"),
            "final_score": sc.get("final"),
            "ml_score": sc.get("ml"),
            "sub_scores_pct": {k: self._pct(sc.get(k)) for k in ("semantic", "skill", "experience", "video", "communication")},
            "weights_used": details.get("weights_used"),
            "points_contributed": details.get("contributions"),
            "score_reasons": details.get("reasons"),
            "experience_years": c["experience_years"],
            "experience_basis": c["experience_method"],
            "education": [self._clean(e, c["name"]) for e in c["education"]],
            "certifications": [self._clean(x, c["name"]) for x in c["certifications"]],
            "skills": sorted(c["skills"]),
            "status": c["status"],
            "has_video": video is not None,
            "video_metrics": None if video is None else {
                "duration_sec": video["duration"], "words_per_minute": video["wpm"],
                "vocabulary_diversity": video["mattr"], "filler_rate": video["filler_rate"],
                "skills_mentioned": video["video_skills"]},
            "security_flags": [f"{f['kind']} ({f['location']})" for f in c["injection_flags"]],
        }

    # ---------------------------------------------------------------- tools
    def get_job_requirements(self) -> dict:
        j = get_job(self.job_id) or {}
        return {k: j.get(k) for k in ("title", "department", "required_skills", "preferred_skills",
                                      "min_years", "max_years", "education_req", "weights")} | {
            "responsibilities": (j.get("responsibilities") or "")[:1500]}

    def list_top_candidates(self, limit: int = 5, status: str | None = None) -> dict:
        limit = max(1, min(int(limit or 5), 25))
        rows = [r for r in ranking_rows(self.job_id) if not status or r["status"] == status]
        return {"total_candidates": len(rows), "candidates": [
            {"rank": r["rank"], "candidate": self._label(r["id"], r["name"]), "final_score": round(r["final"], 1),
             "ml_score": r["ml"], "semantic": round(r["semantic"]), "skill": None if r["skill"] is None else round(r["skill"]),
             "experience": round(r["experience"]), "video": None if r["video"] is None else round(r["video"]),
             "years": r["years"], "required_skills_matched": f'{r["matched"]}/{r["required_total"]}',
             "missing_required": r["missing"][:6], "status": r["status"], "flags": r["flags"]}
            for r in rows[:limit]]}

    def get_candidate_profile(self, candidate: str) -> dict:
        cid, error = self.lookup(candidate)
        if error:
            return {"error": error}
        profile = self._profile(cid)
        return profile or {"error": f"No candidate matching '{candidate}' in this job."}

    def get_skill_gap(self, candidate: str) -> dict:
        cid, error = self.lookup(candidate)
        if error:
            return {"error": error}
        c = get_candidate(cid)
        if not c or c["job_id"] != self.job_id or not c["score"]:
            return {"error": f"No scored candidate matching '{candidate}'."}
        d = c["score"]["details"]
        evidence = {s: [self._clean(x, c["name"]) for x in c["skills"].get(s, [])[:2]]
                    for s in d.get("matched_required", []) + d.get("matched_preferred", [])}
        return {"candidate": self._label(c["id"], c["name"]),
                "matched_required": d.get("matched_required", []), "missing_required": d.get("missing_required", []),
                "matched_preferred": d.get("matched_preferred", []), "missing_preferred": d.get("missing_preferred", []),
                "mentioned_only_in_video": d.get("video_only_skills", []), "evidence_lines": evidence,
                "skill_score_pct": self._pct(c["score"]["skill"])}

    def compare_candidates(self, candidates: list[str]) -> dict:
        profiles, missing, problems = [], [], []
        for ref in (candidates or [])[:4]:
            cid, error = self.lookup(ref)
            p = self._profile(cid) if cid else None
            (profiles.append(p) if p else missing.append(ref))
            if error and len(problems) < 4:
                problems.append(error)
        result: dict = {"candidates": profiles}
        if len(profiles) >= 2:
            skill_sets = [set(p["skills"]) for p in profiles]
            result["shared_skills"] = sorted(set.intersection(*skill_sets))
            result["unique_skills"] = {p["candidate"]: sorted(s - set.union(*(o for o in skill_sets if o is not s)))
                                       for p, s in zip(profiles, skill_sets)}
        if missing:
            result["not_found"] = missing
        if problems:
            result["errors"] = problems
        return result

    def search_evidence(self, query: str, candidate: str | None = None, k: int = 5) -> dict:
        k = max(1, min(int(k or 5), 10))
        ids = None
        if candidate:
            cid, error = self.lookup(candidate)
            if error:
                return {"error": error}
            ids = [cid]
        hits = retrieve(self.job_id, query, ids, k=k, blind=self.blind)
        return {"query": query, "evidence": self._add_evidence(hits)} if hits else {"query": query, "evidence": [],
                                                                                    "note": "No matching evidence."}

    def find_candidates_by_skill(self, skill: str) -> dict:
        canonical = list(extract_skills(skill)) or [skill]
        target = canonical[0]
        found = []
        for r in ranking_rows(self.job_id):
            if target in r["skills"] or target.lower() in (s.lower() for s in r["skills"]):
                found.append({"rank": r["rank"], "candidate": self._label(r["id"], r["name"]),
                              "final_score": round(r["final"], 1), "source": "resume"})
        if not found:
            for e in retrieve(self.job_id, skill, k=8, blind=self.blind):
                if target.lower() in e["text"].lower():
                    found.append({"candidate": e["candidate"], "source": e["source"], "snippet": e["text"][:200]})
        return {"skill": target, "matches": found}

    def interview_question_plan(self, candidate: str, focus: str | None = None) -> dict:
        cid, error = self.lookup(candidate)
        if error:
            return {"error": error}
        profile = self._profile(cid)
        if profile is None:
            return {"error": f"No candidate matching '{candidate}'."}
        gap = self.get_skill_gap(str(cid))
        job = self.get_job_requirements()
        query = focus or ("projects and achievements " + " ".join(gap.get("matched_required", [])[:5]))
        evidence = self._add_evidence(retrieve(self.job_id, query, [cid], k=4, blind=self.blind))
        return {"candidate": profile["candidate"], "job_title": job["title"],
                "missing_required": gap.get("missing_required", []), "matched_required": gap.get("matched_required", []),
                "experience_years": profile["experience_years"], "required_years": job["min_years"],
                "responsibilities_excerpt": job["responsibilities"][:600], "claims_to_verify": evidence}

    # ------------------------------------------------------------- dispatch
    def execute(self, name: str, args: dict) -> str:
        handler = getattr(self, name, None)
        if name not in {t.name for t in self.specs()} or handler is None:
            return json.dumps({"error": f"Unknown tool {name}"})
        try:
            return json.dumps(handler(**args), default=str)
        except TypeError as exc:
            return json.dumps({"error": f"Bad arguments for {name}: {exc}"})
        except Exception as exc:  # tools must never crash the agent loop
            return json.dumps({"error": f"{name} failed: {exc}"})
