"""Stage 4 - multi-feature scoring.

    FINAL = sum_i w_i * S_i / sum_i w_i      over the features that are available

S_SEMANTIC   calibrated cosine(resume vector, JD vector)
S_EXPERIENCE rule-based experience-tier fit
S_SKILL      weighted share of required (x1.0) and preferred (x0.5) skills found;
             skills only mentioned in the video earn half credit
S_VIDEO      calibrated cosine(transcript vector, responsibilities vector)
S_COMM       structural transcript metrics (pace, diversity, fillers)

Features that do not apply (no video, JD without skills) are dropped and the
remaining weights renormalised, so candidates are never penalised for an
optional input - the explanation states this explicitly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from intellihire.config import DEFAULT_WEIGHTS
from intellihire.embeddings.encoder import cosine
from intellihire.nlp.experience import experience_fit

REQUIRED_WEIGHT = 1.0
PREFERRED_WEIGHT = 0.5
VIDEO_ONLY_CREDIT = 0.5
FEATURES = ["semantic", "skill", "experience", "video", "communication"]
FEATURE_LABELS = {
    "semantic": "Semantic match",
    "skill": "Skill validation",
    "experience": "Experience fit",
    "video": "Video relevance",
    "communication": "Communication",
}


@dataclass
class JobSpec:
    required_skills: list[str]
    preferred_skills: list[str]
    min_years: float | None
    max_years: float | None
    jd_vec: np.ndarray | None
    resp_vec: np.ndarray | None
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    # False when the job vectors were produced by a different encoder than the current one
    # (cosine across embedding spaces is meaningless): semantic and video scores are dropped.
    embeddings_ok: bool = True
    embeddings_note: str = ""
    encoder_name: str = ""  # encoder that produced jd_vec/resp_vec


@dataclass
class CandidateSpec:
    skills: dict[str, list[str]]
    years: float
    resume_vec: np.ndarray | None
    video_skills: list[str] = field(default_factory=list)
    video_vec: np.ndarray | None = None
    comm_score: float | None = None
    comm_reliable: bool = True
    # False when the stored resume/transcript vectors come from a different encoder than the job's
    # (and the active one): that single feature is dropped for this candidate.
    resume_ok: bool = True
    video_ok: bool = True


@dataclass
class ScoreBreakdown:
    s_semantic: float
    raw_cosine: float
    s_skill: float | None
    s_experience: float
    s_video: float | None
    raw_video_cosine: float | None
    s_comm: float | None
    final_score: float  # 0-100
    weights_used: dict[str, float]
    contributions: dict[str, float]  # points (0-100 scale) contributed per feature
    matched_required: list[str]
    missing_required: list[str]
    matched_preferred: list[str]
    missing_preferred: list[str]
    video_only_skills: list[str]
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    def feature_value(self, name: str) -> float | None:
        return {
            "semantic": self.s_semantic,
            "skill": self.s_skill,
            "experience": self.s_experience,
            "video": self.s_video,
            "communication": self.s_comm,
        }[name]


def skill_score(job: JobSpec, cand: CandidateSpec) -> tuple[float | None, dict[str, list[str]]]:
    have = set(cand.skills)
    video_have = set(cand.video_skills) - have
    detail = {
        "matched_required": [s for s in job.required_skills if s in have],
        "missing_required": [s for s in job.required_skills if s not in have and s not in video_have],
        "matched_preferred": [s for s in job.preferred_skills if s in have],
        "missing_preferred": [s for s in job.preferred_skills if s not in have and s not in video_have],
        "video_only": [s for s in job.required_skills + job.preferred_skills if s in video_have],
    }
    total = REQUIRED_WEIGHT * len(job.required_skills) + PREFERRED_WEIGHT * len(job.preferred_skills)
    if total == 0:
        return None, detail
    earned = REQUIRED_WEIGHT * len(detail["matched_required"]) + PREFERRED_WEIGHT * len(detail["matched_preferred"])
    for s in detail["video_only"]:
        earned += VIDEO_ONLY_CREDIT * (REQUIRED_WEIGHT if s in job.required_skills else PREFERRED_WEIGHT)
    return round(earned / total, 4), detail


def compute_scores(job: JobSpec, cand: CandidateSpec, calibrate) -> ScoreBreakdown:
    """`calibrate` maps a raw cosine to [0, 1] (encoder specific)."""
    sem_ok = job.embeddings_ok and cand.resume_ok and job.jd_vec is not None and cand.resume_vec is not None
    raw_cos = cosine(cand.resume_vec, job.jd_vec) if sem_ok else 0.0
    s_sem = calibrate(raw_cos) if sem_ok else None
    s_skill, skills = skill_score(job, cand)
    s_exp, exp_reason = experience_fit(cand.years, job.min_years, job.max_years)

    s_video = raw_vid = None
    if job.embeddings_ok and cand.video_ok and cand.video_vec is not None and job.resp_vec is not None:
        raw_vid = cosine(cand.video_vec, job.resp_vec)
        s_video = calibrate(raw_vid)
    s_comm = cand.comm_score if cand.video_vec is not None else None

    values = {"semantic": s_sem, "skill": s_skill, "experience": s_exp, "video": s_video, "communication": s_comm}
    weights = {k: float(job.weights.get(k, DEFAULT_WEIGHTS[k])) for k in FEATURES}
    if s_comm is not None and not cand.comm_reliable:
        weights["communication"] *= 0.5  # very short transcripts count less
    active = {k: w for k, w in weights.items() if values[k] is not None and w > 0}
    total_w = sum(active.values())
    norm = {k: w / total_w for k, w in active.items()} if total_w > 0 else {}
    weights_used = {k: round(w, 4) for k, w in norm.items()}  # display only; the maths uses `norm`
    contributions = {k: round(100 * norm[k] * values[k], 2) for k in norm}
    final = round(sum(100 * norm[k] * values[k] for k in norm), 2)

    reasons = []
    if s_sem is not None:
        reasons.append(f"Semantic similarity to the job description: cosine {raw_cos:.3f} -> {s_sem * 100:.0f}/100")
    else:
        reasons.append(job.embeddings_note if not job.embeddings_ok else
                       ("Semantic similarity unavailable: this resume's embedding was built with a different encoder "
                        "- embeddings unavailable, re-embed (Admin > Re-embed all documents)" if not cand.resume_ok else
                        "Semantic similarity unavailable (missing embeddings); weight redistributed"))
    reasons.append(exp_reason)
    if not active:
        reasons.append("No scoring feature has a positive weight and a value; the final score is 0")
    n_req = len(job.required_skills)
    if s_skill is not None:
        reasons.append(
            f"Has {len(skills['matched_required'])}/{n_req} required and "
            f"{len(skills['matched_preferred'])}/{len(job.preferred_skills)} preferred skills"
        )
        if skills["missing_required"]:
            reasons.append("Missing required: " + ", ".join(skills["missing_required"][:8]))
        if skills["video_only"]:
            reasons.append("Mentioned only in video (half credit): " + ", ".join(skills["video_only"]))
    else:
        reasons.append("Job has no listed skills; skill score not applied")
    if s_video is None and cand.video_vec is not None:
        reasons.append("Video relevance unavailable (job or transcript embeddings are out of date - re-embed); "
                       "weight redistributed")
    elif s_video is None:
        reasons.append("No video introduction - video and communication weights redistributed"
                       if active else "No video introduction")
    else:
        reasons.append(f"Video transcript relevance to responsibilities: cosine {raw_vid:.3f} -> {s_video * 100:.0f}/100")

    return ScoreBreakdown(
        s_semantic=0.0 if s_sem is None else round(s_sem, 4),
        raw_cosine=round(raw_cos, 4),
        s_skill=s_skill,
        s_experience=round(s_exp, 4),
        s_video=None if s_video is None else round(s_video, 4),
        raw_video_cosine=None if raw_vid is None else round(raw_vid, 4),
        s_comm=s_comm,
        final_score=final,
        weights_used=weights_used,
        contributions=contributions,
        matched_required=skills["matched_required"],
        missing_required=skills["missing_required"],
        matched_preferred=skills["matched_preferred"],
        missing_preferred=skills["missing_preferred"],
        video_only_skills=skills["video_only"],
        reasons=reasons,
    )


def feature_row(b: ScoreBreakdown, has_video: bool) -> list[float]:
    """Feature vector for the ML ranker (missing features imputed as 0 + indicator)."""
    return [
        b.s_semantic,
        b.s_skill if b.s_skill is not None else 0.0,
        b.s_experience,
        b.s_video if b.s_video is not None else 0.0,
        b.s_comm if b.s_comm is not None else 0.0,
        1.0 if has_video else 0.0,
    ]


ML_FEATURE_NAMES = ["semantic", "skill", "experience", "video", "communication", "has_video"]
