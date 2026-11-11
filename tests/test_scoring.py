from __future__ import annotations

import numpy as np
import pytest

from intellihire.scoring.features import CandidateSpec, JobSpec, compute_scores, feature_row
from intellihire.scoring.ranker import NotEnoughLabels, predict_scores, status_to_label, train_ranker


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def identity_calibration(x: float) -> float:
    return max(0.0, min(1.0, x))


JOB = JobSpec(
    required_skills=["Python", "SQL", "Docker", "PyTorch"],
    preferred_skills=["Kubernetes", "AWS"],
    min_years=3, max_years=5,
    jd_vec=unit([1, 0, 0, 0]), resp_vec=unit([0, 1, 0, 0]),
)


def test_weights_renormalise_without_video():
    cand = CandidateSpec(skills={"Python": [], "SQL": []}, years=4, resume_vec=unit([1, 0, 0, 0]))
    b = compute_scores(JOB, cand, identity_calibration)
    assert b.s_video is None and b.s_comm is None
    assert set(b.weights_used) == {"semantic", "skill", "experience"}
    assert sum(b.weights_used.values()) == pytest.approx(1.0, abs=1e-3)
    # semantic 1.0, skills 2/5 (2 of 4 required, weights 1, preferred 0.5 -> 2 / 5), experience 1.0
    assert b.s_skill == pytest.approx(2 / 5)
    assert b.final_score == pytest.approx(sum(b.contributions.values()), abs=0.05)
    assert b.missing_required == ["Docker", "PyTorch"]


def test_video_only_skills_get_half_credit_and_video_features_apply():
    cand = CandidateSpec(skills={"Python": []}, years=4, resume_vec=unit([1, 0, 0, 0]),
                         video_skills=["Docker"], video_vec=unit([0, 1, 0, 0]), comm_score=0.8)
    b = compute_scores(JOB, cand, identity_calibration)
    assert b.video_only_skills == ["Docker"]
    assert "Docker" not in b.missing_required
    assert b.s_skill == pytest.approx((1 + 0.5) / 5)
    assert b.s_video == pytest.approx(1.0) and b.s_comm == 0.8
    assert set(b.weights_used) == {"semantic", "skill", "experience", "video", "communication"}


def test_perfect_candidate_scores_100_and_irrelevant_scores_low():
    perfect = CandidateSpec(skills={s: [] for s in JOB.required_skills + JOB.preferred_skills}, years=4,
                            resume_vec=unit([1, 0, 0, 0]))
    assert compute_scores(JOB, perfect, identity_calibration).final_score == pytest.approx(100, abs=0.1)
    weak = CandidateSpec(skills={}, years=0, resume_vec=unit([0, 0, 1, 0]))
    assert compute_scores(JOB, weak, identity_calibration).final_score == pytest.approx(0, abs=0.1)


def test_job_without_skills_drops_skill_feature():
    job = JobSpec([], [], None, None, unit([1, 0]), unit([0, 1]))
    b = compute_scores(job, CandidateSpec({}, 2, unit([1, 0])), identity_calibration)
    assert b.s_skill is None and "skill" not in b.weights_used


def test_feature_row_shape():
    b = compute_scores(JOB, CandidateSpec({"Python": []}, 1, unit([1, 0, 0, 0])), identity_calibration)
    assert len(feature_row(b, has_video=False)) == 6


def test_status_labels():
    assert status_to_label("shortlisted") == 1 and status_to_label("rejected") == 0 and status_to_label("new") is None


def test_ranker_learns_from_decisions(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.random((40, 6))
    y = (X[:, 1] + 0.2 * rng.random(40) > 0.6).astype(int)  # decisions driven by the skill score
    model, report = train_ranker(X, y, save_path=tmp_path / "r.joblib")
    assert (tmp_path / "r.joblib").exists()
    assert report.n_samples == 40 and report.best_model in report.cv_auc
    assert max(report.feature_importance, key=report.feature_importance.get) == "skill"
    probs = predict_scores(model, X)
    assert probs.shape == (40,) and 0 <= probs.min() and probs.max() <= 1


def test_ranker_requires_enough_labels():
    with pytest.raises(NotEnoughLabels):
        train_ranker(np.zeros((4, 6)), np.array([1, 0, 1, 1]))
