"""Load the sample jobs and resumes into the database for a ready-to-explore demo.

It also records a few recruiter decisions on the ML job so the learned ranker
can be trained straight away from the Analytics page.

Usage:  python scripts/seed_demo.py                 (adds to the configured database; re-running skips jobs already loaded)
        python scripts/seed_demo.py --reset --yes   (drops all tables first; prompts for "yes" if --yes is omitted)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import select  # noqa: E402

from intellihire.db.models import Base, User  # noqa: E402
from intellihire.db.session import get_engine, init_db, session_scope  # noqa: E402
from intellihire.ingestion.documents import extract_document  # noqa: E402
from intellihire.nlp.jd_parser import parse_job_description  # noqa: E402
from intellihire.rag.vector_store import get_vector_store  # noqa: E402
from intellihire.services.pipeline import (  # noqa: E402
    add_resumes, find_candidate_ids, list_jobs, save_job, set_status,
)

SAMPLES = ROOT / "sample_data"
DECISIONS = {  # demo labels for the learned ranker
    "Priya Sharma": "shortlisted", "Meera Krishnan": "interview", "Rahul Verma": "shortlisted",
    "Fatima Khan": "shortlisted", "Karan Mehta": "rejected", "Sneha Patel": "rejected", "Vikram Singh": "rejected",
}


def load_job(user_id: int, path: Path, department: str, location: str) -> tuple[int, bool]:
    """Return (job_id, created). An existing job with the same title is reused, not duplicated."""
    text = extract_document(path.name, path.read_bytes()).text
    profile = parse_job_description(text)
    existing = {j["title"]: j["id"] for j in list_jobs()}
    title = profile.title.strip()[:200]
    if title in existing:
        return existing[title], False
    job_id = save_job(user_id, profile, title=profile.title, department=department, location=location,
                      required=profile.required_skills, preferred=profile.preferred_skills,
                      min_years=profile.min_years, max_years=profile.max_years)
    return job_id, True


def confirm_reset(assume_yes: bool) -> None:
    """Show the target database (password masked) and get explicit consent before dropping tables."""
    target = get_engine().url.render_as_string(hide_password=True)
    print(f"--reset will DROP ALL TABLES in: {target}")
    if assume_yes:
        return
    no_input = "Refusing to reset without confirmation: pass --yes to run non-interactively."
    if not sys.stdin.isatty():
        sys.exit(no_input)
    try:
        answer = input('Type "yes" to continue: ')
    except EOFError:
        sys.exit(no_input)
    if answer.strip().lower() != "yes":
        sys.exit("Aborted, nothing was changed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="drop all tables before seeding")
    parser.add_argument("--yes", action="store_true", help="confirm --reset without the interactive prompt")
    args = parser.parse_args()
    if args.yes and not args.reset:
        parser.error("--yes only applies together with --reset")
    if not (SAMPLES / "jobs").exists():
        sys.exit("sample_data/ is missing - run: python scripts/generate_samples.py")
    if args.reset:
        confirm_reset(args.yes)
        Base.metadata.drop_all(get_engine())
        get_vector_store().reset()
    init_db()
    with session_scope() as s:
        admin_id = s.scalar(select(User.id).where(User.role == "admin").order_by(User.id))

    jobs = [
        (SAMPLES / "jobs" / "ml_engineer_jd.txt", "AI Engineering", "Bengaluru", SAMPLES / "resumes" / "ml_engineer"),
        (SAMPLES / "jobs" / "fullstack_developer_jd.docx", "Product Engineering", "Remote", SAMPLES / "resumes" / "fullstack_developer"),
    ]
    for jd_path, dept, loc, resume_dir in jobs:
        job_id, created = load_job(admin_id, jd_path, dept, loc)
        if created:
            files = [(p.name, p.read_bytes()) for p in sorted(resume_dir.iterdir()) if p.is_file()]
            results = add_resumes(admin_id, job_id, files, lambda i, n, name: print(f"  [{i}/{n}] {name}"))
            print(f"Job {job_id} ({jd_path.name}): " + ", ".join(f"{r.name or r.filename} -> {r.status}" for r in results))
        else:
            print(f"Job {job_id} ({jd_path.name}) already loaded - skipping resumes")
        if "ml_engineer" in jd_path.name:
            for name, status in DECISIONS.items():
                set_status(admin_id, find_candidate_ids(job_id, name)[:1], status)
    print("Demo data loaded. Start the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
