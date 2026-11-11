from __future__ import annotations

from datetime import date

from intellihire.nlp.experience import experience_fit, extract_experience, extract_requirement
from intellihire.nlp.jd_parser import parse_job_description, split_sections
from intellihire.nlp.resume_parser import parse_resume
from intellihire.nlp.skills import extract_skills


def test_aliases_map_to_canonical_skills():
    found = extract_skills("Deployed on k8s with postgres; models in sklearn and torch; github for VCS")
    assert {"Kubernetes", "PostgreSQL", "Scikit-learn", "PyTorch", "Git"} <= set(found)


def test_symbols_and_case_sensitive_skills():
    found = extract_skills("Languages: C++, C#, Go, R and .NET. Go to the market fast.")
    assert {"C++", "C#", "Go", "R", "ASP.NET"} <= set(found)
    assert "C" not in found  # 'C' inside C++/C# must not count as C
    assert "Go" not in extract_skills("we go to the gym")  # lowercase word, not the language


def test_no_false_positive_inside_words():
    found = extract_skills("Reacting quickly; javascripting; expressive writing; nodes")
    assert "React" not in found and "JavaScript" not in found


def test_skill_evidence_lines_are_kept():
    found = extract_skills("Summary\nBuilt FAISS vector search for 2M docs\nSkills: Python")
    assert found["Vector Databases"] == ["Built FAISS vector search for 2M docs"]


def test_custom_skills():
    found = extract_skills("Worked on the Zephyr scheduler", extra_skills=("Zephyr",))
    assert "Zephyr" in found


def test_experience_merges_overlapping_ranges():
    text = "Experience\nEngineer, A - Jan 2018 - Dec 2019\nConsultant, B - Jun 2019 - Dec 2020\nEducation\nB.Tech 2014 - 2018"
    info = extract_experience(text, today=date(2026, 1, 1))
    assert info.method == "date ranges"
    assert info.years == 3.0  # Jan 2018 - Dec 2020, overlap counted once, education ignored


def test_experience_present_and_explicit_statement():
    info = extract_experience("Experience\nML Engineer - Mar 2024 - Present", today=date(2026, 3, 1))
    assert info.years == 2.1
    info = extract_experience("I bring 7+ years of experience in data science.")
    assert info.years == 7 and info.method == "explicit statement"


def test_experience_without_section_skips_education_ranges():
    info = extract_experience("B.Tech, XYZ University 2015 - 2019\nAnalyst, Corp 2020 - 2022", today=date(2026, 1, 1))
    # Year-only ranges are measured mid-year to mid-year (T11): 2020 - 2022 is 2 years, not 3.
    assert info.years == 2.0


def test_jd_requirement_patterns():
    assert extract_requirement("We need 3-5 years of experience") == (3.0, 5.0)
    assert extract_requirement("Minimum 4 years in backend") == (4.0, None)
    assert extract_requirement("5+ years Python") == (5.0, None)
    assert extract_requirement("Open to freshers") == (0.0, 1.0)
    assert extract_requirement("Great culture") == (None, None)


def test_experience_fit():
    assert experience_fit(5, 3, 5)[0] == 1.0
    assert experience_fit(1.5, 3, None)[0] == 0.5
    assert experience_fit(12, 3, 5)[0] == 0.85  # over-qualified signal
    assert experience_fit(0, None, None)[0] == 1.0


def test_jd_sections_and_skill_split():
    jd = ("Data Engineer\nResponsibilities:\n- Build Spark pipelines\nRequirements:\n- Python, SQL, Airflow\n"
          "Nice to have:\n- Kafka, Snowflake\n")
    sections = split_sections(jd)
    assert "Spark" in sections["responsibilities"] and "Kafka" in sections["preferred"]
    profile = parse_job_description(jd)
    assert {"Python", "SQL", "Apache Airflow", "Apache Spark"} <= set(profile.required_skills)
    assert set(profile.preferred_skills) == {"Apache Kafka", "Snowflake"}
    assert profile.title == "Data Engineer"


def test_inline_requirements_heading():
    profile = parse_job_description("Backend role\nRequirements: Python, Django, PostgreSQL\nPreferred: Redis")
    assert {"Python", "Django", "PostgreSQL"} <= set(profile.required_skills)
    assert profile.preferred_skills == ["Redis"]


def test_resume_parser_contact_details():
    text = ("PRIYA SHARMA\npriya@example.com | +91 98765 43210 | linkedin.com/in/priya\nExperience\n"
            "ML Engineer, X - Jan 2022 - Dec 2023\nEducation\nB.Tech in CSE, SMIT, 2017 - 2021\n"
            "Certifications\nAWS Certified Machine Learning - Specialty")
    p = parse_resume(text, "priya.pdf")
    assert p.name == "Priya Sharma"
    assert p.email == "priya@example.com"
    assert p.phone.replace(" ", "") == "+919876543210"
    assert p.links == ["linkedin.com/in/priya"]
    assert p.experience.years == 2.0
    assert p.education and "B.Tech" in p.education[0]
    assert p.certifications == ["AWS Certified Machine Learning - Specialty"]


def test_resume_name_falls_back_to_filename():
    assert parse_resume("curriculum vitae\nemail: a@b.co", "rahul_verma_resume.pdf").name == "Rahul Verma"
