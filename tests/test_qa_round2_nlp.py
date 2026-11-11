"""Round-2 regression tests for the NLP layer (O6: education words inside work entries)."""
from datetime import date

import pytest

from intellihire.nlp.experience import extract_experience

TODAY = date(2026, 11, 11)


def _years(text):
    return extract_experience(text, today=TODAY).years


def test_degree_word_in_company_name_is_work():
    # "360 Degree Solutions" is an employer, not a degree; the date range must count.
    assert _years("Experience\nSoftware Engineer, 360 Degree Solutions, Jan 2019 - Dec 2021") == pytest.approx(3.0, abs=0.2)


def test_degree_word_in_company_name_without_heading_is_work():
    assert _years("Software Engineer, 360 Degree Solutions, Jan 2019 - Dec 2021") == pytest.approx(3.0, abs=0.2)


def test_date_line_after_degree_word_company_is_work():
    text = "Experience\nSoftware Engineer, 360 Degree Solutions\nJan 2019 - Dec 2021"
    assert _years(text) == pytest.approx(3.0, abs=0.2)


def test_bachelor_word_in_company_name_is_work():
    assert _years("Experience\nAnalyst at Bachelor & Co, 2019 - 2021") == 2.0


def test_university_employer_is_work():
    assert _years("University of Phoenix — IT Support Specialist, 2019 - 2021") == 2.0
    assert _years("Experience\nUniversity of Phoenix — IT Support Specialist, 2019 - 2021") == 2.0


def test_education_still_excluded_after_company_with_degree_word():
    # A real degree line directly before a date line is still education.
    text = "Experience\nAcme Corp, Developer, 2018 - 2019\nEducation\nB.Tech in CS, IIT Delhi\n2014 - 2018"
    assert _years(text) == 1.0


def test_plain_university_line_without_job_context_still_excluded():
    assert _years("XYZ University 2015 - 2019\nSkills: Python") == 0.0


def test_bachelor_of_line_with_dates_still_excluded():
    assert _years("Bachelor of Technology, XYZ Institute, 2015 - 2019") == 0.0
    assert _years("Experience\nMaster of Science, ABC College, 2020 - 2022") == 0.0


def test_mba_and_phd_lines_still_excluded():
    assert _years("MBA, Finance, 2019 - 2021") == 0.0
    assert _years("Ph.D. Computer Science, 2016 - 2020") == 0.0


def test_dates_under_education_heading_without_experience_section_excluded():
    text = "Education & Training\nState Board Certificate, 2010 - 2012\nSkills\nPython"
    assert _years(text) == 0.0


def test_company_job_after_education_heading_still_counts():
    text = "Education\nB.Tech in CS, IIT Delhi\nSkills\nPython\nAcme Corp, Engineer, 2019 - 2021"
    assert _years(text) == 2.0
