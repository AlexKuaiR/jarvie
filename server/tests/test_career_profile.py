from datetime import date
from pathlib import Path

import pytest

from jarvie.career import profile

EXAMPLE = Path(profile.__file__).with_name("career_profile.example.toml")


def write(tmp_path, text: str) -> Path:
    path = tmp_path / "career_profile.toml"
    path.write_text(text)
    return path


def test_example_file_loads():
    p = profile.load(EXAMPLE)
    assert p.graduation == date(2028, 5, 1)
    assert "machine learning" in p.fields
    assert p.paid_only is True
    assert p.watchlist["simplify"] == ["SimplifyJobs/Summer2027-Internships"]


def test_minimal_profile_gets_defaults(tmp_path):
    resume = tmp_path / "resume.md"
    resume.write_text("# Me\nPython, PyTorch")
    p = profile.load(
        write(tmp_path, f'resume = "{resume}"\ngraduation = "2027-12"\nfields = ["biology"]\n')
    )
    assert p.kinds == ["internship", "research"]
    assert p.min_prestige == 1
    assert p.resume_text().startswith("# Me")


def test_missing_file_explains_what_to_do(tmp_path):
    with pytest.raises(profile.ProfileError, match="copy career_profile.example.toml"):
        profile.load(tmp_path / "nope.toml")


@pytest.mark.parametrize(
    "text, message",
    [
        ('graduation = "2028-05"\nfields = ["x"]', "missing 'resume'"),
        ('resume = "r.md"\ngraduation = "May 2028"\nfields = ["x"]', "2028-05"),
        ('resume = "r.md"\ngraduation = "2028-13"\nfields = ["x"]', "01-12"),
        ('resume = "r.md"\ngraduation = "2028-05"\nfields = []', "at least one"),
        ('resume = "r.md"\ngraduation = "2028-05"\nfields = ["x"]\nkinds = ["job"]', "kinds"),
        ('resume = "r.md"\ngraduation = "2028-05"\nfields = ["x"]\nmin_prestige = 9', "1-5"),
        (
            'resume = "r.md"\ngraduation = "2028-05"\nfields = ["x"]\n[watchlist]\nworkday = []',
            "watchlist",
        ),
        ("resume = ", "valid TOML"),
    ],
)
def test_bad_values_are_named(tmp_path, text, message):
    with pytest.raises(profile.ProfileError, match=message):
        profile.load(write(tmp_path, text))


def test_resume_must_exist_and_be_text(tmp_path):
    p = profile.load(
        write(tmp_path, f'resume = "{tmp_path}/r.pdf"\ngraduation = "2028-05"\nfields = ["x"]')
    )
    with pytest.raises(profile.ProfileError, match="not found"):
        p.resume_text()
    (tmp_path / "r.pdf").write_bytes(b"%PDF")
    with pytest.raises(profile.ProfileError, match=".md or .txt"):
        p.resume_text()
