"""
Tests for textual skills registry.
"""

import tempfile
from pathlib import Path

import pytest

from metalgate_code.skills.textual_skills import TextualSkillRegistry


@pytest.fixture
def project_with_skills():
    tmp_path = Path(tempfile.mkdtemp(prefix="skills_"))

    """Create a temp project with textual skills."""
    skills_dir = tmp_path / ".metalgate" / "skills"
    (skills_dir / "deploy").mkdir(parents=True)
    (skills_dir / "migrations").mkdir(parents=True)
    (skills_dir / "code-review").mkdir(parents=True)
    (skills_dir / "empty-dir").mkdir(parents=True)

    (skills_dir / "deploy" / "SKILL.md").write_text(
        "# Deploy skill\nDeploy instructions."
    )
    (skills_dir / "migrations" / "SKILL.md").write_text(
        "# Migrations skill\nMigration guide."
    )
    (skills_dir / "code-review" / "SKILL.md").write_text(
        "# Code review skill\nReview checklist."
    )

    return tmp_path


@pytest.fixture
def empty_project() -> Path:
    """Create a temp project with no textual skills."""
    return Path(tempfile.mkdtemp(prefix="skills_"))


class TestTextualSkillRegistry:
    def test_load_skills(self, project_with_skills: Path) -> None:
        reg = TextualSkillRegistry()
        reg.load(project_with_skills)

        skills = reg.list()
        assert sorted(skills) == ["code-review", "deploy", "migrations"]

    def test_get_skill(self, project_with_skills: Path) -> None:
        reg = TextualSkillRegistry()
        reg.load(project_with_skills)

        content = reg.get("deploy")
        assert content == "# Deploy skill\nDeploy instructions."

    def test_get_missing(self, project_with_skills: Path) -> None:
        reg = TextualSkillRegistry()
        reg.load(project_with_skills)

        assert reg.get("nonexistent") is None

    def test_empty_project(self, empty_project: Path) -> None:
        reg = TextualSkillRegistry()
        reg.load(empty_project)

        assert reg.list() == []

    def test_reload(self, project_with_skills: Path) -> None:
        reg = TextualSkillRegistry()
        reg.load(project_with_skills)

        # Add a new skill after initial load
        skills_dir = project_with_skills / ".metalgate" / "skills"
        (skills_dir / "new-skill").mkdir(parents=True)
        (skills_dir / "new-skill" / "SKILL.md").write_text("# New skill")

        reg.reload()
        assert "new-skill" in reg.list()

    def test_get_all(self, project_with_skills: Path) -> None:
        reg = TextualSkillRegistry()
        reg.load(project_with_skills)

        all_skills = reg.get_all()
        assert len(all_skills) == 3
        assert "deploy" in all_skills
        assert "migrations" in all_skills
        assert "code-review" in all_skills
