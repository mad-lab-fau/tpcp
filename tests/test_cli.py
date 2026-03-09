"""Tests for the tpcp CLI helpers."""

from pathlib import Path

import pytest

from tpcp import _cli


def _create_skill_tree(base: Path, *skill_names: str) -> Path:
    skill_root = base / "skills"
    skill_root.mkdir()
    for name in skill_names:
        skill_dir = skill_root / name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(f"# {name}\n")
    return skill_root


def test_install_skills_copies_into_agent(tmp_path, monkeypatch):
    """The installer should copy all distributed skills into `.agent`."""
    source = _create_skill_tree(tmp_path, "skill_a", "skill_b")
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.setattr(_cli, "_find_distributed_skills_dir", lambda: source)

    installed = _cli.install_skills(project_dir)

    assert [p.name for p in installed] == ["skill_a", "skill_b"]
    assert (project_dir / ".agent" / "skill_a" / "SKILL.md").exists()
    assert (project_dir / ".agent" / "skill_b" / "SKILL.md").exists()


def test_install_skills_errors_on_existing_without_force(tmp_path, monkeypatch):
    """Existing skills should block installation unless `force=True`."""
    source = _create_skill_tree(tmp_path, "skill_a")
    project_dir = tmp_path / "project"
    existing_dir = project_dir / ".agent" / "skill_a"
    existing_dir.mkdir(parents=True)
    (existing_dir / "SKILL.md").write_text("old\n")
    monkeypatch.setattr(_cli, "_find_distributed_skills_dir", lambda: source)

    with pytest.raises(FileExistsError, match="skill_a"):
        _cli.install_skills(project_dir)


def test_install_skills_force_replaces_existing(tmp_path, monkeypatch):
    """`force=True` should replace already installed skills."""
    source = _create_skill_tree(tmp_path, "skill_a")
    project_dir = tmp_path / "project"
    existing_dir = project_dir / ".agent" / "skill_a"
    existing_dir.mkdir(parents=True)
    (existing_dir / "SKILL.md").write_text("old\n")
    monkeypatch.setattr(_cli, "_find_distributed_skills_dir", lambda: source)

    _cli.install_skills(project_dir, force=True)

    assert (existing_dir / "SKILL.md").read_text() == "# skill_a\n"


def test_main_install_skills_uses_current_working_dir(tmp_path, monkeypatch, capsys):
    """The CLI should default to the current working directory."""
    source = _create_skill_tree(tmp_path, "skill_a")
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.setattr(_cli, "_find_distributed_skills_dir", lambda: source)
    monkeypatch.chdir(project_dir)

    exit_code = _cli.main(["install-skills"])

    assert exit_code == 0
    assert (project_dir / ".agent" / "skill_a" / "SKILL.md").exists()
    assert "Installed 1 skill(s)" in capsys.readouterr().out
