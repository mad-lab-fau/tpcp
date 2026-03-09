"""CLI helpers for tpcp."""

from __future__ import annotations

import argparse
import shutil
import sysconfig
from pathlib import Path


def _find_distributed_skills_dir() -> Path:
    repo_skills = Path(__file__).resolve().parents[2] / "skills" / "tpcp"
    if repo_skills.is_dir():
        return repo_skills

    installed_skills = Path(sysconfig.get_paths()["data"]) / "tpcp"
    if installed_skills.is_dir():
        return installed_skills

    raise FileNotFoundError(
        "Could not locate the distributed tpcp skills. "
        "Expected either a repository checkout at `skills/tpcp` or installed package data at "
        f"`{installed_skills}`."
    )


def install_skills(project_dir: Path, *, force: bool = False) -> list[Path]:
    """Install the distributed tpcp skills into the project's `.agent` folder."""
    source_dir = _find_distributed_skills_dir()
    destination_dir = project_dir / ".agent"
    destination_dir.mkdir(parents=True, exist_ok=True)

    skill_dirs = sorted(p for p in source_dir.iterdir() if p.is_dir())
    if not force:
        conflicts = [
            destination_dir / skill_dir.name for skill_dir in skill_dirs if (destination_dir / skill_dir.name).exists()
        ]
        if conflicts:
            conflict_names = ", ".join(sorted(p.name for p in conflicts))
            raise FileExistsError(
                "The following skills already exist in the destination: "
                f"{conflict_names}. Re-run with `--force` to replace them."
            )

    installed = []
    for skill_dir in skill_dirs:
        target = destination_dir / skill_dir.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.copytree(skill_dir, target)
        installed.append(target)

    return installed


def main(argv: list[str] | None = None) -> int:
    """Run the tpcp CLI."""
    parser = argparse.ArgumentParser(prog="tpcp")
    subparsers = parser.add_subparsers(dest="command")

    install_parser = subparsers.add_parser(
        "install-skills",
        help="Copy the distributed tpcp skills into the current project's `.agent` folder.",
    )
    install_parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path.cwd(),
        help="Project directory that should receive the `.agent` folder. Defaults to the current working directory.",
    )
    install_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace already installed tpcp skills in the destination.",
    )

    args = parser.parse_args(argv)

    if args.command == "install-skills":
        installed = install_skills(args.project_dir.resolve(), force=args.force)
        if installed:
            print(f"Installed {len(installed)} skill(s) into {args.project_dir.resolve() / '.agent'}.")
        else:
            print("No new skills installed.")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
