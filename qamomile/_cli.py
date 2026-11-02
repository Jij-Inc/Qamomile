"""Expose the packaged authoring skill through a local file management CLI."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from . import _skills


def main(argv: Sequence[str] | None = None) -> int:
    """Read or manage the skill bundled with the installed distribution.

    Args:
        argv (Sequence[str] | None): Arguments excluding the executable name.
            Defaults to process arguments.

    Returns:
        int: Zero on success; one for a skill status that is not current.

    Raises:
        SystemExit: With status two for invalid arguments or file errors, or
            zero after displaying help or the distribution version.
    """
    parser = argparse.ArgumentParser(prog="qamomile")
    parser.add_argument("--version", action="version", version=_skills.sdk_version())
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("plugin", "Inspect the bundled Qamomile authoring plugin"),
        ("marketplace", "Inspect the bundled local plugin marketplace"),
    ):
        group = commands.add_parser(name, help=help_text)
        group.add_subparsers(dest="action", required=True).add_parser(
            "path", help=f"Print the bundled {name} directory"
        )
    skill = commands.add_parser(
        "skill", help="Read or manage the Qamomile authoring skill"
    )
    actions = skill.add_subparsers(dest="action", required=True)
    actions.add_parser("path", help="Print the bundled skills directory")
    show = actions.add_parser(
        "show", help="Print SKILL.md or a bundled Markdown reference"
    )
    show.add_argument(
        "reference",
        nargs="?",
        default="SKILL.md",
        help="For example, references/authoring.md",
    )
    for action, help_text in (
        ("install", "Install into a project's agent skills directory"),
        ("status", "Check whether the installed skill matches this distribution"),
        ("update", "Update a managed skill, preserving the previous copy"),
        ("remove", "Remove a managed skill, preserving a backup"),
    ):
        command = actions.add_parser(action, help=help_text)
        command.add_argument(
            "--agent", choices=tuple(_skills.AGENT_PATHS), required=True
        )
        command.add_argument(
            "--project",
            type=Path,
            default=Path.cwd(),
            help="Existing project root (default: current directory)",
        )
    args = parser.parse_args(argv)
    try:
        if args.command == "plugin":
            print(_skills.plugin_root())
            return 0
        if args.command == "marketplace":
            print(_skills.marketplace_root())
            return 0
        if args.action == "path":
            print(_skills.bundled_root())
            return 0
        if args.action == "show":
            print(_skills.read_reference(args.reference), end="")
            return 0
        return _skills.manage(args.action, args.agent, args.project)
    except (OSError, ValueError) as error:
        parser.exit(2, f"qamomile: {error}\n")
