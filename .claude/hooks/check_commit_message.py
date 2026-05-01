#!/usr/bin/env python3
"""Block ``git commit`` invocations whose message contains a bare ``@``-mention.

Reads PreToolUse hook input from stdin (the standard hook payload schema —
``tool_name``, ``tool_input.command``). When the Bash command is a ``git
commit`` whose message contains an ``@<word>`` token outside of
backtick-wrapped code, prints a ``permissionDecision`` of ``deny`` with a
clear reason; otherwise prints ``{}`` and exits 0.

This enforces the project's ``### No @-mentions`` rule from ``CLAUDE.md`` —
it cannot be bypassed by a forgetful agent because the harness runs this
hook before the Bash tool actually executes the commit.

Recognized commit-message forms:

- ``git commit -m "single-line"`` / ``git commit -m 'single-line'``
- ``git commit -m "$(cat <<'EOF' ... EOF)"`` (HEREDOC)
- ``git commit -m "$(cat <<EOF ... EOF)"``

Forms not recognized fall open (allow) — better to under-block than to
falsely reject a legitimate commit. The script only ever rejects when it
has positively identified a bare ``@``-mention in extracted message text.
"""

from __future__ import annotations

import json
import re
import sys


def _extract_commit_message(command: str) -> str | None:
    """Return the proposed commit-message text from a ``git commit`` command.

    Args:
        command: The full Bash command string as captured in
            ``tool_input.command``.

    Returns:
        The extracted message body (subject + body, joined by newlines),
        or ``None`` when the command is not a ``git commit`` or the
        message could not be located in any recognized form.
    """
    if not re.search(r"\bgit\s+commit\b", command):
        return None

    # 1) HEREDOC form: ``-m "$(cat <<'EOF' ... EOF)"`` or ``<<EOF ... EOF``.
    #    The terminator must appear at the start of a line. We allow the
    #    quoted (``<<'EOF'``) and unquoted (``<<EOF``) variants.
    heredoc_match = re.search(
        r"<<-?\s*['\"]?(\w+)['\"]?\s*\n(.*?)\n\1\s*$",
        command,
        re.DOTALL | re.MULTILINE,
    )
    if heredoc_match:
        return heredoc_match.group(2)

    # 2) Plain single-quoted: ``-m '...'`` (no escapes inside in shell).
    plain_single = re.search(r"-m\s+'((?:[^'\\]|\\.)*)'", command, re.DOTALL)
    if plain_single:
        return plain_single.group(1)

    # 3) Plain double-quoted: ``-m "..."`` allowing escaped ``\"``.
    plain_double = re.search(r'-m\s+"((?:[^"\\]|\\.)*)"', command, re.DOTALL)
    if plain_double:
        return plain_double.group(1)

    return None


def _strip_backtick_code(text: str) -> str:
    """Remove fenced (``\\`\\`\\`...\\`\\`\\```) and inline (``\\`...\\```) code spans.

    Args:
        text: The candidate commit-message text.

    Returns:
        ``text`` with all backtick-wrapped code spans replaced by spaces so
        ``@`` characters inside them are not flagged. Spaces preserve token
        offsets in case future error reporting wants to point at a column.
    """
    # Fenced code blocks first (greedy across newlines, but non-greedy for
    # the closing fence to avoid swallowing the whole message).
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    # Then inline code spans (single line).
    text = re.sub(r"`[^`\n]*`", " ", text)
    return text


def _find_at_mentions(message: str) -> list[str]:
    """Return every ``@<word>``-style token outside of backtick-wrapped code.

    The ``@`` must be at a token boundary — i.e., NOT preceded by an
    alphanumeric, ``.``, ``_``, or ``-`` character. This intentionally
    skips email addresses (``noreply@anthropic.com``) and ssh-style URLs
    (``git@github.com``), which GitHub's own mention parser also ignores
    for the same boundary reason. Without this, the Co-Authored-By
    trailer the harness asks Claude to add to every commit (containing
    ``<noreply@anthropic.com>``) would be falsely rejected.

    Args:
        message: The commit-message text after extraction.

    Returns:
        A list of matched tokens in source order (duplicates preserved so
        the caller can choose to dedupe). Each token starts with ``@`` and
        is followed by one or more characters from
        ``[A-Za-z0-9_/-]`` — covers ``@username``, ``@org/team``, and bare
        decorator references like ``@qkernel``.
    """
    stripped = _strip_backtick_code(message)
    return re.findall(
        r"(?<![A-Za-z0-9._-])@[A-Za-z0-9][A-Za-z0-9_/-]*",
        stripped,
    )


def _allow() -> None:
    """Emit an empty hook response (no decision) and exit 0."""
    print(json.dumps({}))
    sys.exit(0)


def _deny(mentions: list[str]) -> None:
    """Emit a PreToolUse ``deny`` decision with an actionable reason.

    Args:
        mentions: The matched ``@<word>`` tokens (deduped before being
            interpolated into the user-facing reason string).
    """
    unique = sorted(set(mentions))
    reason = (
        "Commit message contains bare @-mention(s): "
        f"{', '.join(unique)}.\n\n"
        "Project rule (CLAUDE.md '### No @-mentions'): never write @username, "
        "@org/team, or bare Python decorators like @qkernel in commit messages, "
        "PR / issue titles or bodies, or review comments — they trigger "
        "unintended GitHub mention rendering / notifications.\n\n"
        "Fix: rewrite the offending token in descriptive prose "
        '(e.g. "the qkernel decorator" instead of @qkernel), or — only if the '
        "code form is genuinely required — wrap it in backticks (`@qkernel`). "
        "Then redo the commit."
    )
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }
    print(json.dumps(output))
    sys.exit(0)


def main() -> None:
    """Hook entry point.

    Reads JSON from stdin, decides allow / deny based on the commit message
    body, and prints the resulting hook response to stdout. Any unexpected
    exception falls open (allow) so the hook is never the cause of a
    blocked legitimate commit.
    """
    try:
        data = json.load(sys.stdin)
    except (ValueError, OSError):
        _allow()
        return

    if data.get("tool_name") != "Bash":
        _allow()
        return

    command = data.get("tool_input", {}).get("command", "")
    if not isinstance(command, str):
        _allow()
        return

    message = _extract_commit_message(command)
    if message is None:
        _allow()
        return

    mentions = _find_at_mentions(message)
    if not mentions:
        _allow()
        return

    _deny(mentions)


if __name__ == "__main__":
    main()
