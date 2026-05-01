#!/usr/bin/env python3
"""Block ``git commit`` invocations whose message contains a bare ``@``-mention.

Reads PreToolUse hook input from stdin (the standard hook payload schema —
``tool_name``, ``tool_input.command``). When the Bash command is a ``git
commit`` and the script can extract the proposed message text, it scans
the text and prints a ``permissionDecision`` of ``deny`` with a clear
reason if a bare ``@<word>`` token appears outside of backtick-wrapped
code; otherwise prints ``{}`` and exits 0.

**Best-effort enforcement** of the project's ``### No @-mentions`` rule
from ``CLAUDE.md``. The script handles the commit-message forms most
commonly produced by Claude Code:

- ``git commit -m "single-line"`` / ``git commit -m 'single-line'``
  (POSIX single quotes are literal — backslashes inside are *not*
  interpreted as escapes, matching shell semantics)
- ``git commit -m "$(cat <<'EOF' ... EOF)"`` (HEREDOC, quoted or unquoted
  delimiter, with or without surrounding quotes around the subshell)
- Multiple ``-m`` flags (each contribution treated as its own paragraph,
  matching ``git commit``'s own concatenation behavior)
- Combined short-option groups whose trailing flag is ``m`` — ``-am``,
  ``-sm``, ``-asm``, etc. (commonly ``git commit -am "msg"``)

Forms the script intentionally does **not** inspect — and which therefore
fall open (allow) — include editor-based commits (no ``-m``), ``-F`` /
``--file``, ``--amend`` reusing a previous message, custom shell
wrappers, and any commit whose message extraction fails. A fall-open
result is preferred over a false-positive that would block a legitimate
commit. Treat this hook as a strong nudge against casual mistakes, not a
complete enforcement boundary; the underlying rule still has to be
followed by humans and agents in cases the script can't see.
"""

from __future__ import annotations

import json
import re
import sys


def _extract_commit_message(command: str) -> str | None:
    """Return the proposed commit-message text from a ``git commit`` command.

    Iterates every ``-m``-bearing short-option group in the command
    (``git commit`` concatenates multiple ``-m`` values into one message,
    joined by blank lines) and, for each, tries the recognized argument
    forms in order:

    1. ``-m "$(cat <<EOF ... EOF)"`` / ``-m '$(cat <<EOF ... EOF)'`` /
       ``-m $(cat <<EOF ... EOF)`` — HEREDOC inside a subshell, with
       optional surrounding quotes; quoted (``<<'EOF'``) and unquoted
       (``<<EOF``) delimiter variants.
    2. ``-m "..."`` (plain double-quoted, shell-style escapes allowed).
    3. ``-m '...'`` (plain single-quoted, NO escapes — POSIX single
       quotes are literal, including any backslashes inside).

    Short-option groups that include ``m`` as the trailing flag are also
    recognized: ``-am``, ``-sm``, ``-asm`` etc. — common with
    ``git commit -am "msg"``. Because ``-m`` consumes the rest of the
    short-option group as its value in standard getopt parsing, ``m`` is
    only valid at the end of the group; ``[a-zA-Z]*m\\b`` captures that.

    All matched message segments are joined by a blank line so the
    downstream scanner sees the full effective message, matching what
    ``git commit`` itself would record.

    Args:
        command: The full Bash command string as captured in
            ``tool_input.command``.

    Returns:
        The concatenated message body, or ``None`` when the command is
        not a ``git commit`` or no recognized message argument could be
        located. ``None`` causes the caller to fall open (allow).
    """
    if not re.search(r"\bgit\s+commit\b", command):
        return None

    messages: list[str] = []

    # Walk every short-option group whose trailing flag is ``m`` —
    # matches a bare ``-m`` as well as ``-am`` / ``-sm`` / ``-asm`` /
    # any combination of letters followed by ``m\b``. ``getopt`` requires
    # ``m`` to be last because it takes a value, so this pattern is
    # complete for the realistic Claude Code-produced shapes. Long flags
    # like ``--message-id`` are not matched (they start with ``--``).
    for flag_match in re.finditer(r"(?:^|\s)-[a-zA-Z]*m\b", command):
        remainder = command[flag_match.end() :]

        # 1) HEREDOC inside a subshell. Surrounding quote (if any) before
        #    ``$(`` and after ``)`` must match. The closing ``\1`` must be
        #    on its own line; ``re.DOTALL`` lets ``.*?`` span newlines.
        heredoc_match = re.match(
            r'\s+(["\']?)\$\(\s*cat\s+<<-?\s*[\'"]?(\w+)[\'"]?\s*\n'
            r"(.*?)\n\2\s*\)\1",
            remainder,
            re.DOTALL,
        )
        if heredoc_match:
            messages.append(heredoc_match.group(3))
            continue

        # 2) Plain double-quoted with shell-style escape handling
        #    (``\"``, ``\\``, etc.).
        plain_double = re.match(
            r'\s+"((?:[^"\\]|\\.)*)"',
            remainder,
            re.DOTALL,
        )
        if plain_double:
            messages.append(plain_double.group(1))
            continue

        # 3) Plain single-quoted: POSIX single quotes are LITERAL — no
        #    escape interpretation, no embedded ``'``. Match any non-quote
        #    character. Backslashes inside (Windows paths, regex
        #    fragments) pass through unchanged.
        plain_single = re.match(
            r"\s+'([^']*)'",
            remainder,
            re.DOTALL,
        )
        if plain_single:
            messages.append(plain_single.group(1))
            continue

    if not messages:
        return None

    # ``git commit`` joins multiple ``-m`` values with a blank line.
    return "\n\n".join(messages)


def _strip_backtick_code(text: str) -> str:
    """Remove fenced (``\\`\\`\\`...\\`\\`\\```) and inline (``\\`...\\```) code spans.

    Args:
        text: The candidate commit-message text.

    Returns:
        ``text`` with all backtick-wrapped code spans replaced by a single
        space each, so ``@`` characters inside them are not flagged. The
        replacement is intentionally not length-preserving — the downstream
        scanner only needs to know whether a bare ``@<word>`` exists, not
        where it is in the original message.
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
    skips email addresses (for example ``noreply@example.com``) and
    ssh-style URLs (for example ``git@github.com``), so they are not
    misclassified as user mentions. GitHub's own mention parser also
    ignores these forms for the same boundary reason.

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
