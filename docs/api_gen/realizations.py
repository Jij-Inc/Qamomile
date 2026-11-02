"""Render compiler-owned lowering definitions without a second choice list."""

from qamomile.lowering import list_realizations


def generate_realizations_table() -> str:
    """Return the choice table supplied by the installed native compiler."""
    lines = [
        "\n## Available implementations\n",
        "Generated from the native compiler used for this documentation build. "
        "An empty `RealizationOptions()` uses target-aware defaults; an explicit "
        "name pins a strategy, not a historical gate sequence.\n",
        "```python",
        "from qamomile.lowering import RealizationOptions, list_realizations",
        "",
        "choices = list_realizations()",
        'options = RealizationOptions({"quantum.add_into": "add.qft"})',
        "```\n",
        "| Name | Semantic operation | Required target capabilities | Strategy |",
        "| --- | --- | --- | --- |",
    ]
    for info in list_realizations():
        capabilities = ", ".join(f"`{name}`" for name in info.required_capabilities)
        description = info.description.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| `{info.name}` | `{info.family}` | {capabilities or 'None'} "
            f"| {description} |"
        )
    lines.extend(
        [
            "",
            "Names do not have independent version numbers. Fix compiler and SDK "
            "versions for repeatable compilation, and retain the emitted artifact "
            "for a concrete circuit. SDK-native implementations can change with "
            "the SDK. Old recipes are not retained indefinitely.",
            "",
        ]
    )
    return "\n".join(lines)
