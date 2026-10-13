"""Configure the Spectre PDK include in all three DiffCkt template sets.

Example (from the repository root)::

    python optimization/setup.py --pdk-path /data/tools/PDKs/TSMC/europractice/extracted/65/CMOS/LP/pdk/cadence_oa/tsmcN65/../models/spectre/toplevel.scs

The path names the model file, not the PDK directory. Relative paths are
resolved from the current working directory. The file need not exist on this
machine (for example, when configuring templates for a simulation server).
"""

import argparse
from pathlib import Path
import re


VARIANTS = (
    "BiDiffCkt_DiffCkt",
    "BiDiffCkt_DiffCkt_multilabel",
    "BiDiffCkt_DiffCkt_Digress",
)
HEADERS = ("ac_header", "stb_header")
PDK_INCLUDE = re.compile(
    r'(^[ \t]*// sample pdk import[^\r\n]*\r?\n)'
    r'([ \t]*)(?://[ \t]*)?include[ \t]+"[^"\r\n]*"'
    r'([ \t]+section=)([^\s/]+)([^\r\n]*)',
    re.MULTILINE,
)


def update_templates(pdk_path, section=None, dry_run=False, root=None):
    """Update six headers, validating all matches before writing any files."""
    if not str(pdk_path).strip():
        raise ValueError("PDK path must not be empty")
    model_path = str(Path(pdk_path).expanduser().resolve())
    if any(character in model_path for character in ('"', '\r', '\n', '\x00')):
        raise ValueError("PDK path must not contain quotes, newlines, or NUL")
    if section is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", section):
        raise ValueError("Section must contain only letters, digits, _, ., or -")

    root = Path(root) if root is not None else Path(__file__).resolve().parent
    updates = []
    for variant in VARIANTS:
        for header in HEADERS:
            path = root / variant / "testbench_templates" / header
            original = path.read_bytes()
            content = original.decode("utf-8")
            matches = list(PDK_INCLUDE.finditer(content))
            if len(matches) != 1:
                raise ValueError(
                    f"{path}: expected exactly one PDK include following "
                    f"'// sample pdk import', found {len(matches)}"
                )
            match = matches[0]
            selected_section = section if section is not None else match[4]
            include = (
                f'{match[2]}include "{model_path}"'
                f'{match[3]}{selected_section}{match[5]}'
            )
            replacement = match[1] + include
            updated = (
                content[:match.start()] + replacement + content[match.end():]
            ).encode("utf-8")
            updates.append((path, original, updated, include))

    for path, original, updated, include in updates:
        if original == updated:
            status = "Unchanged"
        elif dry_run:
            status = "Would update"
        else:
            path.write_bytes(updated)
            status = "Updated"
        print(f"{status}: {path}\n  {include.strip()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdk-path", required=True,
        help="Path to the Spectre PDK model file (e.g. toplevel.scs), not a directory.",
    )
    parser.add_argument(
        "--section", default=None,
        help="Override the model section; otherwise preserve each header's section.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview all six includes without changing the templates.",
    )
    args = parser.parse_args()
    try:
        update_templates(args.pdk_path, args.section, args.dry_run)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
