#!/usr/bin/env python3
"""Remove bar-style comment separators from Python files.

Rules (user's style):
- Comments must be plain: "# line 1", "# line 2"
- Remove pure separator lines (lines that are only dashes/equals/spaces/#/whitespace)
- Convert inline-bar comments like "# ---- text ----" to "# text"
- Leave all other comments untouched
"""
import re
import sys

# A pure bar line: optional whitespace, #, then only dashes/equals/spaces, optional trailing #
# e.g.  "# ------------------------------------------------------------------ #"
#       "# ----------------------------------------------------------------------- #"
#       "# ---"
PURE_BAR = re.compile(r'^\s*#\s*[-=]+\s*#?\s*$')

# Inline bar: "# ---- some text ----"  or "# ---- some text"  or "# some text ----"
# Capture the text between/after/before the dashes.
INLINE_BAR_LEADING = re.compile(r'^(\s*)#\s*[-=]+\s+(.*\S)\s*[-=]*\s*$')
INLINE_BAR_TRAILING = re.compile(r'^(\s*)#\s*(.*\S)\s+[-=]+\s*$')


def fix_line(line):
    # Pure bar line -> drop entirely
    if PURE_BAR.match(line):
        return None
    # Inline bar with leading dashes: "# ---- text ----" or "# ---- text"
    m = INLINE_BAR_LEADING.match(line)
    if m:
        indent, text = m.group(1), m.group(2)
        return f"{indent}# {text}"
    # Inline bar with trailing dashes: "# text ----"
    m = INLINE_BAR_TRAILING.match(line)
    if m:
        indent, text = m.group(1), m.group(2)
        return f"{indent}# {text}"
    return line


def process_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    out = []
    changed = False
    for ln in lines:
        new = fix_line(ln)
        if new is None:
            changed = True
            continue
        if new != ln:
            changed = True
        out.append(new)

    # Collapse 3+ consecutive blank lines that may result from removing bar lines
    # into at most 2 (PEP 8). Only do this if we actually changed something.
    if changed:
        result = []
        blank_run = 0
        for ln in out:
            if ln.strip() == "":
                blank_run += 1
                if blank_run <= 2:
                    result.append(ln)
            else:
                blank_run = 0
                result.append(ln)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(result)
        print(f"fixed: {path}")
    else:
        print(f"skip:  {path}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        process_file(p)
