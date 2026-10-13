#!/usr/bin/env python3
"""Small subset of ciphers.py for light testing: 13 ciphers, same JSON output.

    python3 ciphers_lite.py prompt.txt out.json

Categories: rot (rot13, rot07, rot47, rot13_rot5, rot07_rot5, rot13_rot47),
keyboard (keyboard_left, keyboard_right), leet (leetspeak_basic),
grid (grid_polybius, grid_route_columns),
reverse (reverse_characters, reverse_words).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# building blocks: a cipher only touches characters it can express


def rot_letters(text: str, n: int) -> str:
    """Rotate a-z / A-Z by n."""
    out = []
    for ch in text:
        if "a" <= ch <= "z":
            out.append(chr((ord(ch) - 97 + n) % 26 + 97))
        elif "A" <= ch <= "Z":
            out.append(chr((ord(ch) - 65 + n) % 26 + 65))
        else:
            out.append(ch)
    return "".join(out)


def rot_digits(text: str, n: int = 5) -> str:
    """Rotate 0-9 by n."""
    return "".join(
        chr((ord(ch) - 48 + n) % 10 + 48) if "0" <= ch <= "9" else ch for ch in text
    )


def rot_ascii(text: str, n: int = 47) -> str:
    """Rotate printable ASCII 33..126."""
    return "".join(
        chr(33 + (ord(ch) - 33 + n) % 94) if 33 <= ord(ch) <= 126 else ch
        for ch in text
    )


ROWS = ("`1234567890-=", "qwertyuiop[]\\", "asdfghjkl;'", "zxcvbnm,./")


def keyboard(text: str, step: int) -> str:
    """Move each key left (-1) or right (+1) within its row, case preserved."""
    out = []
    for ch in text:
        for row in ROWS:
            i = row.find(ch.lower())
            if i != -1:
                neighbour = row[(i + step) % len(row)]
                if ch.isupper() and neighbour.isalpha():
                    neighbour = neighbour.upper()
                out.append(neighbour)
                break
        else:
            out.append(ch)
    return "".join(out)


LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}


def leetspeak(text: str) -> str:
    """Basic leet substitutions, case-insensitive."""
    return "".join(LEET.get(ch.lower(), ch) for ch in text)


POLYBIUS5 = "abcdefghiklmnopqrstuvwxyz"  # j shares the i cell


def polybius(text: str) -> str:
    """Map letters to two-digit row/column numbers."""
    cells = {ch: divmod(i, 5) for i, ch in enumerate(POLYBIUS5)}
    out = []
    for ch in text:
        key = "i" if ch.lower() == "j" else ch.lower()
        if key in cells:
            row, col = cells[key]
            out.append(f"{row + 1}{col + 1}")
        else:
            out.append(ch)
    return "".join(out)


def route_columns(text: str) -> str:
    """Read the padded square column by column."""
    width = math.isqrt(len(text))
    if width * width != len(text):
        width += 1
    padded = text + "X" * (width * width - len(text))
    return "".join(padded[r * width + c] for c in range(width) for r in range(width))


def reverse_characters(text: str) -> str:
    return text[::-1]


def reverse_words(text: str) -> str:
    return " ".join(reversed(text.split(" ")))


# menu

CIPHERS: dict[str, tuple[str, object]] = {
    # three single rotations
    "rot13": ("rot", lambda t: rot_letters(t, 13)),
    "rot07": ("rot", lambda t: rot_letters(t, 7)),
    "rot47": ("rot", lambda t: rot_ascii(t, 47)),
    # three stacked rotations
    "rot13_rot5": ("rot", lambda t: rot_digits(rot_letters(t, 13))),
    "rot07_rot5": ("rot", lambda t: rot_digits(rot_letters(t, 7))),
    "rot13_rot47": ("rot", lambda t: rot_ascii(rot_letters(t, 13), 47)),
    # keyboard, leet, grid, reversal
    "keyboard_left": ("keyboard", lambda t: keyboard(t, -1)),
    "keyboard_right": ("keyboard", lambda t: keyboard(t, 1)),
    "leetspeak_basic": ("leet", leetspeak),
    "grid_polybius": ("grid", polybius),
    "grid_route_columns": ("grid", route_columns),
    "reverse_characters": ("reverse", reverse_characters),
    "reverse_words": ("reverse", reverse_words),
}


# cli


def read_input(path: Path) -> tuple[str, str]:
    """Read the input, dropping one trailing newline: (text, source)."""
    if str(path) == "-":
        text, source = sys.stdin.read(), "<stdin>"
    else:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SystemExit(f"cannot read {path}: {exc}") from exc
        source = str(path)

    if text.endswith("\n"):
        text = text[:-1]
    if text.endswith("\r"):
        text = text[:-1]
    if not text:
        raise SystemExit(f"input is empty: {source}")
    return text, source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ciphers_lite.py",
        description="Read a plaintext .txt file, write one JSON file with every cipher's output.",
    )
    parser.add_argument("input", type=Path, help="plaintext .txt file, or - for stdin")
    parser.add_argument("output", type=Path, help="JSON file to write")
    parser.add_argument("--only", help="comma-separated cipher names, e.g. rot13,reverse_words")
    args = parser.parse_args(argv)

    names = list(CIPHERS)
    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
        unknown = [n for n in names if n not in CIPHERS]
        if unknown:
            print(f"unknown cipher(s): {', '.join(unknown)}", file=sys.stderr)
            print(f"available: {', '.join(CIPHERS)}", file=sys.stderr)
            return 2

    text, source = read_input(args.input)
    payload = {
        "generator": "testing/ciphers_lite.py",
        "source": source,
        "original": text,
        "original_chars": len(text),
        "cipher_count": len(names),
        "ciphers": [
            {"name": name, "category": CIPHERS[name][0], "output": CIPHERS[name][1](text)}
            for name in names
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"input: {len(text)} chars from {source}")
    for entry in payload["ciphers"]:
        preview = str(entry["output"]).replace("\n", "\\n")
        label = f"{entry['name']}.txt"
        print(f"  {entry['category']:<8} {label:<24} {preview[:48]}")
    print(f"wrote {args.output} with {len(names)} ciphers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
