#!/usr/bin/env python3
"""Read a plaintext .txt file, write one JSON file with the original text and
every cipher's output.

    python3 ciphers.py prompt.txt ciphers.json

Top-level keys: generator, source, original, original_chars, original_sha256,
cipher_count, ciphers. Each entry in "ciphers" has name, category, description,
reversible, chars, sha256, output, and file when --files is used. Cipher
families and names: --list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Sequence

GROUPS = ("rot", "keyboard", "leet", "grid", "reverse")


# registry


@dataclass(frozen=True)
class Cipher:
    """One named transformation from plaintext to ciphertext."""

    name: str
    group: str
    description: str
    func: Callable[[str], str]
    reversible: bool = True


CIPHERS: list[Cipher] = []
_BY_NAME: dict[str, Cipher] = {}


def register(
    name: str, group: str, description: str, reversible: bool = True
) -> Callable[[Callable[[str], str]], Callable[[str], str]]:
    """Register an encoding function under `name`."""

    def decorate(func: Callable[[str], str]) -> Callable[[str], str]:
        if name in _BY_NAME:
            raise ValueError(f"duplicate cipher name: {name}")
        cipher = Cipher(name, group, description, func, reversible)
        CIPHERS.append(cipher)
        _BY_NAME[name] = cipher
        return func

    return decorate


# rot


def rot_alpha(text: str, n: int) -> str:
    """Rotate ASCII letters by n."""
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
    """Rotate ASCII digits by n."""
    out = []
    for ch in text:
        if "0" <= ch <= "9":
            out.append(chr((ord(ch) - 48 + n) % 10 + 48))
        else:
            out.append(ch)
    return "".join(out)


def rot_ascii(text: str, n: int = 47) -> str:
    """Rotate printable ASCII 33..126."""
    out = []
    for ch in text:
        code = ord(ch)
        if 33 <= code <= 126:
            out.append(chr(33 + (code - 33 + n) % 94))
        else:
            out.append(ch)
    return "".join(out)


def rot13_rot5(text: str) -> str:
    """ROT13 on letters plus ROT5 on digits."""
    return rot_digits(rot_alpha(text, 13), 5)


for _n in range(1, 26):
    register(f"rot{_n:02d}", "rot", f"letters rotated by {_n}")(
        partial(rot_alpha, n=_n)
    )

register("rot5_digits", "rot", "digits rotated by 5 (not the same as rot05)")(
    partial(rot_digits, n=5)
)
register("rot13_rot5", "rot", "rot13 on letters + rot5 on digits")(rot13_rot5)
register("rot47", "rot", "printable ASCII rotated by 47")(rot_ascii)


# keyboard

# US QWERTY rows, top to bottom.
KB_UNSHIFTED = (
    "`1234567890-=",
    "qwertyuiop[]\\",
    "asdfghjkl;'",
    "zxcvbnm,./",
)
KB_SHIFTED = (
    "~!@#$%^&*()_+",
    "QWERTYUIOP{}|",
    'ASDFGHJKL:"',
    "ZXCVBNM<>?",
)


def kb_horizontal(
    text: str,
    rows: Sequence[str],
    step: int,
    out_rows: Sequence[str] | None = None,
    fold_case: bool = True,
) -> str:
    """Walk left/right along a keyboard row, wrapping within the row.

    Keys are matched in lowercase and the case is restored when the neighbour
    is a letter, so "C" -> "X" and "c" -> "x".
    """
    out_rows = rows if out_rows is None else out_rows
    out = []
    for ch in text:
        probe = ch.lower() if fold_case else ch
        for row, out_row in zip(rows, out_rows):
            i = row.find(probe)
            if i != -1:
                neighbour = out_row[(i + step) % len(out_row)]
                if fold_case and ch.isupper() and neighbour.isalpha():
                    neighbour = neighbour.upper()
                out.append(neighbour)
                break
        else:
            out.append(ch)
    return "".join(out)


def kb_vertical(text: str, rows: Sequence[str], step: int) -> str:
    """Move to the row above/below at the same column, case preserved."""
    out = []
    for ch in text:
        probe = ch.lower()
        for r, row in enumerate(rows):
            i = row.find(probe)
            if i != -1:
                target = r + step
                if 0 <= target < len(rows):
                    neighbour = rows[target][min(i, len(rows[target]) - 1)]
                    if ch.isupper() and neighbour.isalpha():
                        neighbour = neighbour.upper()
                    out.append(neighbour)
                else:
                    out.append(ch)
                break
        else:
            out.append(ch)
    return "".join(out)


register("keyboard_left", "keyboard", "each key replaced by its left neighbour")(
    partial(kb_horizontal, rows=KB_UNSHIFTED, step=-1)
)
register("keyboard_right", "keyboard", "each key replaced by its right neighbour")(
    partial(kb_horizontal, rows=KB_UNSHIFTED, step=1)
)
register(
    "keyboard_left_shifted",
    "keyboard",
    "left neighbour while Shift is held (letters -> uppercase, digits -> symbols)",
)(partial(kb_horizontal, rows=KB_SHIFTED, step=-1, fold_case=False))
register(
    "keyboard_right_shifted",
    "keyboard",
    "right neighbour while Shift is held (letters -> uppercase, digits -> symbols)",
)(partial(kb_horizontal, rows=KB_SHIFTED, step=1, fold_case=False))
register("keyboard_up", "keyboard", "key above at the same column")(
    partial(kb_vertical, rows=KB_UNSHIFTED, step=-1)
)
register("keyboard_down", "keyboard", "key below at the same column")(
    partial(kb_vertical, rows=KB_UNSHIFTED, step=1)
)


# leet

LEET_LIGHT = {"a": "4", "e": "3", "i": "1", "o": "0"}
LEET_BASIC = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}
LEET_HEAVY = {
    "a": "4",
    "b": "8",
    "e": "3",
    "g": "9",
    "i": "1",
    "l": "1",
    "o": "0",
    "s": "5",
    "t": "7",
    "z": "2",
}


def leetspeak(text: str, table: dict[str, str]) -> str:
    """Substitute letters case-insensitively."""
    return "".join(table.get(ch.lower(), ch) for ch in text)


register("leetspeak_light", "leet", "o->0 e->3 a->4 i->1, nothing else")(
    partial(leetspeak, table=LEET_LIGHT)
)
register("leetspeak_basic", "leet", "a->4 e->3 i->1 o->0 s->5 t->7")(
    partial(leetspeak, table=LEET_BASIC)
)
register(
    "leetspeak_heavy",
    "leet",
    "basic plus b->8 g->9 l->1 z->2",
)(partial(leetspeak, table=LEET_HEAVY))


# grid

POLYBIUS5 = "abcdefghiklmnopqrstuvwxyz"  # 25 cells, i and j share one
POLYBIUS6 = "abcdefghijklmnopqrstuvwxyz0123456789"  # 36 cells
POLYBIUS_FOLD5 = {"j": "i"}  # j folds into i


def polybius(text: str, alphabet: str, fold: dict[str, str] | None = None) -> str:
    """Map letters to two-digit row/column numbers; other characters pass through."""
    width = math.isqrt(len(alphabet))
    cells = {ch: divmod(i, width) for i, ch in enumerate(alphabet)}
    fold = fold or {}
    out = []
    for ch in text:
        key = fold.get(ch.lower(), ch.lower())
        cell = cells.get(key)
        if cell is None:
            out.append(ch)
        else:
            row, col = cell
            out.append(f"{row + 1}{col + 1}")
    return "".join(out)


def square_rows(text: str, pad: str = "X") -> tuple[list[str], int]:
    """Pad the text into the smallest square that fits."""
    width = math.isqrt(len(text))
    if width * width != len(text):
        width += 1
    padded = text + pad * (width * width - len(text))
    return [padded[r * width : (r + 1) * width] for r in range(width)], width


def grid_route(text: str, mode: str) -> str:
    """Read the padded square in the order named by `mode`."""
    rows, width = square_rows(text)

    if mode == "columns":
        return "".join(rows[r][c] for c in range(width) for r in range(width))

    if mode == "columns_up":
        return "".join(
            rows[r][c] for c in range(width) for r in range(width - 1, -1, -1)
        )

    if mode == "spiral":
        out: list[str] = []
        top, bottom, left, right = 0, width - 1, 0, width - 1
        while top <= bottom and left <= right:
            out.extend(rows[top][left : right + 1])
            top += 1
            out.extend(rows[r][right] for r in range(top, bottom + 1))
            right -= 1
            if top <= bottom:
                out.extend(rows[bottom][c] for c in range(right, left - 1, -1))
                bottom -= 1
            if left <= right:
                out.extend(rows[r][left] for r in range(bottom, top - 1, -1))
                left += 1
        return "".join(out)

    raise ValueError(f"unknown grid mode: {mode}")


def rail_fence(text: str, rails: int) -> str:
    """Zigzag across `rails` rows, then read row by row."""
    if rails < 2:
        return text
    cycle = 2 * rails - 2
    fence: list[list[str]] = [[] for _ in range(rails)]
    for i, ch in enumerate(text):
        k = i % cycle
        fence[k if k < rails else cycle - k].append(ch)
    return "".join("".join(row) for row in fence)


register(
    "grid_polybius",
    "grid",
    "5x5 Polybius square, j folds into i, letters -> two digits",
    reversible=False,  # passthrough digits are indistinguishable from output pairs
)(partial(polybius, alphabet=POLYBIUS5, fold=POLYBIUS_FOLD5))
register(
    "grid_polybius6",
    "grid",
    "6x6 Polybius square, a-z plus 0-9 -> two digits",
)(partial(polybius, alphabet=POLYBIUS6))
register("grid_route_columns", "grid", "square grid, written in rows, read in columns")(
    partial(grid_route, mode="columns")
)
register(
    "grid_route_columns_up",
    "grid",
    "square grid, written in rows, columns read bottom to top",
)(partial(grid_route, mode="columns_up"))
register("grid_spiral", "grid", "square grid, read as an inward spiral")(
    partial(grid_route, mode="spiral")
)
register("grid_railfence3", "grid", "3-rail zigzag transposition")(
    partial(rail_fence, rails=3)
)
register("grid_railfence5", "grid", "5-rail zigzag transposition")(
    partial(rail_fence, rails=5)
)


# reverse


def reverse_characters(text: str) -> str:
    return text[::-1]


def reverse_word_order(text: str) -> str:
    return " ".join(reversed(text.split(" ")))


def reverse_letters_in_words(text: str) -> str:
    return " ".join(word[::-1] for word in text.split(" "))


def reverse_words_and_letters(text: str) -> str:
    return " ".join(word[::-1] for word in reversed(text.split(" ")))


register("reverse_characters", "reverse", "the whole string backwards")(
    reverse_characters
)
register("reverse_words", "reverse", "word order backwards, words left intact")(
    reverse_word_order
)
register("reverse_letters_in_words", "reverse", "letters backwards inside each word")(
    reverse_letters_in_words
)
register("reverse_words_and_letters", "reverse", "word order and letters both backwards")(
    reverse_words_and_letters
)


# engine


def select_ciphers(
    groups: Iterable[str] | None = None, names: Iterable[str] | None = None
) -> list[Cipher]:
    """Filter the registry by family and/or name, keeping registry order."""
    chosen = list(CIPHERS)
    if groups:
        wanted_groups = set(groups)
        chosen = [c for c in chosen if c.group in wanted_groups]
    if names:
        wanted_names = set(names)
        unknown = sorted(wanted_names - _BY_NAME.keys())
        if unknown:
            raise KeyError(", ".join(unknown))
        chosen = [c for c in chosen if c.name in wanted_names]
    return chosen


def encode_all(text: str, ciphers: Iterable[Cipher] | None = None) -> list[tuple[Cipher, str]]:
    """Apply each cipher to `text` in registry order."""
    return [(c, c.func(text)) for c in (CIPHERS if ciphers is None else ciphers)]


def _safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._") or "cipher"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _preview(text: str, width: int = 52) -> str:
    flat = text.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def write_outputs(
    results: Sequence[tuple[Cipher, str]],
    outdir: Path,
    ext: str = ".txt",
    ext_from_name: bool = False,
) -> dict[str, str]:
    """Write one .txt per cipher into `outdir`; return cipher name -> file name."""
    outdir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for cipher, encoded in results:
        stem = _safe_stem(cipher.name)
        suffix = f".{stem}" if ext_from_name else ext
        path = outdir / f"{stem}{suffix}"
        path.write_text(encoded, encoding="utf-8")
        written[cipher.name] = path.name
    return written


def build_payload(
    results: Sequence[tuple[Cipher, str]],
    text: str,
    source: str,
    files: dict[str, str] | None = None,
) -> dict[str, object]:
    """The JSON document: the initial text plus one entry per cipher."""
    files = files or {}
    ciphers: list[dict[str, object]] = []
    for cipher, encoded in results:
        entry: dict[str, object] = {
            "name": cipher.name,
            "category": cipher.group,
            "description": cipher.description,
            "reversible": cipher.reversible,
            "chars": len(encoded),
            "sha256": _digest(encoded),
            "output": encoded,
        }
        if cipher.name in files:
            entry["file"] = files[cipher.name]
        ciphers.append(entry)
    return {
        "generator": "testing/ciphers.py",
        "source": source,
        "original": text,
        "original_chars": len(text),
        "original_sha256": _digest(text),
        "cipher_count": len(ciphers),
        "ciphers": ciphers,
    }


def write_json(payload: dict[str, object], path: Path) -> None:
    """Write `payload` as UTF-8 JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


# cli


def read_plaintext(path: Path) -> tuple[str, str]:
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


def print_listing() -> None:
    for group in GROUPS:
        members = [c for c in CIPHERS if c.group == group]
        print(f"\n{group} ({len(members)})")
        for cipher in members:
            print(f"  {cipher.name:<26} {cipher.description}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ciphers.py",
        description=(
            "Read a plaintext .txt file, encode it with every cipher, and write a "
            "single JSON file holding the initial text plus each cipher's output."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input", type=Path, nargs="?", help="plaintext .txt file, or - for stdin"
    )
    parser.add_argument("output", type=Path, nargs="?", help="JSON file to write")
    parser.add_argument(
        "-g",
        "--group",
        action="append",
        choices=GROUPS,
        help="restrict to a family (repeatable)",
    )
    parser.add_argument("--only", help="comma-separated cipher names, see --list")
    parser.add_argument("-l", "--list", action="store_true", help="list ciphers and exit")
    parser.add_argument(
        "--files",
        type=Path,
        help="also write one .txt per cipher into this directory, named after the cipher",
    )
    parser.add_argument(
        "--ext", default=".txt", help="extension used by --files (default: %(default)s)"
    )
    parser.add_argument(
        "--ext-from-name",
        action="store_true",
        help="with --files, name the extension after the cipher too, e.g. rot13.rot13",
    )
    parser.add_argument(
        "-n", "--dry-run", action="store_true", help="print the summary, write nothing"
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="skip the summary")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print_listing()
        return 0
    if args.input is None or args.output is None:
        parser.error("both an input .txt file and an output .json file are required")

    names = [n.strip() for n in args.only.split(",") if n.strip()] if args.only else None
    try:
        selected = select_ciphers(args.group, names)
    except KeyError as exc:
        print(f"unknown cipher name(s): {exc.args[0]}\nsee --list", file=sys.stderr)
        return 2
    if not selected:
        print("no ciphers selected", file=sys.stderr)
        return 2

    text, source = read_plaintext(args.input)
    results = encode_all(text, selected)

    ext = args.ext if args.ext.startswith(".") or not args.ext else "." + args.ext
    files = (
        write_outputs(results, args.files, ext, args.ext_from_name)
        if args.files and not args.dry_run
        else {}
    )
    payload = build_payload(results, text, source, files)
    if not args.dry_run:
        write_json(payload, args.output)

    if not args.quiet:
        verb = "would write" if args.dry_run else "wrote"
        print(f"input: {len(text)} chars from {source}")
        print(f"{verb} {args.output} with {len(results)} ciphers")
        if args.files:
            where = "skipped (dry run)" if args.dry_run else f"in {args.files}/"
            print(f"  plus one .txt per cipher {where}")
        width = max(len(cipher.name) for cipher, _ in results)
        for cipher, encoded in results:
            print(
                f"  {cipher.name:<{width}}  {cipher.group:<8} "
                f"{len(encoded):>5} ch  {_preview(encoded)}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
