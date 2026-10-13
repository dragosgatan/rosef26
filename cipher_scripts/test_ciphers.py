#!/usr/bin/env python3
"""
tests for ciphers.py
run:
    python3 testing/test_ciphers.py
    python3 -m unittest discover -s testing
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ciphers 


def enc(name: str, text: str) -> str:
    return ciphers._BY_NAME[name].func(text)


class RotTests(unittest.TestCase):
    def test_rot13_known_vector(self):
        self.assertEqual(ciphers.rot_alpha("Hello, World!", 13), "Uryyb, Jbeyq!")

    def test_rot13_is_an_involution(self):
        text = "Attack at dawn, 1999!"
        self.assertEqual(ciphers.rot_alpha(ciphers.rot_alpha(text, 13), 13), text)

    def test_rot01_wraps_case_preserving(self):
        self.assertEqual(ciphers.rot_alpha("abcXYZ", 1), "bcdYZA")

    def test_rot5_digits_is_not_rot05(self):
        self.assertEqual(ciphers.rot_alpha("12345", 5), "12345")
        self.assertEqual(ciphers.rot_digits("12345", 5), "67890")

    def test_rot47_is_an_involution(self):
        text = "Hello, World! 123 ~"
        self.assertEqual(ciphers.rot_ascii(ciphers.rot_ascii(text, 47), 47), text)

    def test_rot13_rot5_letters_and_digits(self):
        self.assertEqual(ciphers.rot13_rot5("a1"), "n6")

    def test_rot18_in_the_family_is_a_plain_alphabet_rotation(self):
        self.assertEqual(ciphers.rot_alpha("abc", 18), "stu")
        self.assertEqual(ciphers._BY_NAME["rot18"].func("abc"), "stu")

    def test_rot_leaves_unicode_alone(self):
        self.assertEqual(ciphers.rot_alpha("héllo—ok", 13), "uéyyb—bx")


class KeyboardTests(unittest.TestCase):
    def test_left_then_right_round_trips(self):
        shifted = ciphers.kb_horizontal("hello world", ciphers.KB_UNSHIFTED, 1)
        self.assertEqual(
            ciphers.kb_horizontal(shifted, ciphers.KB_UNSHIFTED, -1), "hello world"
        )

    def test_left_wraps_within_the_row(self):
        self.assertEqual(ciphers.kb_horizontal("`", ciphers.KB_UNSHIFTED, -1), "=")
        self.assertEqual(ciphers.kb_horizontal("q", ciphers.KB_UNSHIFTED, 1), "w")

    def test_characters_without_a_key_survive(self):
        self.assertEqual(ciphers.kb_horizontal(" ", ciphers.KB_UNSHIFTED, 1), " ")

    def test_vertical_walk(self):
        self.assertEqual(ciphers.kb_vertical("g", ciphers.KB_UNSHIFTED, -1), "t")
        self.assertEqual(ciphers.kb_vertical("t", ciphers.KB_UNSHIFTED, 1), "g")

    def test_vertical_walk_off_the_edge_is_a_no_op(self):
        self.assertEqual(ciphers.kb_vertical("z", ciphers.KB_UNSHIFTED, 1), "z")

    def test_vertical_walk_from_the_top_row_lands_on_the_row_above(self):
        self.assertEqual(ciphers.kb_vertical("q", ciphers.KB_UNSHIFTED, -1), "`")
        self.assertEqual(ciphers.kb_vertical("a", ciphers.KB_UNSHIFTED, -1), "q")

    def test_shift_held_rows_map_keys_to_their_symbols(self):
        self.assertEqual(enc("keyboard_left_shifted", "H"), "G")
        self.assertEqual(enc("keyboard_right_shifted", "!"), "@")
        self.assertEqual(enc("keyboard_left", "1"), "`")
        self.assertEqual(enc("keyboard_right", "1"), "2")

    def test_case_is_preserved_when_the_neighbour_is_a_letter(self):
        self.assertEqual(enc("keyboard_left", "C"), "X")
        self.assertEqual(enc("keyboard_left", "c"), "x")
        self.assertEqual(enc("keyboard_down", "A"), "Z")

    def test_uppercase_input_shifts_exactly_like_lowercase(self):
        self.assertEqual(
            enc("keyboard_left", "ATTACK"), enc("keyboard_left", "attack").upper()
        )
        self.assertEqual(enc("keyboard_left", "attack"), "'rr'xj")


class LeetTests(unittest.TestCase):
    def test_basic_vector(self):
        self.assertEqual(ciphers.leetspeak("elite", ciphers.LEET_BASIC), "3l173")

    def test_light_only_touches_four_letters(self):
        self.assertEqual(ciphers.leetspeak("oOaA", ciphers.LEET_LIGHT), "0044")

    def test_heavy_vector(self):
        self.assertEqual(ciphers.leetspeak("bglz", ciphers.LEET_HEAVY), "8912")

    def test_unmapped_characters_survive(self):
        self.assertEqual(ciphers.leetspeak("xX-!", ciphers.LEET_BASIC), "xX-!")

    def test_substitution_is_case_insensitive(self):
        self.assertEqual(
            ciphers.leetspeak("Elite", ciphers.LEET_BASIC),
            ciphers.leetspeak("elite", ciphers.LEET_BASIC),
        )


class GridTests(unittest.TestCase):
    def test_polybius_known_vector(self):
        self.assertEqual(ciphers.polybius("attack", ciphers.POLYBIUS5), "114444111325")

    def test_polybius_merges_i_and_j(self):
        square = ciphers._BY_NAME["grid_polybius"].func
        self.assertEqual(square("ij"), "2424")
        self.assertEqual(square("J"), square("i"))
        self.assertEqual(
            ciphers.polybius("ij", ciphers.POLYBIUS5, ciphers.POLYBIUS_FOLD5), "2424"
        )

    def test_polybius_keeps_spacing_and_punctuation(self):
        self.assertEqual(ciphers.polybius("a t!", ciphers.POLYBIUS5), "11 44!")

    def test_polybius6_covers_digits(self):
        self.assertEqual(ciphers.polybius("0", ciphers.POLYBIUS6), "53")
        self.assertEqual(ciphers.polybius("a", ciphers.POLYBIUS6), "11")

    def test_route_columns(self):
        self.assertEqual(ciphers.grid_route("ABCDEFGHI", "columns"), "ADGBEHCFI")

    def test_route_columns_up(self):
        self.assertEqual(ciphers.grid_route("ABCDEFGHI", "columns_up"), "GDAHEBIFC")

    def test_spiral(self):
        self.assertEqual(ciphers.grid_route("ABCDEFGHI", "spiral"), "ABCFIHGDE")

    def test_rail_fence_known_vector(self):
        self.assertEqual(
            ciphers.rail_fence("WEAREDISCOVEREDFLEEATONCE", 3),
            "WECRLTEERDSOEEFEAOCAIVDEN",
        )

    def test_grid_pads_to_a_square(self):
        rows, width = ciphers.square_rows("abcde")
        self.assertEqual(width, 3)
        self.assertEqual("".join(rows), "abcdeXXXX")

    def test_routes_conserve_characters(self):
        text = "the quick brown fox jumps"
        rows, _ = ciphers.square_rows(text)
        expected = sorted("".join(rows))
        for mode in ("columns", "columns_up", "spiral"):
            with self.subTest(mode=mode):
                self.assertEqual(sorted(ciphers.grid_route(text, mode)), expected)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            ciphers.grid_route("abc", "diagonal")


class ReverseTests(unittest.TestCase):
    def test_characters(self):
        self.assertEqual(ciphers.reverse_characters("abc def"), "fed cba")

    def test_word_order(self):
        self.assertEqual(ciphers.reverse_word_order("hello brave world"), "world brave hello")

    def test_letters_within_words(self):
        self.assertEqual(ciphers.reverse_letters_in_words("ab cd ef"), "ba dc fe")

    def test_both(self):
        self.assertEqual(ciphers.reverse_words_and_letters("ab cd"), "dc ba")

    def test_word_order_keeps_double_spaces(self):
        self.assertEqual(ciphers.reverse_word_order("a  b"), "b  a")

    def test_reversing_twice_is_identity(self):
        text = "the quick brown fox"
        self.assertEqual(ciphers.reverse_characters(text)[::-1], text)
        self.assertEqual(ciphers.reverse_word_order(ciphers.reverse_word_order(text)), text)


class RegistryTests(unittest.TestCase):
    def test_names_are_unique(self):
        names = [c.name for c in ciphers.CIPHERS]
        self.assertEqual(len(names), len(set(names)))

    def test_expected_family_sizes(self):
        sizes = {}
        for cipher in ciphers.CIPHERS:
            sizes[cipher.group] = sizes.get(cipher.group, 0) + 1
        self.assertEqual(
            sizes, {"rot": 28, "keyboard": 6, "leet": 3, "grid": 7, "reverse": 4}
        )

    def test_every_family_is_covered(self):
        self.assertEqual({c.group for c in ciphers.CIPHERS}, set(ciphers.GROUPS))

    def test_select_by_group(self):
        chosen = ciphers.select_ciphers(groups=["leet"])
        self.assertTrue(chosen)
        self.assertTrue(all(c.group == "leet" for c in chosen))

    def test_select_by_name_keeps_registry_order(self):
        chosen = ciphers.select_ciphers(names=["rot13", "rot01"])
        self.assertEqual([c.name for c in chosen], ["rot01", "rot13"])

    def test_group_and_name_intersect(self):
        chosen = ciphers.select_ciphers(groups=["leet"], names=["rot13"])
        self.assertEqual(chosen, [])

    def test_unknown_name_is_reported(self):
        with self.assertRaises(KeyError):
            ciphers.select_ciphers(names=["rot13", "nope"])

    def test_encode_all_covers_every_registered_cipher(self):
        self.assertEqual(len(ciphers.encode_all("abc")), len(ciphers.CIPHERS))

    def test_safe_stem_scrubs_path_traversal(self):
        self.assertEqual(ciphers._safe_stem("../../etc/passwd"), "etc_passwd")


class CliTests(unittest.TestCase):
    """The CLI contract: a .txt goes in, one .json comes out."""

    def setUp(self):
        self.dir = Path(__file__).resolve().parent / "_tmp_cli"
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.source = self.dir / "prompt.txt"
        self.source.write_text("Attack at dawn\n", encoding="utf-8")
        self.out = self.dir / "ciphers.json"

    def run_cli(self, argv):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            code = ciphers.main(argv)
        return code, stream.getvalue()

    def load(self, path=None):
        return json.loads((path or self.out).read_text(encoding="utf-8"))

    def test_writes_one_json_holding_the_original_and_every_cipher(self):
        code, output = self.run_cli([str(self.source), str(self.out)])
        self.assertEqual(code, 0)
        payload = self.load()
        self.assertEqual(payload["original"], "Attack at dawn")
        self.assertEqual(payload["original_chars"], len("Attack at dawn"))
        self.assertEqual(payload["source"], str(self.source))
        self.assertEqual(payload["cipher_count"], len(ciphers.CIPHERS))
        self.assertEqual(
            [entry["name"] for entry in payload["ciphers"]],
            [cipher.name for cipher in ciphers.CIPHERS],
        )
        self.assertIn("wrote", output)

    def test_every_entry_carries_its_category_and_real_output(self):
        self.run_cli([str(self.source), str(self.out)])
        for entry in self.load()["ciphers"]:
            with self.subTest(name=entry["name"]):
                registered = ciphers._BY_NAME[entry["name"]]
                self.assertEqual(entry["category"], registered.group)
                self.assertIn(entry["category"], ciphers.GROUPS)
                self.assertEqual(entry["output"], registered.func("Attack at dawn"))

    def test_original_keeps_inner_newlines_and_drops_one_trailing_one(self):
        multi = self.dir / "multi.txt"
        multi.write_text("line one\nline two\n", encoding="utf-8")
        self.run_cli([str(multi), str(self.out)])
        self.assertEqual(self.load()["original"], "line one\nline two")

    def test_output_name_is_whatever_was_passed(self):
        target = self.dir / "nested" / "deep" / "custom-name.json"
        self.run_cli([str(self.source), str(target)])
        self.assertTrue(target.is_file())
        self.assertEqual(self.load(target)["original"], "Attack at dawn")

    def test_dash_reads_stdin(self):
        original = sys.stdin
        sys.stdin = io.StringIO("Attack at dawn\n")
        try:
            code, _ = self.run_cli(["-", str(self.out)])
        finally:
            sys.stdin = original
        self.assertEqual(code, 0)
        payload = self.load()
        self.assertEqual(payload["source"], "<stdin>")
        self.assertEqual(payload["original"], "Attack at dawn")

    def test_only_limits_the_entries(self):
        self.run_cli([str(self.source), str(self.out), "--only", "rot13"])
        entries = self.load()["ciphers"]
        self.assertEqual([entry["name"] for entry in entries], ["rot13"])
        self.assertEqual(entries[0]["output"], ciphers.rot_alpha("Attack at dawn", 13))
        self.assertEqual(self.load()["cipher_count"], 1)

    def test_group_limits_the_entries(self):
        self.run_cli([str(self.source), str(self.out), "--group", "leet"])
        self.assertEqual(
            [entry["category"] for entry in self.load()["ciphers"]], ["leet", "leet", "leet"]
        )

    def test_unknown_only_name_exits_2_and_writes_nothing(self):
        code, output = self.run_cli([str(self.source), str(self.out), "--only", "nope"])
        self.assertEqual(code, 2)
        self.assertIn("unknown cipher name", output)
        self.assertFalse(self.out.exists())

    def test_missing_input_file_exits_with_a_message(self):
        with self.assertRaises(SystemExit) as caught:
            self.run_cli([str(self.dir / "missing.txt"), str(self.out)])
        self.assertIn("cannot read", str(caught.exception))

    def test_empty_input_is_rejected(self):
        empty = self.dir / "empty.txt"
        empty.write_text("\n", encoding="utf-8")
        with self.assertRaises(SystemExit):
            self.run_cli([str(empty), str(self.out)])
        self.assertFalse(self.out.exists())

    def test_missing_arguments_are_reported(self):
        with self.assertRaises(SystemExit):
            self.run_cli([str(self.source)])

    def test_files_flag_also_writes_one_txt_per_cipher(self):
        files_dir = self.dir / "txt"
        self.run_cli([str(self.source), str(self.out), "--files", str(files_dir)])
        for name in ("rot13.txt", "keyboard_left.txt", "leetspeak_basic.txt",
                     "grid_polybius.txt", "reverse_words.txt"):
            with self.subTest(name=name):
                self.assertTrue((files_dir / name).is_file())
        self.assertEqual(len(list(files_dir.glob("*.txt"))), len(ciphers.CIPHERS))
        self.assertEqual(
            sorted(entry["file"] for entry in self.load()["ciphers"]),
            sorted(path.name for path in files_dir.glob("*.txt")),
        )
        self.assertEqual(
            (files_dir / "rot13.txt").read_text(encoding="utf-8"),
            ciphers.rot_alpha("Attack at dawn", 13),
        )

    def test_files_flag_honours_ext_and_ext_from_name(self):
        files_dir = self.dir / "txt"
        self.run_cli(
            [str(self.source), str(self.out), "--files", str(files_dir),
             "--ext", ".enc", "--only", "rot13"]
        )
        self.assertTrue((files_dir / "rot13.enc").is_file())
        self.run_cli(
            [str(self.source), str(self.out), "--files", str(files_dir),
             "--ext-from-name", "--only", "rot13"]
        )
        self.assertTrue((files_dir / "rot13.rot13").is_file())

    def test_entries_have_no_file_key_unless_files_were_written(self):
        self.run_cli([str(self.source), str(self.out), "--only", "rot13"])
        self.assertNotIn("file", self.load()["ciphers"][0])

    def test_dry_run_writes_nothing(self):
        code, output = self.run_cli([str(self.source), str(self.out), "--dry-run"])
        self.assertEqual(code, 0)
        self.assertFalse(self.out.exists())
        self.assertIn("would write", output)

    def test_list_works_without_input_or_output(self):
        code, output = self.run_cli(["--list"])
        self.assertEqual(code, 0)
        self.assertIn("rot13", output)
        self.assertIn("reverse_words", output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
