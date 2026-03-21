#!/usr/bin/env python3
"""
Sync chord data from song.json files back into amiado/app.js.

Usage:
  python update_app_js.py --song maga-bamilim   # Sync one song
  python update_app_js.py --all                 # Sync all songs that have chords
  python update_app_js.py --dry-run --all       # Preview changes

This script reads the `chords` field from each amiado/songs/{id}/song.json
and writes it into the matching SONGS entry in amiado/app.js.
"""

import re
import sys
import json
import argparse
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
APP_JS       = PROJECT_ROOT / "amiado" / "app.js"
SONGS_DIR    = PROJECT_ROOT / "amiado" / "songs"

# ─────────────────────────────────────────────────────────────────────────────
# JSON → JS LITERAL CONVERTER
# ─────────────────────────────────────────────────────────────────────────────

def to_js(value, indent: int = 0) -> str:
    """
    Serialise a Python value as a compact JavaScript object literal.
    Uses single quotes for strings (matching the app.js style).
    """
    pad  = "  " * indent
    pad1 = "  " * (indent + 1)

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    if isinstance(value, list):
        if not value:
            return "[]"
        # Short lists of strings → inline
        if all(isinstance(v, str) for v in value):
            inner = ", ".join(to_js(v) for v in value)
            return f"[{inner}]"
        # Lists with objects → one object per line
        items = [f"{pad1}{to_js(v, indent + 1)}" for v in value]
        return "[\n" + ",\n".join(items) + f"\n{pad}]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = []
        for k, v in value.items():
            items.append(f"{pad1}{k}: {to_js(v, indent + 1)}")
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    raise TypeError(f"Cannot serialise type {type(value)}")


# ─────────────────────────────────────────────────────────────────────────────
# UPDATER
# ─────────────────────────────────────────────────────────────────────────────

def _find_chords_span(text: str, song_id: str) -> tuple[int, int] | None:
    """
    Locate the `chords: <value>,` span for a given song_id inside app.js text.
    Returns (start, end) byte offsets — or None if not found.
    """
    # Find the song block via its id property
    id_pattern = re.compile(
        r"id:\s*['\"]" + re.escape(song_id) + r"['\"]"
    )
    m = id_pattern.search(text)
    if not m:
        return None

    # Scan forward from there for `    chords:` (indented)
    search_start = m.start()
    # Look for the next occurrence of "chords:" within ~5000 chars
    chunk = text[search_start : search_start + 5000]

    chords_match = re.search(r"\n(\s+)chords:\s*", chunk)
    if not chords_match:
        return None

    chords_key_start = search_start + chords_match.start()   # position of \n
    value_start      = search_start + chords_match.end()     # just after "chords: "

    # Now we need to find the end of the value.
    # Values are either `[]`, `{}…}`, or a multi-line object.
    # We walk the text, tracking bracket depth.
    ch = text[value_start]
    if ch in ("{", "["):
        open_ch, close_ch = ch, "}" if ch == "{" else "]"
        depth = 0
        i     = value_start
        while i < len(text):
            if text[i] == open_ch:
                depth += 1
            elif text[i] == close_ch:
                depth -= 1
                if depth == 0:
                    value_end = i + 1
                    break
            i += 1
        else:
            return None  # unbalanced
    else:
        # Fallback: value ends at the next comma on the same line
        eol = text.find("\n", value_start)
        if eol == -1:
            return None
        line = text[value_start:eol].rstrip()
        value_end = value_start + len(line)
        if value_end > 0 and text[value_end - 1] == ",":
            value_end -= 1  # strip trailing comma — we'll re-add it

    # Span = from "chords:" keyword start to end of value
    # (the \n before `chords:` is not included — we'll keep whitespace intact)
    keyword_start = chords_key_start + 1  # skip the leading \n
    return keyword_start, value_end


def update_song_in_app_js(song_id: str, chord_data: dict, app_js_text: str) -> str | None:
    """
    Replace the chords value for `song_id` in the app.js text.
    Returns the new text, or None if the song was not found.
    """
    span = _find_chords_span(app_js_text, song_id)
    if span is None:
        return None

    start, end = span

    # Detect the indentation used for this chords line
    line_start = app_js_text.rfind("\n", 0, start) + 1
    indent_str = ""
    for ch in app_js_text[line_start:]:
        if ch in (" ", "\t"):
            indent_str += ch
        else:
            break
    indent_level = len(indent_str) // 2  # 2-space indents in this file

    # Serialise the chord object as JS
    js_value = to_js(chord_data, indent=indent_level)

    # Build replacement: `chords: <value>`
    # (trailing comma and newline are preserved from the original)
    new_fragment = f"chords: {js_value}"
    original = app_js_text[start:end]

    # Preserve trailing comma if original had one
    after_end = app_js_text[end:end + 2].lstrip()
    had_comma = app_js_text[end] == ","
    if had_comma:
        end += 1  # include the comma in the replaced range
        new_fragment += ","

    return app_js_text[:start] + new_fragment + app_js_text[end:]


# ─────────────────────────────────────────────────────────────────────────────
# HIGH-LEVEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def sync_song(song_id: str, app_js_text: str, dry_run: bool) -> tuple[bool, str]:
    """Read chords from song.json and update app_js_text. Returns (ok, new_text)."""
    song_json = SONGS_DIR / song_id / "song.json"
    if not song_json.exists():
        print(f"  ✗ {song_id}: no song.json")
        return False, app_js_text

    with open(song_json, encoding="utf-8") as f:
        data = json.load(f)

    chords = data.get("chords")
    if not chords:
        print(f"  –  {song_id}: no chords in song.json (run agent.py first)")
        return False, app_js_text

    new_text = update_song_in_app_js(song_id, chords, app_js_text)
    if new_text is None:
        print(f"  ✗ {song_id}: could not locate chords field in app.js")
        return False, app_js_text

    if dry_run:
        print(f"  ✓ {song_id}: [DRY RUN] would update chords in app.js")
    else:
        print(f"  ✓ {song_id}: chords updated in app.js")

    return True, new_text


def all_song_ids() -> list[str]:
    index = SONGS_DIR / "index.json"
    if index.exists():
        with open(index, encoding="utf-8") as f:
            return [s["id"] for s in json.load(f)]
    return sorted(d.name for d in SONGS_DIR.iterdir()
                  if d.is_dir() and (d / "song.json").exists())


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sync chord data from song.json files into app.js"
    )
    parser.add_argument("--song",    metavar="ID", help="Sync a single song")
    parser.add_argument("--all",     action="store_true", help="Sync all songs")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    if not APP_JS.exists():
        print(f"Error: {APP_JS} not found")
        sys.exit(1)

    with open(APP_JS, encoding="utf-8") as f:
        text = f.read()

    if args.song:
        ids = [args.song]
    elif args.all:
        ids = all_song_ids()
    else:
        parser.print_help()
        return

    ok_count = 0
    for sid in ids:
        ok, text = sync_song(sid, text, dry_run=args.dry_run)
        if ok:
            ok_count += 1

    if ok_count > 0 and not args.dry_run:
        with open(APP_JS, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n✓ app.js updated ({ok_count} song(s))")
    elif ok_count > 0:
        print(f"\n[DRY RUN] {ok_count} song(s) would be updated in app.js")
    else:
        print("\nNothing to update.")


if __name__ == "__main__":
    main()
