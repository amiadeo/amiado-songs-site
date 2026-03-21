#!/usr/bin/env python3
"""
Chord Extraction Agent for Amiado Songs Site
=============================================
Uses librosa for audio analysis + Claude API as an expert music theory agent.

Usage:
  python agent.py --list                        # List songs & audio status
  python agent.py --song maga-bamilim           # Process one song
  python agent.py --all                         # Process all songs
  python agent.py --song maga-bamilim --dry-run # Preview without saving
  python agent.py --song maga-bamilim --force   # Re-extract even if chords exist

Requirements:
  pip install -r requirements.txt
  export ANTHROPIC_API_KEY=sk-ant-...
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

try:
    import librosa
except ImportError:
    print("Error: librosa not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

try:
    import anthropic
except ImportError:
    print("Error: anthropic not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SONGS_DIR    = PROJECT_ROOT / "amiado" / "songs"

# ─────────────────────────────────────────────────────────────────────────────
# CHORD DETECTION — chroma-based template matching
# ─────────────────────────────────────────────────────────────────────────────

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def _build_chord_templates() -> dict[str, np.ndarray]:
    templates = {}
    for i, note in enumerate(NOTES):
        # Major triad: root + major-third (4) + perfect-fifth (7)
        major = np.zeros(12); major[i] = 1; major[(i+4)%12] = 1; major[(i+7)%12] = 1
        templates[note]        = major / np.linalg.norm(major)
        # Minor triad: root + minor-third (3) + perfect-fifth (7)
        minor = np.zeros(12); minor[i] = 1; minor[(i+3)%12] = 1; minor[(i+7)%12] = 1
        templates[f"{note}m"]  = minor / np.linalg.norm(minor)
        # Dominant-7: root + major-third (4) + fifth (7) + minor-7th (10)
        dom7  = np.zeros(12); dom7[i] = 1; dom7[(i+4)%12] = 1; dom7[(i+7)%12] = 1; dom7[(i+10)%12] = 0.8
        templates[f"{note}7"]  = dom7  / np.linalg.norm(dom7)
    return templates

CHORD_TEMPLATES = _build_chord_templates()


def detect_chords(audio_path: str, hop_length: int = 4096) -> list[str]:
    """
    Extract a chord progression from an audio file.
    Returns an ordered list of unique chords (e.g. ['Am', 'G', 'F', 'E']).
    """
    print(f"    Loading audio: {Path(audio_path).name}")
    y, sr = librosa.load(audio_path, mono=True, duration=180)  # max 3 min

    # CQT chroma is more pitch-accurate than STFT for music
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, bins_per_octave=36)

    # Frame-level chord assignment
    frame_chords = []
    for frame in chroma.T:
        norm = np.linalg.norm(frame)
        if norm < 0.1:
            frame_chords.append(None)
            continue
        frame_norm = frame / norm
        best = max(CHORD_TEMPLATES, key=lambda c: np.dot(CHORD_TEMPLATES[c], frame_norm))
        frame_chords.append(best)

    # Keep only chords that appear in >3 % of total frames (filter noise)
    counts = Counter(c for c in frame_chords if c)
    total  = sum(counts.values())
    main   = {c for c, n in counts.items() if n / total > 0.03}

    # Ordered unique progression (remove consecutive duplicates)
    progression = []
    for chord in frame_chords:
        if chord and chord in main and (not progression or chord != progression[-1]):
            progression.append(chord)

    # Cap at 12 distinct chords
    seen, result = set(), []
    for chord in progression:
        if chord not in seen:
            seen.add(chord)
            result.append(chord)
        if len(result) == 12:
            break

    return result


def detect_key_and_tempo(audio_path: str) -> dict:
    """Estimate musical key and tempo (BPM) from audio."""
    y, sr = librosa.load(audio_path, mono=True, duration=180)

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Key via Krumhansl-Schmuckler profiles
    chroma      = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    best_key, best_corr = None, -1.0
    for i in range(12):
        for profile, suffix in [(major_profile, ""), (minor_profile, "m")]:
            corr = float(np.corrcoef(chroma_mean, np.roll(profile, i))[0, 1])
            if corr > best_corr:
                best_corr = corr
                best_key  = f"{NOTES[i]}{suffix}"

    return {"key": best_key, "tempo": round(float(tempo))}


# ─────────────────────────────────────────────────────────────────────────────
# CLAUDE AGENT — validates and structures chord data
# ─────────────────────────────────────────────────────────────────────────────

AGENT_TOOLS = [
    {
        "name": "set_chord_progression",
        "description": (
            "Set the final, validated chord progression for the song. "
            "Call this once you have cleaned up the raw detected chords and "
            "mapped them to each section of the song."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Musical key (e.g. 'Am', 'G', 'Cm')"
                },
                "tempo": {
                    "type": "integer",
                    "description": "Estimated BPM"
                },
                "capo": {
                    "type": "integer",
                    "description": "Suggested capo fret — 0 means no capo"
                },
                "sections": {
                    "type": "array",
                    "description": "Chord progression per section",
                    "items": {
                        "type": "object",
                        "properties": {
                            "section": {
                                "type": "string",
                                "description": "Section name as it appears in the lyrics (Hebrew)"
                            },
                            "chords": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Ordered chord list for this section"
                            },
                            "notes": {
                                "type": "string",
                                "description": "Optional guitarist note (e.g. 'repeat 4x', 'use barre chord')"
                            }
                        },
                        "required": ["section", "chords"]
                    }
                }
            },
            "required": ["key", "tempo", "capo", "sections"]
        }
    }
]


def run_claude_agent(song_data: dict, detected_chords: list[str], key_tempo: dict) -> dict | None:
    """
    Sends detected chord data to Claude, which acts as a music-theory expert.
    Returns structured chord data ready to be saved into song.json.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("    Error: ANTHROPIC_API_KEY not set")
        return None

    client   = anthropic.Anthropic(api_key=api_key)
    sections = [s["section"] for s in song_data.get("lyrics", [])]
    title    = song_data.get("title", "unknown")
    first_verse_lines = (song_data.get("lyrics") or [{}])[0].get("lines", [])

    prompt = f"""You are an expert music theorist and guitarist.

I analyzed a Hebrew song titled "{title}" using audio signal processing.

**Raw detected chord sequence** (from chroma analysis — may contain noise):
{json.dumps(detected_chords, ensure_ascii=False)}

**Estimated key:** {key_tempo.get('key')}
**Estimated tempo:** {key_tempo.get('tempo')} BPM

**Song sections (Hebrew names):**
{json.dumps(sections, ensure_ascii=False)}

**First section lyrics (for context):**
{json.dumps(first_verse_lines, ensure_ascii=False)}

Your task:
1. Clean up the raw detection — remove noise chords, identify the real 3-6 chord loop.
2. Assign chords to each section (verse and chorus often share the same loop).
3. Suggest a capo if it makes the shapes friendlier on guitar.
4. Call `set_chord_progression` with the final result.

Focus on what is actually playable and useful for a guitarist."""

    messages = [{"role": "user", "content": prompt}]
    result   = None
    print("    Calling Claude for chord validation and structuring...")

    for _ in range(5):  # safety limit on agent loop turns
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2048,
            tools=AGENT_TOOLS,
            messages=messages,
        )

        # Collect tool calls
        tool_used = False
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "set_chord_progression":
                result    = block.input
                tool_used = True
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Saved successfully."
                    }]
                })
                break

        if result is not None or response.stop_reason == "end_turn":
            break

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SONG PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

def process_song(song_id: str, dry_run: bool = False, force: bool = False) -> bool:
    song_dir       = SONGS_DIR / song_id
    song_json_path = song_dir / "song.json"
    audio_path     = song_dir / "audio.mp3"

    print(f"\n{'─'*52}")
    print(f"  Song: {song_id}")
    print(f"{'─'*52}")

    if not song_json_path.exists():
        print(f"  ✗ Missing song.json: {song_json_path}")
        return False
    if not audio_path.exists():
        print(f"  ✗ Missing audio.mp3: {audio_path}")
        print(f"    → Download from Suno and save to that path")
        return False

    with open(song_json_path, encoding="utf-8") as f:
        song_data = json.load(f)

    print(f"  Title: {song_data.get('title', 'N/A')}")

    if song_data.get("chords") and not force:
        print("  ℹ  Chords already present — skipping (use --force to re-extract)")
        return True

    # ── Audio analysis ──────────────────────────────────────────────────────
    print("  Analysing audio...")
    detected   = detect_chords(str(audio_path))
    key_tempo  = detect_key_and_tempo(str(audio_path))
    print(f"  Raw chords : {' → '.join(detected)}")
    print(f"  Key/Tempo  : {key_tempo['key']} @ {key_tempo['tempo']} BPM")

    # ── Claude agent ─────────────────────────────────────────────────────────
    chord_result = run_claude_agent(song_data, detected, key_tempo)
    if not chord_result:
        print("  ✗ Claude agent returned no result")
        return False

    # ── Print result ─────────────────────────────────────────────────────────
    print(f"  Key: {chord_result.get('key')}  |  Capo: {chord_result.get('capo', 0)}")
    for sec in chord_result.get("sections", []):
        chords_str = " - ".join(sec["chords"])
        note       = f"  ({sec['notes']})" if sec.get("notes") else ""
        print(f"    {sec['section']}: {chords_str}{note}")

    if dry_run:
        print("\n  [DRY RUN — nothing saved]")
        print(json.dumps(chord_result, ensure_ascii=False, indent=2))
        return True

    # ── Save to song.json ─────────────────────────────────────────────────────
    song_data["chords"] = chord_result
    with open(song_json_path, "w", encoding="utf-8") as f:
        json.dump(song_data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Saved → {song_json_path.relative_to(PROJECT_ROOT)}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def all_song_ids() -> list[str]:
    index_path = SONGS_DIR / "index.json"
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            return [s["id"] for s in json.load(f)]
    return sorted(d.name for d in SONGS_DIR.iterdir()
                  if d.is_dir() and (d / "song.json").exists())


def print_song_list():
    ids = all_song_ids()
    print(f"\nSongs ({len(ids)} total)  ✓=has audio  🎸=has chords\n")
    for sid in ids:
        has_audio  = "✓" if (SONGS_DIR / sid / "audio.mp3").exists() else "✗"
        has_chords = "🎸" if _song_has_chords(sid) else "  "
        print(f"  {has_audio}  {has_chords}  {sid}")


def _song_has_chords(song_id: str) -> bool:
    p = SONGS_DIR / song_id / "song.json"
    if not p.exists():
        return False
    with open(p, encoding="utf-8") as f:
        return bool(json.load(f).get("chords"))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Chord Extraction Agent — Amiado Songs Site",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py --list
  python agent.py --song maga-bamilim
  python agent.py --song maga-bamilim --dry-run
  python agent.py --song maga-bamilim --force
  python agent.py --all
        """
    )
    parser.add_argument("--song",    metavar="ID",  help="Process a single song by ID")
    parser.add_argument("--all",     action="store_true", help="Process all songs")
    parser.add_argument("--list",    action="store_true", help="List songs and their status")
    parser.add_argument("--dry-run", action="store_true", help="Show result without saving")
    parser.add_argument("--force",   action="store_true", help="Re-extract even if chords exist")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    if args.list:
        print_song_list()
        return

    if args.song:
        ok = process_song(args.song, dry_run=args.dry_run, force=args.force)
        sys.exit(0 if ok else 1)

    if args.all:
        ids           = all_song_ids()
        success, fail = 0, 0
        for sid in ids:
            ok = process_song(sid, dry_run=args.dry_run, force=args.force)
            if ok: success += 1
            else:  fail    += 1
        print(f"\n{'─'*52}")
        print(f"Done: {success} processed, {fail} skipped/failed")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
