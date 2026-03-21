#!/usr/bin/env python3
"""
Chord Extraction Agent -- Free Version (no API needed)
======================================================
Uses only librosa for audio analysis. No Claude API required.
Automatically suggests a capo so chords become easy open shapes (Am, Em, G...).

Usage:
  python agent_free.py --list
  python agent_free.py --song gibor-al
  python agent_free.py --song gibor-al --dry-run
  python agent_free.py --all
"""

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

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SONGS_DIR    = PROJECT_ROOT / "amiado" / "songs"

# ─────────────────────────────────────────────────────────────────────────────
# CHORD TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Prefer flat names for "black key" notes (more common in guitar/song sheets)
FLAT_NAMES = {
    'C#': 'Db', 'D#': 'Eb', 'F#': 'F#', 'G#': 'Ab', 'A#': 'Bb',
}

# Easy open-chord shapes every guitarist knows
EASY_CHORDS = {'C', 'D', 'Dm', 'E', 'Em', 'G', 'A', 'Am', 'F', 'Bm', 'B'}

def _build_templates():
    t = {}
    for i, note in enumerate(NOTES):
        maj = np.zeros(12); maj[i] = 1; maj[(i+4)%12] = 1; maj[(i+7)%12] = 1
        t[note]       = maj / np.linalg.norm(maj)
        min_ = np.zeros(12); min_[i] = 1; min_[(i+3)%12] = 1; min_[(i+7)%12] = 1
        t[f"{note}m"] = min_ / np.linalg.norm(min_)
    return t

TEMPLATES = _build_templates()

def pretty(chord: str) -> str:
    """Normalise chord name: prefer flat names (Bb over A#, Eb over D#)."""
    if chord.endswith('m'):
        root, suffix = chord[:-1], 'm'
    else:
        root, suffix = chord, ''
    root = FLAT_NAMES.get(root, root)
    return root + suffix

# ─────────────────────────────────────────────────────────────────────────────
# AUDIO ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyse_audio(audio_path: str) -> dict:
    """
    Detect chords, key, tempo, and build a timeline from an audio file.
    """
    print(f"    Loading: {Path(audio_path).name}")
    y, sr    = librosa.load(audio_path, mono=True, duration=180)
    duration = librosa.get_duration(y=y, sr=sr)

    hop    = 4096
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop, bins_per_octave=36)

    # Frame-level chord assignment
    frame_times  = librosa.frames_to_time(range(chroma.shape[1]), sr=sr, hop_length=hop)
    frame_chords = []
    for frame in chroma.T:
        norm = np.linalg.norm(frame)
        if norm < 0.1:
            frame_chords.append(None)
            continue
        best = max(TEMPLATES, key=lambda c: np.dot(TEMPLATES[c], frame / norm))
        frame_chords.append(best)

    # Filter: keep only chords appearing in >3% of frames (remove noise)
    counts = Counter(c for c in frame_chords if c)
    total  = sum(counts.values())
    main   = {c for c, n in counts.items() if n / total > 0.03}

    # Build timeline: list of {time, chord} on each chord change
    timeline, prev = [], None
    for t, chord in zip(frame_times, frame_chords):
        chord = chord if chord in main else None
        if chord != prev:
            if chord:
                timeline.append({"time": round(float(t), 1), "chord": pretty(chord)})
            prev = chord

    # Unique ordered progression
    seen, progression = set(), []
    for entry in timeline:
        c = entry["chord"]
        if c not in seen:
            seen.add(c)
            progression.append(c)

    # Key estimation (Krumhansl-Schmuckler profiles)
    chroma_mean   = chroma.mean(axis=1)
    major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    best_key, best_corr = None, -1.0
    for i in range(12):
        for profile, suffix in [(major_profile, ""), (minor_profile, "m")]:
            corr = float(np.corrcoef(chroma_mean, np.roll(profile, i))[0, 1])
            if corr > best_corr:
                best_corr = corr
                best_key  = pretty(f"{NOTES[i]}{suffix}")

    # Tempo
    tempo_raw, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo_raw)[0])

    return {
        "key":        best_key,
        "tempo":      round(tempo),
        "duration":   round(duration),
        "progression": progression,
        "timeline":   timeline,
    }

# ─────────────────────────────────────────────────────────────────────────────
# CAPO OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────

def transpose_chord(chord: str, semitones: int) -> str:
    """Transpose a chord DOWN by `semitones` (capo raises pitch, so we go down)."""
    if chord.endswith('m'):
        root, suffix = chord[:-1], 'm'
    else:
        root, suffix = chord, ''
    # Resolve flat names back to sharp for indexing
    SHARP = {'Db':'C#','Eb':'D#','Gb':'F#','Ab':'G#','Bb':'A#'}
    root = SHARP.get(root, root)
    if root not in NOTES:
        return chord
    idx     = NOTES.index(root)
    new_idx = (idx - semitones) % 12
    return pretty(f"{NOTES[new_idx]}{suffix}")


def best_capo(chords: list[str]) -> tuple[int, list[str]]:
    """
    Try capo 0-7. Return (capo_fret, transposed_chords) that maximises
    the number of easy open-chord shapes.
    """
    best_fret, best_list, best_score = 0, chords, -1

    for capo in range(8):
        transposed = [transpose_chord(c, capo) for c in chords]
        score      = sum(1 for c in transposed if c in EASY_CHORDS)
        if score > best_score:
            best_score = score
            best_fret  = capo
            best_list  = transposed

    return best_fret, best_list

# ─────────────────────────────────────────────────────────────────────────────
# SECTION ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def assign_chords_to_sections(sections: list[dict], timeline: list[dict],
                               duration: float, capo: int) -> list[dict]:
    """
    Split song into equal time windows (one per section) and pick the
    most prominent chords in each window, transposed for the capo.
    """
    if not sections or not timeline:
        return []

    n       = len(sections)
    seg_len = duration / n
    result  = []

    for i, sec in enumerate(sections):
        t_start = i * seg_len
        t_end   = (i + 1) * seg_len

        seen, chords = set(), []
        for entry in timeline:
            if t_start <= entry["time"] < t_end:
                c = transpose_chord(entry["chord"], capo)
                if c not in seen:
                    seen.add(c)
                    chords.append(c)
                if len(chords) == 4:
                    break

        result.append({"section": sec["section"], "chords": chords})

    return result

# ─────────────────────────────────────────────────────────────────────────────
# SONG PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

def process_song(song_id: str, dry_run: bool = False, force: bool = False) -> bool:
    song_json_path = SONGS_DIR / song_id / "song.json"
    audio_path     = SONGS_DIR / song_id / "audio.mp3"

    print(f"\n{'='*52}")
    print(f"  Song: {song_id}")
    print(f"{'='*52}")

    if not song_json_path.exists():
        print("  X Missing song.json"); return False
    if not audio_path.exists():
        print("  X Missing audio.mp3 (download from Suno first)"); return False

    with open(song_json_path, encoding="utf-8") as f:
        song_data = json.load(f)

    print(f"  Title: {song_data.get('title', 'N/A')}")

    if song_data.get("chords") and not force:
        print("  -> Chords already present. Use --force to re-extract."); return True

    # Analyse
    print("  Analysing audio...")
    result = analyse_audio(str(audio_path))

    print(f"  Key    : {result['key']}")
    print(f"  Tempo  : {result['tempo']} BPM")
    print(f"  Chords : {' - '.join(result['progression'])}")

    # Find best capo
    capo, capo_chords = best_capo(result["progression"])
    print(f"  Capo   : {capo}  (chords become: {' - '.join(capo_chords)})")

    # Assign to sections
    sections   = song_data.get("lyrics", [])
    sec_chords = assign_chords_to_sections(sections, result["timeline"], result["duration"], capo)

    print("  Sections:")
    for s in sec_chords:
        line = ' - '.join(s['chords']) if s['chords'] else '(none detected)'
        print(f"    {s['section']}: {line}")

    chord_obj = {
        "key":      result["key"],
        "tempo":    result["tempo"],
        "capo":     capo,
        "sections": sec_chords,
    }

    if dry_run:
        print("\n  [DRY RUN - nothing saved]")
        print(json.dumps(chord_obj, ensure_ascii=False, indent=2))
        return True

    song_data["chords"] = chord_obj
    with open(song_json_path, "w", encoding="utf-8") as f:
        json.dump(song_data, f, ensure_ascii=False, indent=2)
    print(f"  OK Saved -> {song_json_path.relative_to(PROJECT_ROOT)}")
    return True

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def all_song_ids() -> list[str]:
    index = SONGS_DIR / "index.json"
    if index.exists():
        with open(index, encoding="utf-8") as f:
            return [s["id"] for s in json.load(f)]
    return sorted(d.name for d in SONGS_DIR.iterdir()
                  if d.is_dir() and (d / "song.json").exists())


def print_list():
    ids = all_song_ids()
    print(f"\nSongs ({len(ids)} total)   v=has audio   G=has chords\n")
    for sid in ids:
        audio  = "v" if (SONGS_DIR / sid / "audio.mp3").exists() else "x"
        p      = SONGS_DIR / sid / "song.json"
        chords = "G" if p.exists() and json.load(open(p, encoding="utf-8")).get("chords") else " "
        print(f"  {audio}  {chords}  {sid}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Chord Extraction - Free (librosa only, no API needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent_free.py --list
  python agent_free.py --song gibor-al --dry-run
  python agent_free.py --song gibor-al
  python agent_free.py --all
        """
    )
    parser.add_argument("--song",    metavar="ID")
    parser.add_argument("--all",     action="store_true")
    parser.add_argument("--list",    action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force",   action="store_true")
    args = parser.parse_args()

    if args.list:
        print_list(); return

    if args.song:
        ok = process_song(args.song, dry_run=args.dry_run, force=args.force)
        sys.exit(0 if ok else 1)

    if args.all:
        ids      = all_song_ids()
        ok_count = sum(process_song(s, dry_run=args.dry_run, force=args.force) for s in ids)
        print(f"\nDone: {ok_count}/{len(ids)} songs processed")
        return

    parser.print_help()


if __name__ == "__main__":
    main()