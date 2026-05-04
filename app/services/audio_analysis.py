import librosa
import numpy as np

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
ANALYSIS_SAMPLE_RATE = 22050
MAX_ANALYSIS_DURATION_SEC = 120

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

def detect_scale(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    best_score = -1
    best_key = "Unknown"

    for i in range(12):
        major_score = np.corrcoef(chroma_mean, np.roll(MAJOR_PROFILE, i))[0, 1]
        minor_score = np.corrcoef(chroma_mean, np.roll(MINOR_PROFILE, i))[0, 1]

        if major_score > best_score:
            best_score = major_score
            best_key = f"{NOTE_NAMES[i]} Major"

        if minor_score > best_score:
            best_score = minor_score
            best_key = f"{NOTE_NAMES[i]} Minor"

    return best_key

def midi_to_note(midi_note):
    note_index = int(round(midi_note)) % 12
    octave = int(round(midi_note)) // 12 - 1
    return f"{NOTE_NAMES[note_index]}{octave}"

def safe_tempo(y, sr):
    """
    Estimate tempo without crashing on large files.
    """
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    except Exception:
        return 0

    tempo_arr = np.asarray(tempo)
    if tempo_arr.ndim == 0:
        return int(round(float(tempo_arr)))
    if tempo_arr.size > 0:
        return int(round(float(np.mean(tempo_arr))))
    return 0

def analyze_audio(file_path):
    y, sr = librosa.load(
        file_path,
        sr=ANALYSIS_SAMPLE_RATE,
        mono=True,
        duration=MAX_ANALYSIS_DURATION_SEC,
    )

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    pitch_values = []
    pitch_timeline = []

    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        time = librosa.frames_to_time(i, sr=sr)

        if pitch > 0:
            pitch_values.append(pitch)

            pitch_timeline.append({
                "time": round(float(time), 2),
                "pitch": round(float(pitch), 2)
            })

    pitch_timeline = pitch_timeline[::5]

    if not pitch_values:
        return {
            "pitch": "N/A",
            "note": "N/A",
            "midiNote": None,
            "scale": detect_scale(y, sr),
            "confidence": 0,
            "tempo": 0,
            "notes": [],
            "pitchTimeline": pitch_timeline, 
        }

    avg_pitch = float(np.mean(pitch_values))
    midi_note = int(round(librosa.hz_to_midi(avg_pitch)))
    note = midi_to_note(midi_note)

    tempo = safe_tempo(y, sr)
    scale = detect_scale(y, sr)

    return {
        "pitch": f"{avg_pitch:.2f} Hz",
        "note": note,
        "midiNote": midi_note,
        "scale": scale,
        "confidence": 0.9,
        "tempo": int(round(tempo)),
        "notes": [note],
        "pitchTimeline": pitch_timeline,
    }   