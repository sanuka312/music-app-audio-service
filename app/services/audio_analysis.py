
import librosa
import numpy as np

# The 12 chromatic note names (in semitone order starting from C) used to convert
# pitch class indices (0-11) into human-readable note labels.
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Target sample rate for analysis. 22050 Hz is a common librosa default that keeps
# enough frequency range for music while halving the data size vs. 44100 Hz.
ANALYSIS_SAMPLE_RATE = 22050

# Cap the analysed audio at 2 minutes to keep processing time and memory bounded
# on longer tracks.
MAX_ANALYSIS_DURATION_SEC = 120

# Krumhansl-Schmuckler key profiles: empirical weights describing how strongly
# each pitch class is perceived in a major / minor key. They are correlated with
# a song's chroma vector to estimate its key.
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def detect_scale(y, sr):
    # Compute a chromagram using the Constant-Q Transform. Each column is the energy in each of the 12 pitch classes for one time frame.
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    # Average the chroma over time to get a single 12-element vector summarising
    # the overall pitch-class distribution of the track.
    chroma_mean = np.mean(chroma, axis=1)
    # Convert the reference key profiles into numpy arrays so we can use them
    # with numpy operations like correlation and rotation.
    major_profile = np.array(MAJOR_PROFILE)
    minor_profile = np.array(MINOR_PROFILE)

    # Track the highest correlation score found so far and the matching key.
    # -1 is a safe starting value because correlations are in [-1, 1].
    best_score = -1
    best_key = "Unknown"

    # Try every possible tonic (C, C#, D, ... B) by rotating the key profiles.
    for i in range(12):
        # Compare the song's average chroma against the major profile shifted
        # to tonic `i`. np.corrcoef returns a 2x2 matrix; [0,1] is the
        # Pearson correlation between the two inputs.
        major_score = np.corrcoef(chroma_mean, np.roll(major_profile, i))[0, 1]
        # Same comparison but using the minor key profile.
        minor_score = np.corrcoef(chroma_mean, np.roll(minor_profile, i))[0, 1]

        # If this rotation gives a better major-key match than anything seen so
        # far, remember it as the current best guess.
        if major_score > best_score:
            best_score = major_score
            best_key = f"{NOTE_NAMES[i]} Major"

        # Same check for minor keys; this can overwrite the major guess if the
        # minor correlation is stronger.
        if minor_score > best_score:
            best_score = minor_score
            best_key = f"{NOTE_NAMES[i]} Minor"

    # Return the best matching key as a string like "G Major" or "A Minor".
    return best_key


def midi_to_note(midi_note):
    # MIDI numbers cycle through 12 pitch classes; modulo 12 gives the position
    # within NOTE_NAMES (0 = C, 1 = C#, ...).
    note_index = int(round(midi_note)) % 12
    # In MIDI, note 12 is C0, so dividing by 12 and subtracting 1 yields the
    # octave number used in scientific pitch notation.
    octave = int(round(midi_note)) // 12 - 1
    # Combine note name and octave, e.g. "A4" or "C#5".
    return f"{NOTE_NAMES[note_index]}{octave}"


def safe_tempo(y, sr):
    # librosa's beat tracker can throw on very short / silent audio, so wrap it
    # in a try/except and fall back to 0 BPM rather than propagating the error.
    try:
        # beat_track returns (tempo, beat_frames); we only need the tempo here.
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    except Exception:
        return 0

    # Newer librosa versions can return either a scalar or an array of tempi,
    # so normalise to a numpy array for uniform handling.
    tempo_arr = np.asarray(tempo)
    # Scalar (0-dim) case: just round it to the nearest integer BPM.
    if tempo_arr.ndim == 0:
        return int(round(float(tempo_arr)))
    # Array case with at least one value: average the candidates and round.
    if tempo_arr.size > 0:
        return int(round(float(np.mean(tempo_arr))))
    # Empty array fallback (no tempo detected).
    return 0


def analyze_audio(file_path):
    # Load the audio file: force the chosen sample rate, downmix to mono, and
    # truncate to MAX_ANALYSIS_DURATION_SEC to keep analysis fast and bounded.
    y, sr = librosa.load(
        file_path,
        sr=ANALYSIS_SAMPLE_RATE,
        mono=True,
        duration=MAX_ANALYSIS_DURATION_SEC,
    )

    # piptrack returns two matrices of shape (freq_bins, time_frames):
    # `pitches` holds candidate pitch frequencies and `magnitudes` their
    # strengths, so we can pick the dominant pitch per time frame.
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    # Collected dominant pitches across all frames (used for the average pitch).
    pitch_values = []
    # Time-aligned (time, pitch) samples used to draw a pitch contour on the UI.
    pitch_timeline = []

    # Iterate over every time frame in the piptrack output.
    for i in range(pitches.shape[1]):
        # Find the frequency bin with the largest magnitude for this frame —
        # that's the most prominent pitch at that moment.
        index = magnitudes[:, i].argmax()
        # Look up the actual frequency (in Hz) corresponding to that bin.
        pitch = pitches[index, i]
        # Convert the frame index to a timestamp in seconds.
        time = librosa.frames_to_time(i, sr=sr)

        # Drop frames where no pitch was detected (piptrack reports 0 Hz then).
        if pitch > 0:
            # Save the pitch for later averaging across the whole clip.
            pitch_values.append(pitch)

            # Record a UI-friendly point with rounded values for the timeline.
            pitch_timeline.append({
                "time": round(float(time), 2),
                "pitch": round(float(pitch), 2)
            })

    # Downsample the timeline by taking every 5th point to keep the response
    # payload small while still showing the overall pitch contour.
    pitch_timeline = pitch_timeline[::5]

    # If no pitched content was found, return a placeholder response so the
    # caller can still display scale/tempo info without crashing on missing keys.
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

    # Average pitch across all detected frames, used as a single representative
    # frequency for the clip.
    avg_pitch = float(np.mean(pitch_values))
    # Convert that frequency into the closest MIDI note number (integer).
    midi_note = int(round(librosa.hz_to_midi(avg_pitch)))
    # Turn the MIDI number into a readable note label like "A4".
    note = midi_to_note(midi_note)

    # Estimate tempo (BPM) and musical key/scale from the loaded audio.
    tempo = safe_tempo(y, sr)
    scale = detect_scale(y, sr)

    # Final response packaged as a dict consumed by the API layer.
    return {
        "pitch": f"{avg_pitch:.2f} Hz",
        "note": note,
        "midiNote": midi_note,
        "scale": scale,
        # Static confidence value — there's no real probabilistic model behind
        # the analysis yet, so we report a fixed 0.9 when results exist.
        "confidence": 0.9,
        "tempo": int(round(tempo)),
        "notes": [note],
        "pitchTimeline": pitch_timeline,
    }
