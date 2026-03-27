import opensmile
import librosa
import numpy as np

"""
eGeMAPSv02 is an audio encoding that INTERSPEECH (largest conference in the field of speech) for "paralingual" tasks (emotion, personality, likability, etc.). See: https://ieeexplore.ieee.org/document/7160715 for more
"""


class ProsodicFeatureExtractor:
    def __init__(self):
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    def extract_trust_relevant(self, audio, sample_rate):
        """
        Features of interest:
        Pitch variability and dynamism: deceptive
        Speech rate change
        Energy patterns
        Voiced ration

        CITATION NEEDED. CLAUDE TOLD ME THESE WHERE WHAT TO LOOK FOR.

        """

        # Pitch
        f0, voice_flag, _ = librosa.pyin(audio, fmin=50, fmax=500, sr=sample_rate)
        valid_f0 = f0[~np.isnan(f0)]

        # Speech rate (zero crossing rate, i.e. how many times the audio signal meets zero)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]

        # Energy contour via root mean square
        rms = librosa.feature.rms(audio)[0]

        pitch_features = [
            np.mean(valid_f0),  # mean pitch
            np.std(valid_f0),  # pitch variation
            np.max(valid_f0) - np.min(valid_f0),  # pitch range
            np.mean(np.abs(np.diff(valid_f0))),  # pitch dynamism
        ]

        features = np.concat(pitch_features + [zcr] + [rms])  # probabily correct

        return np.array(features, dtype=np.float32)

    def extract(self, audio: np.ndarray, sample_rate=16000) -> dict:
        feature_df = self.smile.process_signal(audio, sample_rate)
        feature_vector = feature_df.values.flatten()
        trust_features = self.extract_trust_relevant(audio, sample_rate)

        return {
            "opensmile_features": feature_vector,  # 88-dim
            "trust_features": trust_features,
        }
