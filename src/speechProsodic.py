import opensmile
import numpy as np

"""
eGeMAPSv02 is an audio encoding that INTERSPEECH (largest conference in the field of speech) for "paralingual" tasks (emotion, personality, likability, etc.). See: https://ieeexplore.ieee.org/document/7160715 for more.

"""


class ProsodicFeatureExtractor:
    def __init__(self):
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    def extract(self, audio: np.ndarray, sample_rate=16000) -> dict:
        prosodic_features_names = [
            "F0semitoneFrom27.5Hz_sma3nz_amean",
            "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
            "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",  # pitch range 20th-80th percentile
            "F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope",  # intonation shape
            "F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope",  # intonation shape
            "loudness_sma3_amean",
            "loudness_sma3_stddevNorm",
            "loudnessPeaksPerSec",  # speach rate/rhythe proxy
            "alphaRatioV_sma3nz_amean",  # energy balance around 1kHz (higher = tense, lower = relaxed)
            "hammarbergIndexV_sma3nz_amean",  # energy peak ratio
            "jitterLocal_sma3nz_amean",  # pitch pertubation (linked with stress)
            "shimmerLocaldB_sma3nz_amean",  # amplitude perturbation
            "HNRdBACF_sma3nz_amean",  # harmonics-to-noise ratio (breathy voice)
            "F1frequency_sma3nz_amean",  # formant position (correlates with jaw motion)
            "F2frequency_sma3nz_amean",  # formant position (correlates with tongue position)
            "F3frequency_sma3nz_amean",  # formant position (color/brightness of voice)
            "MeanVoicedSegmentLengthSec",  # pause pattern variability
            "MeanUnvoicedSegmentLength",  # average pause duration
            "StddevUnvoicedSegmentLength",  # speech fluency
        ]

        feature_df = self.smile.process_signal(audio, sample_rate)
        trust_features = np.nan_to_num(
            feature_df[prosodic_features_names].values.flatten(), nan=0.0
        )
        features = np.nan_to_num(feature_df.values.flatten(), nan=0.0)
        feature_names = list(feature_df.columns)

        return {
            "all_features": features,  # 88-dim
            "feature_names": feature_names,
            "trust_features": trust_features,  # 19-dim
            "trust_feature_names:": prosodic_features_names,
        }
