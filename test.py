from ohca_predictor.pipeline import OHCAPredictorPipeline
from types import type
pipe = OHCAPredictorPipeline.load("models/ohca_predictor_final.weights.h5")
pipe.predict(...)
# call signature (ecg: ndarray[_AnyShape, dtype[Any]], accelerometer: ndarray[_AnyShape, dtype[Any]], ppg: ndarray[_AnyShape, dtype[Any]], demographics: ndarray[_AnyShape, dtype[Any]], medications: ndarray[_AnyShape, dtype[Any]], comorbidities: ndarray[_AnyShape, dtype[Any]], lab_values: ndarray[_AnyShape, dtype[Any]]) -> OHCAResult

# OHCAResult Dataclass:

# @dataclass
# class OHCAResult:
#     """Structured prediction result."""

#     risk: float
#     raw_risk: float
#     uncertainty: float
#     survival: np.ndarray
#     high_risk: bool
#     risk_percentiles: Dict[str, float] = field(default_factory=dict)

#     def to_dict(self) -> Dict[str, Any]:
#         return {
#             "risk": self.risk,
#             "raw_risk": self.raw_risk,
#             "uncertainty": self.uncertainty,
#             "survival": self.survival.tolist(),
#             "high_risk": self.high_risk,
#             "risk_percentiles": self.risk_percentiles,
#         }