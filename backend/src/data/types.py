from pydantic import BaseModel

class MLInferenceResult(BaseModel):
    technique_counts: dict
    ralley_length: int
    forehand_percentage: float
    backhand_percentage: float

