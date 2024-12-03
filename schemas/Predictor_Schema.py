from pydantic import BaseModel

class PredictionResponse(BaseModel):
    tumor_class: str
    confidence: float
    probabilidad: float