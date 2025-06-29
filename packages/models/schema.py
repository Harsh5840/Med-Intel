

from typing import List, Optional
from pydantic import BaseModel


class SymptomInput(BaseModel):
    symptoms: List[str]


class DiagnosisResponse(BaseModel):
    diagnosis: str
    confidence: Optional[float] = None


class SpecialtyPrediction(BaseModel):
    specialty: str
    confidence: Optional[float] = None


class ErrorResponse(BaseModel):
    detail: str
