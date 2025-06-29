# packages/models/schema.py

from pydantic import BaseModel, Field
from typing import List, Optional


class ClassificationRequest(BaseModel):
    text: str = Field(..., example="Patient shows signs of myocardial infarction.")


class ClassificationResponse(BaseModel):
    predicted_label: str
    confidence: Optional[float] = None  # Optional if softmax is used


class EmbeddingRequest(BaseModel):
    text: str = Field(..., example="The MRI results indicate possible glioblastoma.")


class EmbeddingResponse(BaseModel):
    embedding: List[float]


class TrainRequest(BaseModel):
    csv_path: str = Field(..., example="/data/cleaned_dataset.csv")
    text_column: str = Field(default="abstract")
    label_column: str = Field(default="category")
