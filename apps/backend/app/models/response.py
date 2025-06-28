from pydantic import BaseModel
from typing import List, Optional

class SourceDocument(BaseModel):
    title: str
    url: Optional[str]
    snippet: Optional[str]

class MedicalQueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
