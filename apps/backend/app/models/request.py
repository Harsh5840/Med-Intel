from pydantic import BaseModel

class MedicalQueryRequest(BaseModel):
    query: str
