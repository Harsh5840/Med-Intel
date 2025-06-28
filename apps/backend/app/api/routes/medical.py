from fastapi import APIRouter, HTTPException
from app.models.request import MedicalQueryRequest
from app.models.response import MedicalQueryResponse
from packages.rag_engine.query_engine import get_medical_answer

router = APIRouter()

@router.post("/query", response_model=MedicalQueryResponse)
async def medical_query(payload: MedicalQueryRequest):
    try:
        result = get_medical_answer(payload.query)
        return MedicalQueryResponse(answer=result.answer, sources=result.sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
