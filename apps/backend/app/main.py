from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import medical  # ✅ corrected import: 'routed' → 'routes'
from app.core.config import settings  # ✅ assuming you have a settings object

app = FastAPI(
    title="MedIntel API",
    description="Distributed Medical Knowledge Assistant",
    version="0.1.0"
)

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ change this in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include medical routes
app.include_router(medical.router, prefix="/api")

# Health check
@app.get("/health")
def health_check():
    return {"status": "ok"}

