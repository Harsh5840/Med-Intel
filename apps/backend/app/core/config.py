from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "MedIntel"
    environment: str = "development"
    model_path: str = "models/your_model"
    chroma_db_path: str = "packages/embeddings/chroma_db"
    enable_debug: bool = True

    class CustomConfig:
        env_file = ".env"

settings = Settings()
