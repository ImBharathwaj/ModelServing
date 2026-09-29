from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_path: str
    model_version: str
    api_env: str = "development"
    log_level: str = "INFO"

    class Config:
        env_file='.env'

settings = Settings()