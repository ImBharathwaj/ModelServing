import time
from app.config import settings
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib

logging.basicConfig(
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class HouseInput(BaseModel):
    area: float = Field(gt=0)
    bedrooms: int = Field(gt=0)
    bathrooms: int = Field(gt=0)
    age: int = Field(ge=0)

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    inference_time_ms: float

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Loading model: %s",settings.model_path)
    
    app.state.model = joblib.load(
        settings.model_path
    )

    logger.info("Model loaded: %s", settings.model_version)

    yield

    logger.info("Shutting down application")
    app.state.model = None

app = FastAPI(
    title="House Price Prediction API",
    version = "1.0.0",
    lifespan=lifespan
)

@app.get("/")
def home():
    return {"Message": "ML Model Serving API"}

@app.get("/health")
def health():
    if app.state.model is None:
        return {"status": "unhealthy"}
    return {"status": "ok"}

@app.post("/predict", response_model = PredictionResponse)
def predict(data: HouseInput):
    
    if app.state.model is None:
        raise HTTPException(
            status_code=503,
            details="Model is not available"
        )
    
    features = [[
        data.area,
        data.bedrooms,
        data.bathrooms,
        data.age
    ]]

    """
    app.state.model
    app.state.tokenizer
    app.state.database
    app.state.config
    """

    start = time.perf_counter()
    
    try:
        prediction = app.state.model.predict(features)
    except Exception as exc:
        logger.exception("Model inference failed")

        raise HTTPException(
            status_code=500,
            detail="Model inference failed"
        ) from exc
    
    latency = (time.perf_counter() - start) * 1000

    logger.info("Prediction completed | model=%s | latency=%.2fms", settings.model_version, latency)
    
    return PredictionResponse(
        prediction =float(prediction[0]),
        model_version=settings.model_version,
        inference_time_ms = latency
    )