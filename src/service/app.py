from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.config import feature_columns, load_params
from src.logging_setup import setup_logging
from src.service.middleware import RequestIDMiddleware
from src.service.model_loader import holder
from src.service.schemas import (
    BatchPrediction,
    BatchRequest,
    Customer,
    Explanation,
    HealthResponse,
    Prediction,
)

log = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Модель грузится ОДИН раз на старте. Если грузить в обработчике,
    # каждый запрос будет идти секундами.
    try:
        holder.load()
    except Exception as exc:
        log.error("не удалось загрузить модель: %s", exc)
    yield


app = FastAPI(
    title="Churn Prediction Service",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(RequestIDMiddleware)


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    """Жив ли процесс и загружена ли модель."""
    return HealthResponse(
        status="ok" if holder.loaded else "degraded",
        model_loaded=holder.loaded,
        model_version=holder.version,
    )


@app.post("/predict", response_model=Prediction, tags=["inference"])
def predict(customer: Customer) -> Prediction:
    params = load_params()
    threshold = params["evaluate"]["threshold"]
    if not holder.loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    df = pd.DataFrame([customer.model_dump()])[feature_columns(params)]
    proba = float(holder.model.predict_proba(df)[0, 1])
    return Prediction(
        churn_probability=round(proba, 6),
        churn=int(proba >= threshold),
        threshold=threshold,
        model_version=holder.version,
    )


@app.post("/predict/batch", response_model=BatchPrediction, tags=["inference"])
def predict_batch(customers: BatchRequest) -> BatchPrediction:
    params = load_params()
    threshold = params["evaluate"]["threshold"]
    if not holder.loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    df = pd.DataFrame([c.model_dump() for c in customers.items])[feature_columns(params)]
    probas = holder.model.predict_proba(df)[:, 1]
    return BatchPrediction(
        count=len(probas),
        predictions=[
            Prediction(
                churn_probability=round(float(proba), 6),
                churn=int(proba >= threshold),
                threshold=threshold,
                model_version=holder.version,
            )
            for proba in probas
        ],
    )


@app.post("/explain", response_model=Explanation, tags=["inference"])
def explain(customer: Customer) -> Explanation:
    """Вклад каждого признака в предсказание.

    Для линейной модели это коэффициент, умноженный на значение признака
    после препроцессинга. Для деревьев — feature_importances_, они
    глобальные, а не для конкретного клиента, и это нужно честно писать
    в ответе, чтобы бизнес не принял одно за другое.
    """
    params = load_params()
    if not holder.loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    df = pd.DataFrame([customer.model_dump()])[feature_columns(params)]
    explanation = holder.explain(df)
    return Explanation(
        method=explanation["method"],
        contributions=explanation["contributions"],
        model_version=holder.version,
    )


@app.post("/reload", response_model=HealthResponse, tags=["service"])
def reload_model() -> HealthResponse:
    holder.load()
    return HealthResponse(
        status="ok",
        model_loaded=holder.loaded,
        model_version=holder.version,
    )
