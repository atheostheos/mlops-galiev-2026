from typing import Literal, Optional

from pydantic import BaseModel, Field

ContractType = Literal["month-to-month", "one_year", "two_year"]
InternetService = Literal["fiber", "dsl", "none"]
PaymentMethod = Literal["electronic_check", "mailed_check", "bank_transfer", "credit_card"]


class HealthResponse(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "ok",
                    "model_loaded": True,
                    "model_version": "registry:churn-classifier@champion",
                }
            ]
        }
    }

    status: str
    model_loaded: bool
    model_version: str


class Customer(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tenure_months": 24,
                    "monthly_charges": 75.5,
                    "total_charges": 1812.0,
                    "contract_type": "one_year",
                    "internet_service": "fiber",
                    "payment_method": "electronic_check",
                    "num_support_calls": 2,
                    "has_tech_support": 1,
                    "is_senior": 0,
                    "avg_monthly_gb": 250.0,
                }
            ]
        }
    }

    tenure_months: int = Field(..., ge=0, le=200, description="Стаж клиента в месяцах")
    monthly_charges: float = Field(..., ge=0, le=1000)
    total_charges: Optional[float] = Field(None, ge=0)
    contract_type: ContractType
    internet_service: InternetService
    payment_method: PaymentMethod
    num_support_calls: int = Field(..., ge=0, le=100)
    has_tech_support: int = Field(..., ge=0, le=1)
    is_senior: int = Field(..., ge=0, le=1)
    avg_monthly_gb: float = Field(..., ge=0, le=5000)


class BatchRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "items": [
                        {
                            "tenure_months": 24,
                            "monthly_charges": 75.5,
                            "total_charges": 1812.0,
                            "contract_type": "one_year",
                            "internet_service": "fiber",
                            "payment_method": "electronic_check",
                            "num_support_calls": 2,
                            "has_tech_support": 1,
                            "is_senior": 0,
                            "avg_monthly_gb": 250.0,
                        }
                    ]
                }
            ]
        }
    }

    # Верхняя граница обязательна: запрос на миллион строк положит сервис.
    # Понятный отказ лучше, чем OOM.
    items: list[Customer] = Field(..., min_length=1, max_length=1000)


class Prediction(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "churn_probability": 0.42,
                    "churn": 1,
                    "threshold": 0.245,
                    "model_version": "registry:churn-classifier@champion",
                }
            ]
        }
    }

    churn_probability: float
    churn: int
    threshold: float
    model_version: str


class BatchPrediction(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "count": 2,
                    "predictions": [
                        {
                            "churn_probability": 0.42,
                            "churn": 1,
                            "threshold": 0.245,
                            "model_version": "registry:churn-classifier@champion",
                        },
                        {
                            "churn_probability": 0.18,
                            "churn": 0,
                            "threshold": 0.245,
                            "model_version": "registry:churn-classifier@champion",
                        },
                    ],
                }
            ]
        }
    }

    count: int
    predictions: list[Prediction]


class Explanation(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "method": "linear_contribution",
                    "contributions": [
                        {"feature": "cat__contract_type_month-to-month", "contribution": 0.31},
                        {"feature": "num__tenure_months", "contribution": -0.24},
                        {"feature": "num__monthly_charges", "contribution": 0.15},
                        {"feature": "cat__internet_service_fiber", "contribution": 0.09},
                        {"feature": "num__num_support_calls", "contribution": 0.06},
                    ],
                    "model_version": "registry:churn-classifier@champion",
                }
            ]
        }
    }

    method: str
    contributions: list[dict]
    model_version: str
