import pytest
from fastapi.testclient import TestClient

VALID = {
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


@pytest.fixture(scope="module")
def client(trained_pipeline):
    """Use the trained fixture instead of requiring a saved model artifact."""
    from src.service import app as app_module

    app_module.holder.model = trained_pipeline
    app_module.holder.version = "test:fixture"
    # The app lifespan normally loads a model from the registry or disk.
    # Keep the fixture model in place for these API contract tests.
    app_module.holder.load = lambda: None
    with TestClient(app_module.app) as test_client:
        yield test_client


def test_health_reports_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "test:fixture"


def test_predict_returns_valid_probability(client):
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["model_version"] == "test:fixture"


def test_missing_field_returns_422(client):
    invalid = VALID.copy()
    del invalid["tenure_months"]
    assert client.post("/predict", json=invalid).status_code == 422


def test_value_out_of_range_returns_422(client):
    assert client.post("/predict", json=dict(VALID, tenure_months=-1)).status_code == 422


def test_unknown_category_returns_422(client):
    assert client.post("/predict", json=dict(VALID, contract_type="lifetime")).status_code == 422


def test_batch_prediction_returns_predictions(client):
    response = client.post("/predict/batch", json={"items": [VALID, VALID]})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_openapi_documents_customer_and_batch_examples(client):
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    customer_example = schemas["Customer"]["examples"][0]
    batch_example = schemas["BatchRequest"]["examples"][0]
    health_example = schemas["HealthResponse"]["examples"][0]
    prediction_example = schemas["Prediction"]["examples"][0]
    batch_response_example = schemas["BatchPrediction"]["examples"][0]
    explanation_example = schemas["Explanation"]["examples"][0]
    assert customer_example["contract_type"] == "one_year"
    assert len(batch_example["items"]) == 1
    assert health_example["model_loaded"] is True
    assert prediction_example["churn"] in (0, 1)
    assert batch_response_example["count"] == len(batch_response_example["predictions"])
    assert len(explanation_example["contributions"]) == 5


def test_empty_batch_returns_422(client):
    assert client.post("/predict/batch", json={"items": []}).status_code == 422
