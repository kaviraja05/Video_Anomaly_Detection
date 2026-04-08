from fastapi.testclient import TestClient
from backend_api import app

client = TestClient(app)

def test_protected_route():
    # Attempting to fetch preprocessing-proof should be protected
    response = client.get("/preprocessing-proof")
    assert response.status_code == 401

def test_prediction_api():
    # Prediction API without auth token
    response = client.post("/predict")
    assert response.status_code == 401
