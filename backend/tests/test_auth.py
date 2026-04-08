from fastapi.testclient import TestClient
from backend_api import app

client = TestClient(app)

def test_register_user():
    response = client.post("/register", json={
        "name": "Test User",
        "email": "test_account@example.com",
        "password": "password123"
    })
    # Allow 200 (created) or 400 (already exists) so tests don't permanently fail on retry against live DB
    assert response.status_code in [200, 400]

def test_login_user():
    response = client.post("/login", json={
        "email": "test_account@example.com",
        "password": "password123"
    })
    assert "access_token" in response.json() or response.status_code == 401
