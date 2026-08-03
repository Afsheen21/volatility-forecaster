from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_predict_valid_ticker():
    response = client.get("/predict/AAPL")
    assert response.status_code == 200
    data = response.json()
    assert "predicted_next_day_volatility" in data
    assert isinstance(data["predicted_next_day_volatility"], float)

def test_predict_invalid_ticker():
    response = client.get("/predict/INVALIDXYZ123")
    assert response.status_code == 200
    data = response.json()
    assert "error" in data