from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_check_returns_healthy_status():
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["success"] is True
    assert body["message"] == "AI service is healthy"
    assert body["data"]["status"] == "healthy"
    assert body["data"]["service"] == "ai-recruiter-mini-ai-service"