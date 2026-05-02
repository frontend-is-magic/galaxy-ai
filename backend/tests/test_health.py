from fastapi.testclient import TestClient

from app.main import app


def test_health_reports_local_backend_status():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "galaxy-ai-backend",
        "version": "0.1.0",
        "bind": {
            "host": "127.0.0.1",
            "port": 8000,
            "local_only": True,
        },
    }
