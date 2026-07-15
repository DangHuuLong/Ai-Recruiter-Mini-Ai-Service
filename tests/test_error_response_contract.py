from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.document_text_extraction_service import DocumentTextExtractionError

client = TestClient(app)


class TestValidationErrorConformsToApiErrorResponse:
    """docs/ai-service-contract.md Section 3 promises
    {success, message, errors[]} for every error response — without an
    explicit exception handler, FastAPI's default {"detail": [...]} shape
    leaks through instead."""

    def test_missing_required_input_returns_api_error_response_shape(self) -> None:
        response = client.post("/parse/resume", json={"resume_id": "r1"})

        assert response.status_code == 422
        body = response.json()
        assert body["success"] is False
        assert isinstance(body["message"], str) and body["message"]
        assert isinstance(body["errors"], list)
        assert "detail" not in body

    def test_signed_url_without_file_fields_returns_api_error_response_shape(self) -> None:
        response = client.post(
            "/parse/job-description",
            json={"signed_url": "https://example.com/jd.pdf"},
        )

        assert response.status_code == 422
        body = response.json()
        assert body["success"] is False
        assert isinstance(body["errors"], list)
        assert len(body["errors"]) >= 1
        assert "detail" not in body


class TestHttpExceptionConformsToApiErrorResponse:
    def test_document_extraction_error_returns_api_error_response_shape(self) -> None:
        with patch(
            "app.services.parsing_service.document_text_extraction_service.extract_from_signed_url",
            side_effect=DocumentTextExtractionError("Unsupported document file type: XYZ"),
        ):
            response = client.post(
                "/parse/resume",
                json={
                    "resume_id": "r1",
                    "signed_url": "https://example.com/cv.xyz",
                    "file_name": "cv.xyz",
                    "file_type": "XYZ",
                },
            )

        assert response.status_code == 422
        body = response.json()
        assert body["success"] is False
        assert body["message"] == "Unsupported document file type: XYZ"
        assert body["errors"] == []
        assert "detail" not in body
