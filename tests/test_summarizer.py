import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from main import app
from app.models.schema import SummaryLength, SummaryStyle, SummaryResponse, TimeStampedSection


client = TestClient(app)


@pytest.fixture
def mock_api_key():
    return "test_api_key"


@pytest.fixture
def mock_headers(mock_api_key):
    return {"X-API-Key": mock_api_key}


@patch("app.services.summarizer.SummarizerService.summarize_youtube_video")
async def test_youtube_summarization(mock_summarize, mock_headers):
    # Mock response
    mock_response = SummaryResponse(
        summary="Test summary for YouTube video",
        source_type="youtube",
        source_info={
            "video_id": "test123",
            "title": "Test Video",
            "channel": "Test Channel",
            "duration": "10:00",
            "url": "https://www.youtube.com/watch?v=test123"
        },
        sections=[
            TimeStampedSection(time="00:00", text="Introduction"),
            TimeStampedSection(time="02:30", text="Main content")
        ],
        key_points=["Point 1", "Point 2"],
        metadata={
            "summary_length": SummaryLength.MEDIUM,
            "summary_style": SummaryStyle.NARRATIVE
        }
    )
    mock_summarize.return_value = mock_response

    # Test request
    request_data = {
        "video_url": "https://www.youtube.com/watch?v=test123",
        "summary_length": "medium",
        "summary_style": "narrative",
        "include_timestamps": True
    }

    response = client.post(
        "/api/v1/summarize/youtube",
        json=request_data,
        headers=mock_headers
    )

    assert response.status_code == 200
    assert response.json()["summary"] == "Test summary for YouTube video"
    assert response.json()["source_type"] == "youtube"
    assert len(response.json()["key_points"]) == 2


@patch("app.services.summarizer.SummarizerService.summarize_text")
async def test_text_summarization(mock_summarize, mock_headers):
    # Mock response
    mock_response = SummaryResponse(
        summary="Test summary for text content",
        source_type="text",
        source_info={
            "character_count": 1000,
            "word_count": 200
        },
        sections=None,
        key_points=["Point 1", "Point 2", "Point 3"],
        metadata={
            "summary_length": SummaryLength.SHORT,
            "summary_style": SummaryStyle.BULLET_POINTS
        }
    )
    mock_summarize.return_value = mock_response

    # Test request
    request_data = {
        "text": "This is a test text that needs to be summarized. " * 50,  # Make it long enough
        "summary_length": "short",
        "summary_style": "bullet_points"
    }

    response = client.post(
        "/api/v1/summarize/text",
        json=request_data,
        headers=mock_headers
    )

    assert response.status_code == 200
    assert response.json()["summary"] == "Test summary for text content"
    assert response.json()["source_type"] == "text"
    assert len(response.json()["key_points"]) == 3