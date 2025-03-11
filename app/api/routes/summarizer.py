from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status, Cookie, Response
from typing import Optional, List, Union
from pydantic import parse_obj_as, HttpUrl
from datetime import datetime, timedelta

from app.models.schema import (
    YouTubeSummaryRequest,
    PDFSummaryRequest,
    TextSummaryRequest,
    SummaryResponse,
    QuizResponse,
    SummaryLength,
    SummaryStyle,
    OutputType
)
from app.services.summarizer import SummarizerService
from app.api.dependencies import get_summarizer_service
from app.core.errors import BadRequestException

router = APIRouter()


@router.post("/youtube", response_model=Union[SummaryResponse, QuizResponse])
async def summarize_youtube_video(
    video_url: str = Form(..., description="URL of the YouTube video"),
    output_type: OutputType = Form(OutputType.SUMMARY, description="Type of output to generate (summary or quiz)"),
    summary_length: SummaryLength = Form(SummaryLength.MEDIUM, description="Desired length of the summary"),
    summary_style: SummaryStyle = Form(SummaryStyle.NARRATIVE, description="Style of the summary"),
    include_timestamps: bool = Form(True, description="Whether to include timestamps in the summary"),
    focus_topics: Optional[str] = Form(None, description="Comma-separated list of topics to focus on"),
    num_quiz_questions: Optional[int] = Form(5, description="Number of quiz questions to generate"),
    session_id: Optional[str] = Cookie(None, description="Session ID from cookie"),
    response: Response = Response(),
    summarizer_service: SummarizerService = Depends(get_summarizer_service),
):
    """
    Summarize a YouTube video from its URL or generate a quiz

    Form Fields:
    - **video_url**: Full URL of the YouTube video (e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ)
    - **output_type**: Type of output to generate (summary or quiz)
    - **summary_length**: short, medium, or long (for summary output)
    - **summary_style**: narrative, bullet_points, academic, or simplified (for summary output)
    - **include_timestamps**: Whether to include timestamps in the summary (for summary output)
    - **focus_topics**: Optional comma-separated list of topics to focus on
    - **num_quiz_questions**: Number of quiz questions to generate (for quiz output)

    The session ID is automatically managed via cookies.
    """
    # Parse focus topics if provided
    parsed_focus_topics = None
    if focus_topics:
        parsed_focus_topics = [topic.strip() for topic in focus_topics.split(',') if topic.strip()]

    # Validate YouTube URL
    if not ('youtube.com' in video_url or 'youtu.be' in video_url):
        raise BadRequestException(detail="URL must be a valid YouTube URL")

    # Create request model
    request = YouTubeSummaryRequest(
        video_url=video_url,
        output_type=output_type,
        summary_length=summary_length,
        summary_style=summary_style,
        include_timestamps=include_timestamps,
        focus_topics=parsed_focus_topics,
        num_quiz_questions=num_quiz_questions
    )

    # Set session cookie if not present
    if not session_id:
        new_session_id = datetime.now().strftime("%Y%m%d%H%M%S") + "-" + video_url.split("=")[-1]
        expires = datetime.now() + timedelta(days=90)
        response.set_cookie(
            key="session_id",
            value=new_session_id,
            expires=int(expires.timestamp()),
            httponly=True,
            samesite="lax",
            secure=False,  # Set to True in production with HTTPS
            path="/"
        )

    return await summarizer_service.summarize_youtube_video(request)


@router.post("/pdf", response_model=Union[SummaryResponse, QuizResponse])
async def summarize_pdf(
    file: UploadFile = File(..., description="PDF file to summarize"),
    output_type: OutputType = Form(OutputType.SUMMARY, description="Type of output to generate (summary or quiz)"),
    summary_length: SummaryLength = Form(SummaryLength.MEDIUM, description="Desired length of the summary"),
    summary_style: SummaryStyle = Form(SummaryStyle.NARRATIVE, description="Style of the summary"),
    page_range: Optional[str] = Form(None, description="Range of pages to summarize (e.g., '1-5,10,15-20')"),
    focus_topics: Optional[str] = Form(None, description="Comma-separated list of topics to focus on"),
    num_quiz_questions: Optional[int] = Form(5, description="Number of quiz questions to generate"),
    session_id: Optional[str] = Cookie(None, description="Session ID from cookie"),
    response: Response = Response(),
    summarizer_service: SummarizerService = Depends(get_summarizer_service),
):
    """
    Summarize a PDF document from an uploaded file or generate a quiz

    Form Fields:
    - **file**: PDF file to upload
    - **output_type**: Type of output to generate (summary or quiz)
    - **summary_length**: short, medium, or long (for summary output)
    - **summary_style**: narrative, bullet_points, academic, or simplified (for summary output)
    - **page_range**: Optional range of pages to summarize (e.g., '1-5,10,15-20')
    - **focus_topics**: Optional comma-separated list of topics to focus on
    - **num_quiz_questions**: Number of quiz questions to generate (for quiz output)

    The session ID is automatically managed via cookies.
    """
    # Validate file type
    if not file.content_type or "pdf" not in file.content_type.lower():
        raise BadRequestException(detail="Uploaded file must be a PDF document")

    # Read file content
    file_content = await file.read()
    if not file_content:
        raise BadRequestException(detail="Uploaded file is empty")

    # Parse focus topics if provided
    parsed_focus_topics = None
    if focus_topics:
        parsed_focus_topics = [topic.strip() for topic in focus_topics.split(',') if topic.strip()]

    # Create request model
    request = PDFSummaryRequest(
        output_type=output_type,
        summary_length=summary_length,
        summary_style=summary_style,
        page_range=page_range,
        focus_topics=parsed_focus_topics,
        num_quiz_questions=num_quiz_questions
    )

    # Set session cookie if not present
    if not session_id:
        new_session_id = datetime.now().strftime("%Y%m%d%H%M%S") + "-pdf"
        expires = datetime.now() + timedelta(days=90)
        response.set_cookie(
            key="session_id",
            value=new_session_id,
            expires=int(expires.timestamp()),
            httponly=True,
            samesite="lax",
            secure=False,  # Set to True in production with HTTPS
            path="/"
        )

    return await summarizer_service.summarize_pdf(file_content, request)


@router.post("/text", response_model=Union[SummaryResponse, QuizResponse])
async def summarize_text(
    text: str = Form(..., description="Text content to summarize"),
    output_type: OutputType = Form(OutputType.SUMMARY, description="Type of output to generate (summary or quiz)"),
    summary_length: SummaryLength = Form(SummaryLength.MEDIUM, description="Desired length of the summary"),
    summary_style: SummaryStyle = Form(SummaryStyle.NARRATIVE, description="Style of the summary"),
    focus_topics: Optional[str] = Form(None, description="Comma-separated list of topics to focus on"),
    num_quiz_questions: Optional[int] = Form(5, description="Number of quiz questions to generate"),
    session_id: Optional[str] = Cookie(None, description="Session ID from cookie"),
    response: Response = Response(),
    summarizer_service: SummarizerService = Depends(get_summarizer_service),
):
    """
    Summarize plain text content or generate a quiz

    Form Fields:
    - **text**: Text content to summarize
    - **output_type**: Type of output to generate (summary or quiz)
    - **summary_length**: short, medium, or long (for summary output)
    - **summary_style**: narrative, bullet_points, academic, or simplified (for summary output)
    - **focus_topics**: Optional comma-separated list of topics to focus on
    - **num_quiz_questions**: Number of quiz questions to generate (for quiz output)

    The session ID is automatically managed via cookies.
    """
    if not text or len(text.strip()) < 100:
        raise BadRequestException(detail="Text content must be at least 100 characters long")

    # Parse focus topics if provided
    parsed_focus_topics = None
    if focus_topics:
        parsed_focus_topics = [topic.strip() for topic in focus_topics.split(',') if topic.strip()]

    # Create request model
    request = TextSummaryRequest(
        text=text,
        output_type=output_type,
        summary_length=summary_length,
        summary_style=summary_style,
        focus_topics=parsed_focus_topics,
        num_quiz_questions=num_quiz_questions
    )

    # Set session cookie if not present
    if not session_id:
        new_session_id = datetime.now().strftime("%Y%m%d%H%M%S") + "-text"
        expires = datetime.now() + timedelta(days=90)
        response.set_cookie(
            key="session_id",
            value=new_session_id,
            expires=int(expires.timestamp()),
            httponly=True,
            samesite="lax",
            secure=False,  # Set to True in production with HTTPS
            path="/"
        )

    return await summarizer_service.summarize_text(request)