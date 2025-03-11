from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, AnyUrl, validator
from datetime import datetime
from enum import Enum
import re
import uuid


class MessageRole(str, Enum):
    """
    Role of the message sender
    """
    USER = "user"
    BOT = "bot"
    SYSTEM = "system"


class Message(BaseModel):
    """
    Chat message model
    """
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), description="Message ID")
    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ChatRequest(BaseModel):
    """
    Request model for sending a message to the chatbot
    """
    message: str = Field(..., description="Message content")
    session_id: Optional[str] = Field(None, description="Session ID for continuing a conversation")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for the message")


class ChatResponse(BaseModel):
    """
    Response model from the chatbot
    """
    message: Message = Field(..., description="Bot's response message")
    session_id: str = Field(..., description="Session ID")
    suggestions: Optional[List[str]] = Field(None, description="Suggested follow-up messages")


class ConversationResponse(BaseModel):
    """
    Response model for conversation history
    """
    conversation_id: str = Field(..., description="Conversation ID")
    messages: List[Message] = Field(default_factory=list, description="List of messages in the conversation")
    created_at: datetime = Field(..., description="Conversation creation timestamp")
    updated_at: datetime = Field(..., description="Conversation last update timestamp")


class ErrorResponse(BaseModel):
    """
    Error response model
    """
    status: str = Field("error", description="Status of the response")
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    errors: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


# New models for summarization

class SummaryLength(str, Enum):
    """
    Options for summary length
    """
    SHORT = "short"  # ~1-2 paragraphs
    MEDIUM = "medium"  # ~3-5 paragraphs
    LONG = "long"  # ~5-7 paragraphs


class SummaryStyle(str, Enum):
    """
    Options for summary style
    """
    BULLET_POINTS = "bullet_points"  # Bullet point format
    NARRATIVE = "narrative"  # Flowing text format
    ACADEMIC = "academic"  # Formal, structured format
    SIMPLIFIED = "simplified"  # Easy to understand, non-technical


class YouTubeSummaryRequest(BaseModel):
    """
    Request model for summarizing a YouTube video
    """
    video_url: str = Field(..., description="URL of the YouTube video")
    summary_length: SummaryLength = Field(default=SummaryLength.MEDIUM, description="Desired length of the summary")
    summary_style: SummaryStyle = Field(default=SummaryStyle.NARRATIVE, description="Style of the summary")
    include_timestamps: bool = Field(default=True, description="Whether to include timestamps in the summary")
    focus_topics: Optional[List[str]] = Field(None, description="Specific topics to focus on in the summary")

    @validator('video_url')
    def must_be_youtube_url(cls, v):
        if not ('youtube.com' in v or 'youtu.be' in v):
            raise ValueError('URL must be a valid YouTube URL')
        return v

    def extract_video_id(self) -> Optional[str]:
        """Extract YouTube video ID from the URL"""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, self.video_url)
            if match:
                return match.group(1)

        return None


class PDFSummaryRequest(BaseModel):
    """
    Request model for summarizing a PDF document
    """
    summary_length: SummaryLength = Field(default=SummaryLength.MEDIUM, description="Desired length of the summary")
    summary_style: SummaryStyle = Field(default=SummaryStyle.NARRATIVE, description="Style of the summary")
    focus_topics: Optional[List[str]] = Field(None, description="Specific topics to focus on in the summary")
    page_range: Optional[str] = Field(None, description="Range of pages to summarize (e.g., '1-5,10,15-20')")


class TextSummaryRequest(BaseModel):
    """
    Request model for summarizing text content
    """
    text: str = Field(..., description="Text content to summarize")
    summary_length: SummaryLength = Field(default=SummaryLength.MEDIUM, description="Desired length of the summary")
    summary_style: SummaryStyle = Field(default=SummaryStyle.NARRATIVE, description="Style of the summary")
    focus_topics: Optional[List[str]] = Field(None, description="Specific topics to focus on in the summary")


class TimeStampedSection(BaseModel):
    """
    Model for a section of content with a timestamp
    """
    time: str = Field(..., description="Timestamp in format 'MM:SS' or 'HH:MM:SS'")
    text: str = Field(..., description="Content at this timestamp")


class SummaryResponse(BaseModel):
    """
    Response model for summarized content
    """
    summary: str = Field(..., description="The generated summary")
    source_type: str = Field(..., description="Type of the original content (PDF, YouTube, text)")
    source_info: Dict[str, Any] = Field(..., description="Information about the original source")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the summary was generated")
    sections: Optional[List[TimeStampedSection]] = Field(None, description="Timestamped sections (for videos)")
    key_points: Optional[List[str]] = Field(None, description="Key points extracted from the content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata about the summarization")