import os
import re
import tempfile
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import uuid
import validators
from pathlib import Path
import aiofiles
from loguru import logger
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.formatters import TextFormatter
import pdfplumber
import httpx
import spacy
from bs4 import BeautifulSoup

from app.models.schema import (
    SummaryLength,
    SummaryStyle,
    SummaryResponse,
    TimeStampedSection,
    YouTubeSummaryRequest,
    PDFSummaryRequest,
    TextSummaryRequest,
    OutputType,
    QuizResponse,
    QuizQuestion
)
from app.core.errors import ChatbotException, BadRequestException
from app.services.ollama_service import OllamaService

env_model = os.getenv('CHATBOT_MODEL')

class SummarizerService:
    """
    Service for handling summarization of different content types
    """
    def __init__(self):
        """Initialize the summarizer service with the Ollama service and YouTube transcripts cache"""
        logger.info("Initializing SummarizerService...")

        # Get the model name from environment variables
        env_model = os.getenv('CHATBOT_MODEL')

        # Initialize the Ollama service
        self.ollama_service = OllamaService(model_name=env_model)
        logger.info(f"Initialized Ollama service with model: {env_model}")

        # YouTube transcripts cache
        self.transcripts_cache = {}

        # Load SpaCy for basic text processing
        try:
            # Try to load the English model, or download it if not available
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded SpaCy en_core_web_sm model")
        except OSError:
            logger.warning("SpaCy model not found, downloading en_core_web_sm...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        logger.info("SummarizerService initialized successfully")

    async def initialize(self):
        """
        Properly initialize async components of the service
        """
        logger.info("Initializing async components of SummarizerService")
        # Initialize the Ollama service properly in an async context
        await self.ollama_service.initialize()
        logger.info("SummarizerService async initialization complete")

    async def summarize_youtube_video(self, request: YouTubeSummaryRequest) -> Union[SummaryResponse, QuizResponse]:
        """
        Summarize a YouTube video using its transcript or create a quiz
        """
        try:
            # Extract video ID from URL
            video_id = request.extract_video_id()
            if not video_id:
                raise BadRequestException(detail="Invalid YouTube URL format")

            # Fetch video metadata
            metadata = await self._fetch_youtube_metadata(video_id)

            # Get transcript
            transcript_list = self._get_youtube_transcript(video_id)

            # Format transcript to plain text
            formatter = TextFormatter()
            transcript_text = formatter.format_transcript(transcript_list)

            # Handle based on output type
            if request.output_type == OutputType.QUIZ:
                # Generate quiz using Ollama
                quiz_questions = await self.ollama_service.create_quiz(
                    content=transcript_text,
                    num_questions=request.num_quiz_questions or 5,
                    focus_topics=request.focus_topics
                )

                # Convert to QuizQuestion model objects
                questions = []
                for q in quiz_questions:
                    questions.append(QuizQuestion(
                        question=q["question"],
                        options=q["options"],
                        correct_answer=q["correct_answer"],
                        explanation=q.get("explanation")
                    ))

                # Create response
                return QuizResponse(
                    questions=questions,
                    source_type="youtube",
                    source_info=metadata,
                    metadata={
                        "video_id": video_id,
                        "focus_topics": request.focus_topics,
                        "num_questions": request.num_quiz_questions
                    }
                )
            else:
                # Extract timestamped sections if requested
                sections = None
                if request.include_timestamps:
                    sections = self._extract_timestamped_sections(transcript_list)

                # Generate summary using Ollama
                summary = await self.ollama_service.generate_summary(
                    content=transcript_text,
                    style=request.summary_style.value,
                    length=request.summary_length.value,
                    focus_topics=request.focus_topics
                )

                # Extract key points using Ollama
                key_points = await self.ollama_service.extract_key_points(summary)

                # Create response
                return SummaryResponse(
                    summary=summary,
                    source_type="youtube",
                    source_info=metadata,
                    sections=sections,
                    key_points=key_points,
                    metadata={
                        "video_id": video_id,
                        "summary_style": request.summary_style,
                        "summary_length": request.summary_length,
                        "focus_topics": request.focus_topics
                    }
                )

        except (TranscriptsDisabled, NoTranscriptFound) as e:
            raise BadRequestException(detail=f"Could not retrieve transcript: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing YouTube video: {str(e)}")
            raise ChatbotException(detail=f"Failed to process YouTube video: {str(e)}")

    async def summarize_pdf(self, file_content: bytes, request: PDFSummaryRequest) -> Union[SummaryResponse, QuizResponse]:
        """
        Summarize a PDF document or create a quiz
        """
        try:
            # Create a temporary file to store the PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            # Extract text from PDF
            text, page_count = self._extract_text_from_pdf(temp_file_path, request.page_range)

            # Clean up the temporary file
            os.unlink(temp_file_path)

            if not text.strip():
                raise BadRequestException(detail="Could not extract text from PDF. The file may be scanned or protected.")

            # Handle based on output type
            if request.output_type == OutputType.QUIZ:
                # Generate quiz using Ollama
                quiz_questions = await self.ollama_service.create_quiz(
                    content=text,
                    num_questions=request.num_quiz_questions or 5,
                    focus_topics=request.focus_topics
                )

                # Convert to QuizQuestion model objects
                questions = []
                for q in quiz_questions:
                    questions.append(QuizQuestion(
                        question=q["question"],
                        options=q["options"],
                        correct_answer=q["correct_answer"],
                        explanation=q.get("explanation")
                    ))

                # Create response
                return QuizResponse(
                    questions=questions,
                    source_type="pdf",
                    source_info={
                        "page_count": page_count,
                        "processed_pages": request.page_range or f"1-{page_count}",
                        "text_length": len(text)
                    },
                    metadata={
                        "focus_topics": request.focus_topics,
                        "num_questions": request.num_quiz_questions
                    }
                )
            else:
                # Generate summary using Ollama
                summary = await self.ollama_service.generate_summary(
                    content=text,
                    style=request.summary_style.value,
                    length=request.summary_length.value,
                    focus_topics=request.focus_topics
                )

                # Extract key points using Ollama
                key_points = await self.ollama_service.extract_key_points(summary)

                # Create response
                return SummaryResponse(
                    summary=summary,
                    source_type="pdf",
                    source_info={
                        "page_count": page_count,
                        "processed_pages": request.page_range or f"1-{page_count}",
                        "text_length": len(text)
                    },
                    key_points=key_points,
                    sections=None,  # PDFs don't have timestamped sections
                    metadata={
                        "summary_style": request.summary_style,
                        "summary_length": request.summary_length,
                        "focus_topics": request.focus_topics
                    }
                )

        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise ChatbotException(detail=f"Failed to process PDF: {str(e)}")

    async def summarize_text(self, request: TextSummaryRequest) -> Union[SummaryResponse, QuizResponse]:
        """
        Summarize text content or create a quiz
        """
        try:
            if not request.text.strip():
                raise BadRequestException(detail="Text content cannot be empty")

            # Handle based on output type
            if request.output_type == OutputType.QUIZ:
                # Generate quiz using Ollama
                quiz_questions = await self.ollama_service.create_quiz(
                    content=request.text,
                    num_questions=request.num_quiz_questions or 5,
                    focus_topics=request.focus_topics
                )

                # Convert to QuizQuestion model objects
                questions = []
                for q in quiz_questions:
                    questions.append(QuizQuestion(
                        question=q["question"],
                        options=q["options"],
                        correct_answer=q["correct_answer"],
                        explanation=q.get("explanation")
                    ))

                # Create response
                return QuizResponse(
                    questions=questions,
                    source_type="text",
                    source_info={
                        "text_length": len(request.text),
                        "word_count": len(request.text.split())
                    },
                    metadata={
                        "focus_topics": request.focus_topics,
                        "num_questions": request.num_quiz_questions
                    }
                )
            else:
                # Generate summary using Ollama
                summary = await self.ollama_service.generate_summary(
                    content=request.text,
                    style=request.summary_style.value,
                    length=request.summary_length.value,
                    focus_topics=request.focus_topics
                )

                # Extract key points using Ollama
                key_points = await self.ollama_service.extract_key_points(summary)

                # Create response
                return SummaryResponse(
                    summary=summary,
                    source_type="text",
                    source_info={
                        "text_length": len(request.text),
                        "word_count": len(request.text.split())
                    },
                    key_points=key_points,
                    sections=None,  # Text doesn't have timestamped sections
                    metadata={
                        "summary_style": request.summary_style,
                        "summary_length": request.summary_length,
                        "focus_topics": request.focus_topics
                    }
                )

        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            raise ChatbotException(detail=f"Failed to process text: {str(e)}")

    async def _fetch_youtube_metadata(self, video_id: str) -> Dict[str, Any]:
        """
        Fetch metadata for a YouTube video
        """
        try:
            # Use YouTube's oEmbed API to get basic video info
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"

            async with httpx.AsyncClient() as client:
                response = await client.get(oembed_url)

                if response.status_code == 200:
                    data = response.json()
                    return {
                        "title": data.get("title", f"YouTube Video {video_id}"),
                        "author": data.get("author_name", "Unknown"),
                        "author_url": data.get("author_url", ""),
                        "thumbnail_url": data.get("thumbnail_url", "")
                    }

                # Fallback to scraping if oEmbed fails
                page_url = f"https://www.youtube.com/watch?v={video_id}"
                page_response = await client.get(page_url)

                if page_response.status_code == 200:
                    soup = BeautifulSoup(page_response.text, 'html.parser')
                    title = soup.find('title')
                    title_text = title.text.replace(' - YouTube', '') if title else f"YouTube Video {video_id}"

                    return {
                        "title": title_text,
                        "video_id": video_id,
                        "url": page_url
                    }

                # If all else fails, return minimal info
                return {
                    "title": f"YouTube Video {video_id}",
                    "video_id": video_id,
                    "url": f"https://www.youtube.com/watch?v={video_id}"
                }

        except Exception as e:
            logger.warning(f"Error fetching YouTube metadata: {str(e)}")
            # Return minimal info on error
            return {
                "title": f"YouTube Video {video_id}",
                "video_id": video_id,
                "url": f"https://www.youtube.com/watch?v={video_id}"
            }

    def _get_youtube_transcript(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Get transcript for a YouTube video
        """
        try:
            # Try to get transcript in English
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            return transcript_list
        except NoTranscriptFound:
            # If no English transcript, try to get any available transcript and translate it
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                return transcript_list
            except:
                # If that fails too, try to get auto-generated transcript
                try:
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en-US', 'en-GB', 'en'])
                    return transcript_list
                except Exception as e:
                    # Create a more specific error message but use the proper exception format
                    raise ChatbotException(detail=f"No transcript found for video {video_id}: {str(e)}")
        except Exception as e:
            raise ChatbotException(detail=f"Error getting YouTube transcript: {str(e)}")

    def _extract_timestamped_sections(self, transcript_list: List[Dict[str, Any]]) -> List[TimeStampedSection]:
        """
        Extract timestamped sections from a transcript
        """
        # Group transcript entries into logical sections
        sections = []
        current_section_text = ""
        current_section_start = 0

        # Parameters for section detection
        min_section_duration = 60  # Minimum section duration in seconds
        max_pause_in_section = 3   # Maximum pause between entries in the same section

        for i, entry in enumerate(transcript_list):
            # Get current entry details
            start = entry.get('start', 0)
            duration = entry.get('duration', 0)
            text = entry.get('text', '')

            # If this is the first entry, initialize the section
            if i == 0:
                current_section_text = text
                current_section_start = start
                continue

            # Get previous entry details
            prev_start = transcript_list[i-1].get('start', 0)
            prev_duration = transcript_list[i-1].get('duration', 0)
            prev_end = prev_start + prev_duration

            # Check if there's a significant pause or if the current section is long enough
            if (start - prev_end > max_pause_in_section) or (start - current_section_start > min_section_duration):
                # Add the current section to the list
                sections.append(TimeStampedSection(
                    time=self._format_timestamp(current_section_start),
                    text=current_section_text.strip()
                ))

                # Start a new section
                current_section_text = text
                current_section_start = start
            else:
                # Continue the current section
                current_section_text += " " + text

        # Add the last section
        if current_section_text:
            sections.append(TimeStampedSection(
                time=self._format_timestamp(current_section_start),
                text=current_section_text.strip()
            ))

        return sections

    def _format_timestamp(self, seconds: float) -> str:
        """
        Format seconds as MM:SS or HH:MM:SS
        """
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"

    def _extract_text_from_pdf(self, pdf_path: str, page_range: Optional[str] = None) -> Tuple[str, int]:
        """
        Extract text from a PDF file
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)

                # Parse page range if provided
                pages_to_extract = self._parse_page_range(page_range, total_pages) if page_range else range(1, total_pages + 1)

                # Extract text from specified pages
                text_parts = []
                for page_num in pages_to_extract:
                    if 1 <= page_num <= total_pages:
                        page = pdf.pages[page_num - 1]  # pdfplumber uses 0-based indexing
                        text = page.extract_text() or ""
                        text_parts.append(text)

                return "\n\n".join(text_parts), total_pages

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise ChatbotException(detail=f"Failed to extract text from PDF: {str(e)}")

    def _parse_page_range(self, page_range_str: str, max_pages: int) -> List[int]:
        """
        Parse a page range string like "1-5,7,9-11" into a list of page numbers
        """
        if not page_range_str:
            return list(range(1, max_pages + 1))

        pages = []
        parts = page_range_str.split(',')

        for part in parts:
            part = part.strip()

            if '-' in part:
                # Handle range like "1-5"
                try:
                    start, end = map(int, part.split('-'))
                    # Ensure start <= end and both are within bounds
                    start = max(1, min(start, max_pages))
                    end = max(start, min(end, max_pages))
                    pages.extend(range(start, end + 1))
                except ValueError:
                    # Skip invalid ranges
                    continue
            else:
                # Handle single page like "7"
                try:
                    page = int(part)
                    if 1 <= page <= max_pages:
                        pages.append(page)
                except ValueError:
                    # Skip invalid page numbers
                    continue

        # Remove duplicates and sort
        return sorted(set(pages))