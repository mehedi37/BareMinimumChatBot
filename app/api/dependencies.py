from fastapi import Depends
from typing import Optional
from loguru import logger

from app.services.chatbot import ChatbotService
from app.services.summarizer import SummarizerService

# Cached services for reuse
_summarizer_service = None
_chatbot_service = None

async def get_chatbot_service():
    """
    Get or create the chatbot service with proper async initialization
    """
    global _chatbot_service
    if _chatbot_service is None:
        logger.info("Creating new ChatbotService instance")
        _chatbot_service = ChatbotService()
        # Ensure Ollama service is properly initialized in async context
        await _chatbot_service.initialize()
    return _chatbot_service

async def get_summarizer_service():
    """
    Get or create the summarizer service with proper async initialization
    """
    global _summarizer_service
    if _summarizer_service is None:
        logger.info("Creating new SummarizerService instance")
        _summarizer_service = SummarizerService()
        # Ensure Ollama service is properly initialized in async context
        await _summarizer_service.initialize()
    return _summarizer_service