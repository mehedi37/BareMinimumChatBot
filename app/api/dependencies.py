from fastapi import Depends
from typing import Optional

from app.services.chatbot import ChatbotService
from app.services.summarizer import SummarizerService


def get_chatbot_service():
    """
    Get the chatbot service
    """
    return ChatbotService()


def get_summarizer_service():
    """
    Get the summarizer service
    """
    return SummarizerService()