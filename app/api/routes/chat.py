from fastapi import APIRouter, Depends, Form, Cookie, Response
from typing import Optional
from datetime import datetime, timedelta

from app.models.schema import ChatRequest, ChatResponse, Message
from app.services.chatbot import ChatbotService
from app.api.dependencies import get_chatbot_service
from app.core.errors import NotFoundException

router = APIRouter()


@router.post("/message", response_model=ChatResponse, status_code=200)
async def send_message(
    message: str = Form(..., description="Message content to send to the chatbot"),
    session_id: Optional[str] = Cookie(None, description="Session ID from cookie"),
    response: Response = Response(),
    chatbot_service: ChatbotService = Depends(get_chatbot_service),
):
    """
    Send a message to the chatbot and get a response

    Form Fields:
    - **message**: The message text to send to the chatbot

    The session ID is automatically managed via cookies - no need to provide it manually.
    """
    # Create request model
    request = ChatRequest(
        message=message,
        session_id=session_id,
        context=None
    )

    # Process the message
    chat_response = await chatbot_service.process_message(request)

    # Set the session ID cookie if it's a new session
    if session_id != chat_response.session_id:
        # Set cookie to expire in 3 months (typical long-lived session)
        expires = datetime.now() + timedelta(days=90)
        response.set_cookie(
            key="session_id",
            value=chat_response.session_id,
            expires=int(expires.timestamp()),  # Convert to int for compatibility
            httponly=True,
            samesite="lax",
            secure=False,  # Set to True in production with HTTPS
            path="/"
        )

    return chat_response