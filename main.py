from fastapi import FastAPI
import uvicorn
from app.api.routes import chat, summarizer
from app.core.config import settings
from app.core.logger import setup_logging
from app.api.errors import add_exception_handlers
from fastapi.middleware.cors import CORSMiddleware

# Setup logging
logger = setup_logging()

# Create FastAPI app
app = FastAPI(
    title="BareMinimum Chatbot API",
    description="API for interacting with an AI chatbot and summarizing content with various styles and formats",
    version="0.1.0",
)

# Add CORS middleware to allow browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the domains that can access the API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add exception handlers
add_exception_handlers(app)

# Include routers
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(summarizer.router, prefix="/api/v1/summarize", tags=["summarize"])

# Health check endpoint
@app.get("/api/v1/health", tags=["health"])
async def health_check():
    """
    Health check endpoint to verify API is running
    """
    return {
        "status": "ok",
        "version": "0.1.0",
        "features": {
            "chat": "Enhanced interactive chatbot with natural responses and topic awareness",
            "summarization": "Content summarization with topic detection and customizable formats",
            "quiz_generation": "Generate quizzes from content (text, PDF, YouTube) with customizable options",
            "session_management": "Browser-based cookie sessions for seamless user experience"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )