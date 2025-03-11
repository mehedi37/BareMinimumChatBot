import uuid
import random
import re
import spacy
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from loguru import logger

from app.models.schema import Message, MessageRole, ChatRequest, ChatResponse
from app.core.errors import ChatbotException
from app.services.ollama_service import OllamaService


class ChatbotService:
    """
    Service for handling chatbot interactions with more engaging responses
    """
    def __init__(self):
        # In-memory storage for session data
        self.sessions: Dict[str, List[Message]] = {}

        # Initialize Ollama service
        self.ollama_service = OllamaService(model_name="llama3:latest")
        logger.info("Initialized Ollama service with llama3:latest model")

        # Load SpaCy for NLP processing (used for pattern matching and basic NLP tasks)
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded SpaCy en_core_web_sm model")
        except OSError:
            logger.warning("SpaCy model not found, attempting to download...")
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # Define conversation topics and patterns for better responses
        self._init_knowledge_base()

        logger.info("ChatbotService initialized with enhanced response capabilities")

    def _init_knowledge_base(self):
        """Initialize the knowledge base with topics and response patterns"""
        self.greeting_patterns = [
            r"(hi|hello|hey|greetings|howdy)",
            r"(good|great) (morning|afternoon|evening|day)",
            r"(how are you|how's it going|how are things|what's up)"
        ]

        self.farewell_patterns = [
            r"(bye|goodbye|see you|farewell|later|take care)"
        ]

        self.gratitude_patterns = [
            r"(thanks|thank you|appreciate|grateful)"
        ]

        self.self_intro_patterns = [
            r"(who are you|what are you|tell me about yourself|introduce yourself)"
        ]

        self.help_patterns = [
            r"(help|assist|support|guide)",
            r"(what can you do|how do you work|your capabilities|your features)"
        ]

        self.summarize_patterns = [
            r"(summarize|summary|summarization)",
            r"(shorten|condense|brief)"
        ]

        # Response templates for various contexts
        self.greeting_responses = [
            "Hello there! How can I assist you today?",
            "Hi! I'm your AI assistant. What can I help you with?",
            "Hey! Great to see you. What would you like to know?",
            "Greetings! I'm here to help with your questions."
        ]

        self.farewell_responses = [
            "Goodbye! Feel free to return if you have more questions.",
            "Take care! I'll be here when you need assistance again.",
            "Until next time! Have a great day.",
            "Farewell! Looking forward to our next chat."
        ]

        self.gratitude_responses = [
            "You're welcome! Is there anything else I can help with?",
            "My pleasure! What else would you like to know?",
            "Happy to help! Any other questions?",
            "No problem at all! Let me know if you need anything else."
        ]

        self.self_intro_responses = [
            "I'm an AI assistant designed to help with chat interactions and content summarization. I can summarize YouTube videos, PDF documents, and text content. How can I assist you today?",
            "I'm your friendly AI chatbot! I specialize in summarizing content and engaging in meaningful conversations. What would you like me to help you with?",
            "Hello! I'm a conversational AI that can help you summarize content from various sources and answer your questions. What can I do for you today?"
        ]

        self.help_responses = [
            "I can assist with several tasks:\n- Summarize YouTube videos\n- Create summaries of PDF documents\n- Condense long text into concise summaries\n- Chat about various topics\n\nWhat would you like to do first?",
            "Here's what I can do for you:\n• Generate YouTube video summaries\n• Extract key information from PDFs\n• Create text summaries in different styles\n• Have conversations on various topics\n\nHow can I help you today?",
            "My capabilities include:\n1. Summarizing YouTube videos (with timestamps)\n2. Creating PDF document summaries\n3. Condensing text into clear summaries\n4. Engaging in interactive conversations\n\nWhat are you interested in?"
        ]

        self.summarize_info_responses = [
            "I can summarize content from YouTube videos, PDF documents, or plain text. Would you like to know more about these options?",
            "My summarization features include different styles (narrative, bullet points, academic, simplified) and lengths (short, medium, long). Would you like to try one of these options?",
            "For summarization, I can process content from various sources. You can specify the length, style, and even focus topics. Which source would you like to summarize?"
        ]

        # General knowledge for common topics
        self.topic_knowledge = {
            "weather": "I don't have real-time weather data, but I can help you summarize weather reports or articles if you provide them.",
            "news": "While I don't have access to current news, I can help summarize news articles that you provide in text form or as links to YouTube videos.",
            "technology": "Technology is constantly evolving. I can help summarize tech articles, videos, or discussions on specific technologies you're interested in.",
            "summarization": "I offer various summarization options including YouTube videos, PDF documents, and text content. You can choose different styles and lengths for your summaries.",
            "ai": "Artificial Intelligence is a broad field involving machine learning, natural language processing, and other technologies that enable systems to perform tasks that typically require human intelligence.",
            "chatbot": "Chatbots like me use natural language processing to understand and respond to user queries. I'm designed to help with summarization tasks and general conversations."
        }

        # Question starters to make the bot more interactive
        self.follow_up_questions = [
            "What would you like to know more about?",
            "Do you have any specific questions about this topic?",
            "What aspects of this are you most interested in?",
            "Would you like me to elaborate on any particular point?",
            "Is there a specific area you'd like to focus on?",
            "How can I help you further with this?"
        ]

    async def process_message(self, request: ChatRequest) -> ChatResponse:
        """
        Process a message from the user and generate a response

        Args:
            request: The chat request containing the message and session info

        Returns:
            ChatResponse with the bot's message and session info
        """
        try:
            # Get or create session ID
            session_id = request.session_id or str(uuid.uuid4())

            # Create user message
            user_message = Message(
                role=MessageRole.USER,
                content=request.message,
                timestamp=datetime.now().isoformat()
            )

            # Initialize session if it doesn't exist
            if session_id not in self.sessions:
                self.sessions[session_id] = []

            # Add user message to session
            self.sessions[session_id].append(user_message)

            # Generate response using Ollama
            response_content, suggestions = await self._generate_contextual_response(
                request.message,
                session_id,
                request.context
            )

            # Create bot message
            bot_message = Message(
                role=MessageRole.BOT,
                content=response_content,
                timestamp=datetime.now().isoformat()
            )

            # Add bot message to session
            self.sessions[session_id].append(bot_message)

            # Return response
            return ChatResponse(
                message=bot_message,
                session_id=session_id,
                suggestions=suggestions
            )

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            raise ChatbotException(detail=f"Error processing message: {str(e)}")

    async def _generate_contextual_response(self, message: str, session_id: str, context: Optional[Dict[str, Any]] = None) -> Tuple[str, List[str]]:
        """
        Generate a contextual response based on the message and conversation history

        Args:
            message: The user's message
            session_id: The session ID
            context: Optional additional context

        Returns:
            Tuple of (response content, suggested follow-up messages)
        """
        # Check for simple patterns first to avoid unnecessary LLM calls
        if any(re.search(pattern, message.lower()) for pattern in self.greeting_patterns):
            return random.choice(self.greeting_responses), []

        if any(re.search(pattern, message.lower()) for pattern in self.farewell_patterns):
            return random.choice(self.farewell_responses), []

        if any(re.search(pattern, message.lower()) for pattern in self.gratitude_patterns):
            return random.choice(self.gratitude_responses), []

        if re.search(r"(who are you|what are you|tell me about yourself|introduce yourself)", message.lower()):
            return random.choice(self.self_intro_responses), []

        if re.search(r"(help|what can you do|capabilities|features)", message.lower()):
            return random.choice(self.help_responses), []

        if re.search(r"(summarize|summarization|summary)", message.lower()):
            return random.choice(self.summarize_info_responses), []

        # For all other messages, use the Ollama LLM
        # Format conversation history for context
        history = []
        for msg in self.sessions.get(session_id, [])[-10:]:  # Get last 10 messages for context
            history.append({
                "role": "user" if msg.role == MessageRole.USER else "assistant",
                "content": msg.content
            })

        # Convert context dict to string if present
        context_str = None
        if context:
            context_str = ", ".join([f"{k}: {v}" for k, v in context.items()])

        # Generate response using Ollama
        response_content = await self.ollama_service.generate_chat_response(
            message=message,
            history=history,
            context=context_str
        )

        # Generate follow-up suggestions
        doc = self.nlp(message + " " + response_content)
        suggestions = await self._generate_follow_up_suggestions(message, doc)

        return response_content, suggestions

    async def _generate_follow_up_suggestions(self, message: str, doc) -> List[str]:
        """
        Generate follow-up suggestions based on the message and response

        Args:
            message: The user's message
            doc: The SpaCy doc of the message and response

        Returns:
            List of suggested follow-up messages
        """
        # Extract potential topics from the message
        topics = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 3]

        if not topics:
            return ["Tell me more", "How can I help you?", "What else would you like to know?"]

        # Use the first identified topic to generate suggestions
        main_topic = topics[0]

        # If the topic is in our knowledge base, use predefined suggestions
        if main_topic in self.topic_knowledge:
            return [
                f"Tell me more about {main_topic}",
                f"How does {main_topic} work?",
                f"What are the benefits of {main_topic}?"
            ]

        # Otherwise, generate generic suggestions
        return [
            f"Would you like to know more about {main_topic}?",
            "Can I help you with anything else?",
            "Would you like a summary of our conversation?"
        ]