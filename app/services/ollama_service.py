"""
Ollama service for handling LLM interactions
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set
from loguru import logger
import os
import json
import httpx
import re
import time
import threading
from dotenv import load_dotenv

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama Python client not installed. Using direct API calls instead.")

# Load environment variables
load_dotenv()

# Pre-compile regex patterns for better performance
JSON_BLOCK_PATTERN = re.compile(r'```(?:json)?\s*([\s\S]*?)\s*```')
JSON_ARRAY_PATTERN = re.compile(r'\[\s*\{')
JSON_OBJECT_PATTERN = re.compile(r'\{\s*"')
TRAILING_COMMA_PATTERN = re.compile(r',\s*([\]\}])')
QUESTION_PATTERN = re.compile(r'^[Qq](?:uestion)?\s*\d+|.*\?$')
OPTION_PATTERN = re.compile(r'^[A-D][\s:\.\)]+')
ANSWER_PATTERN = re.compile(r'[A-D]')
NUMBERING_PATTERN = re.compile(r'^[\d\-\*\•\.\)]+\s*')
LETTERING_PATTERN = re.compile(r'^[A-D][\s\.\:\)]+')

env_model = os.getenv('CHATBOT_MODEL')

class OllamaService:
    """
    Service for interacting with Ollama LLM using direct API calls with optimized connection handling
    """
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the Ollama service with the specified model

        Args:
            model_name: Name of the Ollama model to use (optional, defaults to env variable or fallback)
        """
        # Use model from environment variable or parameter, with fallback
        env_model = os.getenv('CHATBOT_MODEL')
        if env_model and env_model.lower() != 'default':
            self.model_name = env_model
        elif model_name:
            self.model_name = model_name
        else:
            self.model_name = "llama3:8b"  # Default fallback

        logger.info(f"Initializing OllamaService with model: {self.model_name}")

        # Connection settings
        self.base_url = 'http://localhost:11434'
        self.api_verified = False
        self.last_verification_time = 0
        self.verification_interval = 300  # 5 minutes between checks

        # Client and connection management
        self.client = None
        # Use thread lock instead of asyncio.Lock to avoid event loop issues
        self.init_lock = threading.Lock()
        self.model_cache: Set[str] = set()

        # Initialize async lock only when needed
        self._async_lock = None
        # Don't start async initialization in constructor
        self.initialization_task = None
        self._initialized = False

    async def _get_async_lock(self):
        """Get or create the async lock"""
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock

    async def initialize(self):
        """Initialize the service asynchronously - call this from an async context"""
        if self._initialized:
            return

        # Create initialization task only in async context
        if self.initialization_task is None:
            self.initialization_task = asyncio.create_task(self._async_init())
            await self.initialization_task
            self._initialized = True

    async def _async_init(self):
        """Initialize the service asynchronously"""
        # Get async lock in async context
        async_lock = await self._get_async_lock()

        async with async_lock:
            try:
                await self._verify_ollama_connection()
                # Add current model to cache if it's not None
                if self.model_name:
                    self.model_cache.add(self.model_name)
                # Preload the model
                await self.preload_model()
            except Exception as e:
                logger.error(f"Error during async initialization: {str(e)}")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or initialize the HTTP client"""
        # Always initialize before using
        await self.initialize()

        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=90.0,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=20)
            )
        return self.client

    async def preload_model(self, model_name: Optional[str] = None):
        """
        Send a trivial request to ensure model is loaded

        Args:
            model_name: Optional model name to preload (defaults to current model)
        """
        # Initialize if needed
        await self.initialize()

        target_model = model_name or self.model_name
        if not target_model:
            logger.warning("No model specified for preloading")
            return False

        logger.info(f"Preloading model {target_model}...")

        client = await self._get_client()
        try:
            payload = {
                "model": target_model,
                "prompt": "Hello",
                "stream": False,
                "options": {
                    "num_predict": 1  # Minimize token generation
                }
            }

            response = await client.post(f"{self.base_url}/api/generate", json=payload)

            if response.status_code == 200:
                logger.info(f"Model {target_model} preloaded successfully")
                # Add to model cache
                self.model_cache.add(target_model)
                return True
            else:
                logger.warning(f"Model preloading returned status: {response.status_code}")
                return False

        except Exception as e:
            logger.warning(f"Failed to preload model: {str(e)}")
            return False

    async def switch_model(self, new_model: str) -> str:
        """
        Switch to a different model with optimized handling

        Args:
            new_model: The name of the model to switch to

        Returns:
            The previous model name
        """
        if new_model == self.model_name:
            return self.model_name

        old_model = self.model_name
        self.model_name = new_model

        # If we haven't used this model before, preload it
        if new_model not in self.model_cache:
            await self.preload_model()

        return old_model

    async def _smart_verify_connection(self, force: bool = False) -> bool:
        """
        Smart connection verification that limits how often checks happen

        Args:
            force: Force verification regardless of time elapsed

        Returns:
            True if connection is verified
        """
        current_time = time.time()

        # Skip verification if recently done, unless forced
        if not force and self.api_verified and (current_time - self.last_verification_time) < self.verification_interval:
            return True

        # Get async lock for thread-safe verification
        async_lock = await self._get_async_lock()
        async with async_lock:
            await self._verify_ollama_connection()
            self.last_verification_time = time.time()
            return self.api_verified

    async def _verify_ollama_connection(self):
        """Verify connection to Ollama and determine API structure"""
        try:
            client = await self._get_client()

            # Try direct HTTP request to check if Ollama is running
            response = await client.get(f"{self.base_url}")
            if response.status_code == 200:
                logger.info("Ollama server is running.")

                # Try to get list of models using /api/tags
                try:
                    models_response = await client.get(f"{self.base_url}/api/tags")
                    if models_response.status_code == 200:
                        data = models_response.json()
                        available_models = [model.get('name') for model in data.get('models', [])]
                        logger.info(f"Available Ollama models: {available_models}")

                        # Add available models to cache for faster switching
                        self.model_cache.update(available_models)

                    # Try also /api/ps for running models
                    ps_response = await client.get(f"{self.base_url}/api/ps")
                    if ps_response.status_code == 200:
                        data = ps_response.json()
                        running_models = [model.get('name') for model in data.get('models', [])]
                        logger.info(f"Currently running models: {running_models}")

                except Exception as e:
                    logger.warning(f"Could not get list of models: {str(e)}")

                self.api_verified = True
            else:
                logger.warning(f"Ollama server returned unexpected status: {response.status_code}")
                self.api_verified = False
        except Exception as e:
            logger.warning(f"Could not connect to Ollama server: {str(e)}")
            self.api_verified = False

    async def generate_chat_response(self,
                                    message: str,
                                    history: Optional[List[Dict[str, str]]] = None,
                                    context: Optional[str] = None,
                                    temperature: float = 0.7,
                                    max_tokens: int = 500) -> str:
        """
        Generate a chat response using the Ollama model

        Args:
            message: The user's message
            history: List of previous messages in the conversation
            context: Additional context for the conversation
            temperature: Controls randomness (0.0-1.0)
            max_tokens: Maximum number of tokens to generate

        Returns:
            Generated response text
        """
        try:
            # Verify connection only if needed
            await self._smart_verify_connection()

            # Format the conversation history for the model
            formatted_messages = []

            # Add context as a system message if provided
            if context:
                formatted_messages.append({
                    "role": "system",
                    "content": f"Context information: {context}"
                })

            # Add conversation history
            if history:
                for msg in history:
                    formatted_messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })

            # Add the current message
            formatted_messages.append({
                "role": "user",
                "content": message
            })

            logger.debug(f"Sending chat request with model: {self.model_name}")

            # Get client
            client = await self._get_client()

            # Direct API call using the documented format
            payload = {
                "model": self.model_name,
                "messages": formatted_messages,
                "stream": False,  # Don't stream the response
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }

            logger.debug(f"Sending request to {self.base_url}/api/chat")
            response = await client.post(f"{self.base_url}/api/chat", json=payload)

            if response.status_code == 200:
                response_data = response.json()
                # Extract the content from the response format
                if "message" in response_data and "content" in response_data["message"]:
                    return response_data["message"]["content"]
                else:
                    error_msg = f"Unexpected response format: {response_data}"
                    logger.error(error_msg)
                    return f"Error: Unexpected response format from Ollama API. {error_msg}"
            else:
                error_msg = f"Ollama API returned error: {response.status_code}, {response.text}"
                logger.error(error_msg)

                # Reset verification flag if we get a significant error
                if response.status_code >= 500:
                    self.api_verified = False

                # Check if it's a model not found error
                if "model not found" in response.text.lower():
                    return f"Error: The model '{self.model_name}' was not found. Available models are shown in the logs. Try using '{env_model}' instead or run 'ollama pull {self.model_name}' to download it."

                return f"Error calling Ollama API: {response.status_code}. Please check that Ollama is running with the correct model available."

        except Exception as e:
            logger.error(f"Error generating chat response: {str(e)}")
            # Reset verification flag on connection errors
            self.api_verified = False
            return f"Error: Cannot connect to Ollama server. Please ensure it's running at {self.base_url}."

    async def generate_summary(self,
                              content: str,
                              style: str = "narrative",
                              length: str = "medium",
                              focus_topics: Optional[List[str]] = None) -> str:
        """
        Generate a summary of the provided content

        Args:
            content: The text content to summarize
            style: Summary style (narrative, bullet_points, academic, simplified)
            length: Summary length (short, medium, long)
            focus_topics: Optional list of topics to focus on

        Returns:
            Generated summary
        """
        try:
            # Verify connection if needed
            await self._smart_verify_connection()

            # Create a prompt for the summarization task
            prompt = f"""Please summarize the following content.

Style: {style}
Length: {length}
"""

            if focus_topics:
                # Clean focus topics to prevent formatting issues
                clean_topics = [topic.replace('\n', ' ').strip() for topic in focus_topics]
                topics_list = ", ".join([f'"{topic}"' for topic in clean_topics])
                prompt += f"Focus on these topics: {topics_list}\n"

            prompt += f"\nContent to summarize:\n{content}\n\nSummary:"

            logger.debug(f"Sending generate request with model: {self.model_name}")

            # Get client
            client = await self._get_client()

            # Direct API call
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 1000
                }
            }

            logger.debug("Sending request to /api/generate")
            response = await client.post(f"{self.base_url}/api/generate", json=payload)

            if response.status_code == 200:
                response_data = response.json()
                return response_data.get("response", "")
            else:
                error_msg = f"Ollama API returned error: {response.status_code}, {response.text}"
                logger.error(error_msg)

                # Reset verification flag if we get a significant error
                if response.status_code >= 500:
                    self.api_verified = False

                return f"Error calling Ollama API: {response.status_code}. Please check that Ollama is running with the correct model available."

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            # Reset verification flag on connection errors
            self.api_verified = False
            return f"Error: {str(e)}. Cannot connect to Ollama server."

    async def extract_key_points(self, text: str, num_points: int = 5) -> List[str]:
        """
        Extract key points from a text using the LLM

        Args:
            text: The text to extract key points from
            num_points: Number of key points to extract

        Returns:
            List of key points
        """
        try:
            prompt = f"""Extract exactly {num_points} key points from the following text.
Format each point as a single concise sentence that captures an important idea.
Do not use bullet points or numbering in your response.
Separate each point with a newline character.

Text:
{text}

Key Points:"""

            # Call generate and process the response
            response_text = await self.generate_summary(content=prompt, style="bullet_points", length="short")

            # Process the response to get individual points
            points = [point.strip() for point in response_text.split('\n')
                     if point.strip() and not point.strip().startswith(('•', '-', '*', '1.', '2.'))]

            # Limit to requested number of points
            return points[:num_points] if points else ["Unable to extract key points"]

        except Exception as e:
            logger.error(f"Error extracting key points: {str(e)}")
            return ["Unable to extract key points. Please check that Ollama is running and the model is available."]

    async def analyze_sentiment(self, text: str) -> str:
        """
        Analyze the sentiment of a text

        Args:
            text: The text to analyze

        Returns:
            Sentiment analysis result
        """
        try:
            prompt = f"""Analyze the sentiment of the following text.
Classify it as positive, negative, or neutral, and explain why in one sentence.

Text:
{text}

Sentiment:"""

            # Reuse the generate_summary method with our sentiment analysis prompt
            return await self.generate_summary(content=prompt, style="narrative", length="short")

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return f"Error analyzing sentiment: {str(e)}"

    async def create_quiz(self,
                         content: str,
                         num_questions: int = 5,
                         focus_topics: Optional[List[str]] = None,
                         model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate a quiz from the provided content with improved accuracy

        Args:
            content: The text content to create a quiz from
            num_questions: Number of questions to generate
            focus_topics: Optional list of topics to focus on
            model_name: Optional model name to use instead of the default

        Returns:
            List of quiz questions with multiple choice options
        """
        # Verify connection
        await self._smart_verify_connection()

        # Use a specific model if provided
        original_model = self.model_name
        if model_name:
            await self.switch_model(model_name)
            logger.info(f"Using model '{self.model_name}' for quiz generation")

        try:
            # Generate quiz using a single optimized approach
            questions = await self._generate_quiz(content, num_questions, focus_topics)
            if questions and len(questions) > 0:
                logger.info(f"Successfully generated {len(questions)} quiz questions")
                return questions

            # Return empty list if generation failed
            logger.warning("Quiz generation failed to produce valid questions")
            return []

        except Exception as e:
            logger.error(f"Error generating quiz: {str(e)}")
            return []
        finally:
            # Restore original model if changed
            if model_name and original_model != self.model_name:
                await self.switch_model(original_model)

    async def _generate_quiz(self,
                            content: str,
                            num_questions: int = 5,
                            focus_topics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Generate quiz questions using a single optimized approach with a comprehensive prompt

        Args:
            content: The text content to create a quiz from
            num_questions: Number of questions to generate
            focus_topics: Optional list of topics to focus on

        Returns:
            List of quiz questions with multiple choice options
        """
        # Create a comprehensive prompt with explicit formatting instructions
        #region Prompt Creation
        prompt = f"""Generate a quiz with {num_questions} multiple-choice questions based on the provided content.

REQUIREMENTS:
1. Each question must have EXACTLY 4 options (A, B, C, D)
2. Only ONE option should be correct
3. Include a brief explanation (1-2 sentences) for the correct answer
4. Questions should be diverse and test understanding rather than just recall
5. Make sure the correct answer appears in different positions across questions (randomize the position)
6. Avoid redundant or duplicate questions
7. Ensure all questions are factually accurate based on the content
8. Use clear and concise language for both questions and options
9. Avoid overly complex or technical language unless necessary for the topic
10. Ensure the questions are relevant to the content provided
11. Ensure no redundant or duplicate options are generated

RESPONSE FORMAT:
Your response must be valid JSON with this exact structure:
```json
[
  {{
    "question": "Clear question text here?",
    "options": ["First option", "Second option", "Third option", "Fourth option"],
    "correct_answer": 0,
    "explanation": "Brief explanation of why the correct answer is right"
  }},
  ...more questions...
]
```

IMPORTANT NOTES:
- The "correct_answer" must be the index (0, 1, 2, or 3) of the correct option in the options array
- All options should be plausible and similar in style/length
- Questions should be substantive and focus on key concepts
- The explanation should be concise but informative"""
        #endregion

        if focus_topics and len(focus_topics) > 0:
            topics_str = ", ".join([f'"{topic}"' for topic in focus_topics])
            prompt += f"\n\nFOCUS TOPICS:\nFocus your questions specifically on these topics: {topics_str}"

        # Add content at the end to keep it fresh in the model's context
        prompt += f"\n\nCONTENT:\n{content}\n"

        # Get client
        client = await self._get_client()

        try:
            # Direct API call
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.5,  # Balanced temperature for creativity and accuracy
                    "num_predict": 4000  # Ensure enough tokens for multiple questions
                }
            }

            logger.debug("Sending quiz generation request")
            response = await client.post(f"{self.base_url}/api/generate", json=payload)

            if response.status_code == 200:
                response_data = response.json()
                response_text = response_data.get("response", "")

                # Process the response to extract JSON
                questions = self._process_quiz_response(response_text)

                # Validate each question meets our requirements
                validated_questions = []
                for q in questions:
                    if (isinstance(q, dict) and
                        'question' in q and
                        'options' in q and
                        'correct_answer' in q and
                        isinstance(q['options'], list) and
                        len(q['options']) == 4 and
                        isinstance(q['correct_answer'], int) and
                        0 <= q['correct_answer'] < 4):
                        validated_questions.append(q)

                return validated_questions
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return []

        except Exception as e:
            logger.error(f"Quiz generation error: {str(e)}")
            return []

    def _process_quiz_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Process raw response text to extract and validate quiz questions

        Args:
            response_text: Raw response from LLM

        Returns:
            List of parsed and validated questions
        """
        # Try to extract JSON from the response
        try:
            # Method 1: Look for JSON block in markdown format
            json_match = JSON_BLOCK_PATTERN.search(response_text)
            if json_match:
                try:
                    questions_json = json_match.group(1)
                    # Fix common JSON issues like trailing commas
                    questions_json = TRAILING_COMMA_PATTERN.sub(r'\1', questions_json)
                    questions = json.loads(questions_json)
                    if isinstance(questions, list):
                        return questions
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON block, trying alternative methods")

            # Method 2: Try to find and parse the entire response as JSON
            try:
                # Check if the response resembles a JSON array
                stripped_text = response_text.strip()
                if (stripped_text.startswith('[') and stripped_text.endswith(']')) or (stripped_text.startswith('{') and stripped_text.endswith('}')):
                    # Fix common JSON issues
                    fixed_json = TRAILING_COMMA_PATTERN.sub(r'\1', stripped_text)
                    questions = json.loads(fixed_json)
                    if isinstance(questions, dict):
                        questions = [questions]  # Single question case
                    if isinstance(questions, list):
                        return questions
            except json.JSONDecodeError:
                logger.warning("Failed to parse entire response as JSON")

            # If all methods fail, return empty list
            logger.warning("JSON parsing failed")
            return []

        except Exception as e:
            logger.error(f"Error processing quiz response: {str(e)}")
            return []

    # Close client on shutdown
    async def close(self):
        """Close the HTTP client to release resources"""
        if self.client:
            await self.client.aclose()
            self.client = None
            logger.info("Closed OllamaService HTTP client")