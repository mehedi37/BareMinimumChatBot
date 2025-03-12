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
        # Try to extract key topics if none provided
        if not focus_topics:
            try:
                # Try to import spaCy
                import spacy

                # Try to load spaCy model
                try:
                    nlp = spacy.load("en_core_web_sm")

                    # Try to import KeyBERT
                    try:
                        from keybert import KeyBERT

                        # Initialize KeyBERT
                        kw_model = KeyBERT()

                        # Extract key entities and keywords
                        doc = nlp(content[:5000])  # Process first 5000 chars
                        entities = [ent.text.replace('\n', ' ').strip() for ent in doc.ents if ent.label_ in ("ORG", "PERSON", "GPE", "PRODUCT", "EVENT")]

                        # Extract keywords using KeyBERT
                        keywords = kw_model.extract_keywords(content[:5000], keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
                        # Extract just the keyword strings from the results, ignoring scores
                        extracted_keywords = []
                        for keyword_item in keywords:
                            if isinstance(keyword_item, tuple):
                                extracted_keywords.append(keyword_item[0].replace('\n', ' ').strip())
                            elif isinstance(keyword_item, list) and keyword_item and isinstance(keyword_item[0], tuple):
                                extracted_keywords.append(keyword_item[0][0].replace('\n', ' ').strip())
                            elif isinstance(keyword_item, str):
                                extracted_keywords.append(keyword_item.replace('\n', ' ').strip())

                        # Combine entities and keywords
                        focus_topics = list(set(entities + extracted_keywords))[:5]
                        logger.info(f"Automatically extracted focus topics: {focus_topics}")
                    except ImportError:
                        # KeyBERT not available, fall back to just spaCy
                        logger.warning("Could not import KeyBERT for topic extraction. Using spaCy only.")
                        doc = nlp(content[:5000])
                        entities = [ent.text.replace('\n', ' ').strip() for ent in doc.ents if ent.label_ in ("ORG", "PERSON", "GPE", "PRODUCT", "EVENT")]
                        focus_topics = entities[:5] if entities else None
                        if focus_topics:
                            logger.info(f"Extracted entity topics: {focus_topics}")
                except Exception as e:
                    logger.warning(f"Failed to load spaCy model: {e}")
            except ImportError:
                logger.warning("spaCy not available for topic extraction")
            except Exception as e:
                logger.warning(f"Could not extract topics: {e}")

        # Verify connection
        await self._smart_verify_connection()

        # Use a specific model if provided
        original_model = self.model_name
        if model_name:
            await self.switch_model(model_name)
            logger.info(f"Using model '{self.model_name}' for quiz generation")

        try:
            # Try the template-based approach first
            logger.info("Using template-based quiz generation approach")
            questions = await self._create_quiz_template(content, num_questions, focus_topics)
            if questions and len(questions) > 0:
                logger.info(f"Successfully generated {len(questions)} questions using template approach")
                return questions

            # Try the concept-based approach if template approach fails
            logger.info("Template approach failed, trying concept-based quiz generation")
            questions = await self._create_quiz_concept_based(content, num_questions, focus_topics)
            if questions and len(questions) > 0:
                logger.info(f"Successfully generated {len(questions)} questions using concept-based approach")
                return questions

            # Fall back to standard method if both approaches fail
            logger.info("Concept-based approach failed, falling back to standard quiz generation method")
            return await self._create_quiz_standard(content, num_questions, focus_topics)

        except Exception as e:
            logger.error(f"Error generating quiz: {str(e)}")
            # Return empty list on failure
            return []
        finally:
            # Restore original model if changed
            if model_name and original_model != self.model_name:
                await self.switch_model(original_model)

    async def _batch_process_prompts(self, prompts: List[str],
                                    temperature: float = 0.3,
                                    max_tokens: int = 1000) -> List[Optional[str]]:
        """
        Process multiple prompts in parallel for better performance

        Args:
            prompts: List of prompts to process
            temperature: Temperature setting for generation
            max_tokens: Maximum tokens to generate per request

        Returns:
            List of responses (None for failed requests)
        """
        if not prompts:
            return []

        client = await self._get_client()
        results: List[Optional[str]] = []

        # Create all tasks
        tasks = []
        for prompt in prompts:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            # Create task but don't await it yet
            task = asyncio.create_task(self._safe_execute_prompt(client, payload))
            tasks.append(task)

        # Wait for all tasks to complete
        for task in tasks:
            try:
                result = await task
                results.append(result)
            except Exception as e:
                logger.error(f"Task failed: {str(e)}")
                results.append(None)

        return results

    async def _safe_execute_prompt(self, client: httpx.AsyncClient, payload: Dict[str, Any]) -> Optional[str]:
        """
        Safely execute a prompt with error handling

        Args:
            client: The HTTP client to use
            payload: The request payload

        Returns:
            Response text or None if failed
        """
        try:
            response = await client.post(f"{self.base_url}/api/generate", json=payload, timeout=90.0)

            if response.status_code == 200:
                response_data = response.json()
                return response_data.get("response", "")
            else:
                logger.error(f"API error: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Request error: {str(e)}")
            return None

    async def _create_quiz_template(self,
                                   content: str,
                                   num_questions: int = 5,
                                   focus_topics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Generate quiz questions using a template-based approach with chain-of-thought reasoning

        Args:
            content: The text content to create a quiz from
            num_questions: Number of questions to generate
            focus_topics: Optional list of topics to focus on

        Returns:
            List of quiz questions with multiple choice options
        """
        # Create a system prompt that guides the model through the process
        prompt = f"""You are a professional quiz creator. Create {num_questions} multiple-choice questions based on the content provided.

For each question, follow this exact structured process:
1. Identify an important concept or fact from the content
2. Formulate a clear, concise question about it
3. Create the correct answer and three incorrect but plausible options
4. Provide a brief explanation of why the correct answer is right

Your response must be valid JSON in this exact format:
```json
[
  {{
    "question": "Question text here?",
    "options": ["Correct option", "Wrong option 1", "Wrong option 2", "Wrong option 3"],
    "correct_answer": 0,
    "explanation": "Why the correct answer is right"
  }},
  // more questions...
]
```

The correct_answer MUST be the index (0-3) of the correct option in the options array.

Make sure:
- Questions are substantive and test understanding of core concepts
- Options are all plausible and similar in length and style
- The correct answer is placed at random positions (not always first)
- The explanation is factual and informative"""

        if focus_topics:
            topics_str = ", ".join([f'"{topic}"' for topic in focus_topics])
            prompt += f"\n\nFocus on these topics: {topics_str}"

        prompt += f"\n\nContent:\n{content}"

        # Get client
        client = await self._get_client()

        try:
            # Direct API call
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.5,  # Lower temperature for more factual responses
                    "num_predict": 4000  # Ensure enough tokens for multiple questions
                }
            }

            logger.debug("Sending quiz generation request using template approach")
            response = await client.post(f"{self.base_url}/api/generate", json=payload)

            if response.status_code == 200:
                response_data = response.json()
                response_text = response_data.get("response", "")

                # Process the response to extract JSON
                return self._process_quiz_response(response_text)
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return []

        except Exception as e:
            logger.error(f"Template quiz generation error: {str(e)}")
            return []

    async def _create_quiz_concept_based(self,
                                       content: str,
                                       num_questions: int = 5,
                                       focus_topics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Generate quiz questions using a concept-based approach that first extracts key concepts
        and then generates questions about them

        Args:
            content: The text content to create a quiz from
            num_questions: Number of questions to generate
            focus_topics: Optional list of topics to focus on

        Returns:
            List of quiz questions with multiple choice options
        """
        # First, extract key concepts from the content if focus topics not provided
        extracted_concepts = focus_topics or []

        if not extracted_concepts:
            try:
                # First prompt to extract key concepts
                concept_prompt = f"""Extract {min(num_questions + 3, 10)} key concepts or facts from the following content.
Respond with a JSON array of strings, each representing an important concept.

Content:
{content[:5000]}

Format your response as:
```json
["Concept 1", "Concept 2", "Concept 3", ...]
```"""

                # Get client
                client = await self._get_client()

                # Direct API call for concept extraction
                payload = {
                    "model": self.model_name,
                    "prompt": concept_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,  # Lower temperature for more factual responses
                        "num_predict": 1000  # Enough tokens for concepts
                    }
                }

                logger.debug("Extracting key concepts for quiz generation")
                response = await client.post(f"{self.base_url}/api/generate", json=payload)

                if response.status_code == 200:
                    response_data = response.json()
                    concept_text = response_data.get("response", "")

                    # Extract JSON array from the response
                    json_match = JSON_BLOCK_PATTERN.search(concept_text)
                    if json_match:
                        try:
                            concepts_json = json_match.group(1)
                            extracted_concepts = json.loads(concepts_json)
                            logger.info(f"Extracted {len(extracted_concepts)} concepts for quiz generation")
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse extracted concepts JSON")

                    if not extracted_concepts:
                        # Try simple line splitting as fallback
                        lines = [line.strip() for line in concept_text.split('\n') if line.strip()]
                        extracted_concepts = [line for line in lines if not line.startswith('```')][:10]
            except Exception as e:
                logger.warning(f"Error extracting concepts: {str(e)}")

        # Now generate questions for each concept
        if not extracted_concepts:
            logger.warning("No concepts extracted, falling back")
            return []

        all_questions = []
        concepts_to_use = extracted_concepts[:num_questions]

        # Create a batch of prompts for each concept
        question_prompts = []
        for concept in concepts_to_use:
            prompt = f"""Create one multiple-choice question about the concept: "{concept}" based on this content:

{content[:1000]}

Your response must be valid JSON in this format:
```json
{{
  "question": "Question text here?",
  "options": ["Correct option", "Wrong option 1", "Wrong option 2", "Wrong option 3"],
  "correct_answer": 0,
  "explanation": "Why the correct answer is right"
}}
```

The correct_answer MUST be the index (0-3) of the correct option in the options array.
Make sure all options are plausible and the correct answer is factually accurate."""

            question_prompts.append(prompt)

        # Batch process all concept prompts
        results = await self._batch_process_prompts(question_prompts, temperature=0.5)

        # Process each result
        for result_text in results:
            if not result_text:
                continue

            # Process the response to extract JSON
            processed_questions = self._process_quiz_response(result_text)
            if processed_questions:
                all_questions.extend(processed_questions)

        # Return the questions, limited to the requested number
        return all_questions[:num_questions]

    async def _create_quiz_standard(self,
                                   content: str,
                                   num_questions: int = 5,
                                   focus_topics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Standard quiz generation approach - most reliable general method
        that works with various LLMs

        Args:
            content: The text content to create a quiz from
            num_questions: Number of questions to generate
            focus_topics: Optional list of topics to focus on

        Returns:
            List of quiz questions with multiple choice options
        """
        # Create a simple, direct prompt that works with most LLMs
        prompt = f"""Generate {num_questions} multiple-choice quiz questions based on the following content.

Each question should have:
1. A clear question
2. Four possible answer options (A, B, C, D)
3. Identification of the correct answer
4. A brief explanation

Format your response as a JSON array:
```json
[
  {{
    "question": "What is the main topic discussed in the content?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": 0,
    "explanation": "Option A is correct because..."
  }},
  // more questions...
]
```

Make the questions diverse and ensure they test understanding rather than just recall.
The correct_answer MUST be the index (0-3) of the correct option in the options array."""

        if focus_topics:
            topics_str = ", ".join([f'"{topic}"' for topic in focus_topics])
            prompt += f"\n\nFocus on these topics: {topics_str}"

        prompt += f"\n\nContent:\n{content}"

        # Get client
        client = await self._get_client()

        try:
            # Direct API call
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,  # Slightly higher temperature for diversity
                    "num_predict": 4000  # Ensure enough tokens for multiple questions
                }
            }

            logger.debug("Sending quiz generation request using standard approach")
            response = await client.post(f"{self.base_url}/api/generate", json=payload)

            if response.status_code == 200:
                response_data = response.json()
                response_text = response_data.get("response", "")

                # Process the response to extract JSON
                return self._process_quiz_response(response_text)
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return self._create_fallback_questions(content, num_questions)

        except Exception as e:
            logger.error(f"Standard quiz generation error: {str(e)}")
            return self._create_fallback_questions(content, num_questions)

    def _process_quiz_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Process raw response text to extract and validate quiz questions

        Args:
            response_text: Raw response from LLM

        Returns:
            List of parsed and validated questions
        """
        # Try several methods to parse the JSON
        try:
            # Method 1: Look for JSON block in markdown format
            json_match = JSON_BLOCK_PATTERN.search(response_text)
            if json_match:
                try:
                    questions_json = json_match.group(1)
                    # Fix common JSON issues like trailing commas
                    questions_json = TRAILING_COMMA_PATTERN.sub(r'\1', questions_json)
                    questions = json.loads(questions_json)
                    return self._validate_and_format_questions(questions)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON block, trying alternative methods")

            # Method 2: Try to find and parse the entire response as JSON
            try:
                # Check if the response starts with [ and ends with ]
                stripped_text = response_text.strip()
                if stripped_text.startswith('[') and stripped_text.endswith(']'):
                    # Fix common JSON issues
                    fixed_json = TRAILING_COMMA_PATTERN.sub(r'\1', stripped_text)
                    questions = json.loads(fixed_json)
                    return self._validate_and_format_questions(questions)
            except json.JSONDecodeError:
                logger.warning("Failed to parse entire response as JSON")

            # Method 3: Try to find JSON object in the text
            if '{' in response_text and '}' in response_text:
                try:
                    # Extract text between the first { and the last }
                    start_idx = response_text.find('{')
                    end_idx = response_text.rfind('}') + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        potential_json = response_text[start_idx:end_idx]
                        # Check if it's a single object or an array of objects
                        if potential_json.startswith('{'):
                            potential_json = f"[{potential_json}]"  # Wrap in array
                        # Fix common JSON issues
                        fixed_json = TRAILING_COMMA_PATTERN.sub(r'\1', potential_json)
                        questions = json.loads(fixed_json)
                        return self._validate_and_format_questions(questions)
                except json.JSONDecodeError:
                    logger.warning("Failed to extract JSON object from text")

            # Method 4: Attempt to parse a manually constructed JSON from the text
            questions = self._parse_questions_from_text(response_text)
            if questions:
                return questions

            # If all methods fail, return empty list
            logger.warning("All JSON parsing methods failed")
            return []

        except Exception as e:
            logger.error(f"Error processing quiz response: {str(e)}")
            return []

    def _validate_and_format_questions(self, questions: Any) -> List[Dict[str, Any]]:
        """
        Validate and format parsed questions to ensure they match the expected structure

        Args:
            questions: Parsed JSON data

        Returns:
            Validated and formatted questions
        """
        validated_questions = []

        # Ensure questions is a list
        if not isinstance(questions, list):
            if isinstance(questions, dict):
                # Convert single question dict to list
                questions = [questions]
            else:
                return []

        for q in questions:
            if not isinstance(q, dict):
                continue

            # Check required fields
            if 'question' not in q or 'options' not in q:
                continue

            # Ensure options is a list with at least 2 items
            if not isinstance(q['options'], list) or len(q['options']) < 2:
                continue

            # Format question
            formatted_q = {
                'question': q.get('question', '').strip(),
                'options': [str(opt).strip() for opt in q.get('options', [])],
                'correct_answer': 0,  # Default to first option if not specified
                'explanation': q.get('explanation', 'No explanation provided').strip()
            }

            # Handle correct_answer - ensure it's a valid index
            if 'correct_answer' in q:
                try:
                    correct_idx = int(q['correct_answer'])
                    if 0 <= correct_idx < len(formatted_q['options']):
                        formatted_q['correct_answer'] = correct_idx
                except (ValueError, TypeError):
                    # If correct_answer is a string like "A", "B", etc.
                    if isinstance(q['correct_answer'], str):
                        answer_str = q['correct_answer'].upper().strip()
                        if answer_str in "ABCD":
                            # Convert A->0, B->1, etc.
                            correct_idx = ord(answer_str) - ord('A')
                            if 0 <= correct_idx < len(formatted_q['options']):
                                formatted_q['correct_answer'] = correct_idx

            # Only add questions with valid content
            if formatted_q['question'] and len(formatted_q['options']) >= 2:
                validated_questions.append(formatted_q)

        return validated_questions

    def _parse_questions_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse questions from plain text when JSON parsing fails

        Args:
            text: Raw text response

        Returns:
            Extracted questions in the proper format
        """
        questions = []

        # Split by numbered patterns like "1.", "Question 1:", etc.
        question_blocks = re.split(r'(?:\n|^)\s*(?:Q(?:uestion)?\s*\d+[.:]?|[\d]+\s*[).\:])', text)
        question_blocks = [block.strip() for block in question_blocks if block.strip()]

        for block in question_blocks:
            try:
                # Split into lines
                lines = [line.strip() for line in block.split('\n') if line.strip()]

                if not lines:
                    continue

                # First line is usually the question
                question_text = lines[0].rstrip('?:') + '?'

                # Find options - look for lines starting with A, B, C, D
                options = []
                correct_idx = 0
                correct_letter = None
                explanation = ""

                # Look for lines with option patterns
                option_lines = []
                for i, line in enumerate(lines[1:], 1):
                    # Check for option patterns (A. B. C. D. or A) B) C) D) etc.)
                    if re.match(r'^[A-D][.):]\s', line):
                        # Extract the option text, removing the prefix
                        option_text = re.sub(r'^[A-D][.):]\s*', '', line)
                        options.append(option_text)
                        option_lines.append(i)

                        # Check if this option is marked as correct
                        if '*' in line or '(correct)' in line.lower() or 'correct' in line.lower():
                            correct_letter = line[0].upper()

                # If we found options
                if options:
                    # Look for explanation after the options
                    if option_lines and option_lines[-1] + 1 < len(lines):
                        explanation_lines = lines[option_lines[-1] + 1:]

                        # Join explanation lines, ignoring any that look like "correct answer" indicators
                        explanation = ' '.join([
                            line for line in explanation_lines
                            if not re.match(r'^correct\s+answer\s*[:=]', line.lower())
                        ])

                    # Look for "correct answer" indicator
                    if not correct_letter:
                        for line in lines:
                            match = re.search(r'(?:correct\s+answer\s*[:=]\s*|answer\s*[:=]\s*)([A-D])', line, re.IGNORECASE)
                            if match:
                                correct_letter = match.group(1).upper()
                                break

                    # Convert correct letter to index (A->0, B->1, etc.)
                    if correct_letter and correct_letter in "ABCD":
                        correct_idx = ord(correct_letter) - ord('A')
                        if correct_idx >= len(options):
                            correct_idx = 0

                    # Create the question dictionary
                    if len(options) >= 2:
                        questions.append({
                            'question': question_text,
                            'options': options,
                            'correct_answer': correct_idx,
                            'explanation': explanation or "No explanation provided"
                        })
            except Exception as e:
                logger.warning(f"Error parsing question block: {str(e)}")

        return questions

    def _create_fallback_questions(self, content: str, num_questions: int = 5) -> List[Dict[str, Any]]:
        """
        Create simple fallback questions when all generation methods fail

        Args:
            content: Content to create questions about
            num_questions: Number of questions to create

        Returns:
            List of basic quiz questions
        """
        # Extract a few sentences to use in the questions
        sentences = re.split(r'(?<=[.!?])\s+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:num_questions*2]

        if not sentences:
            # If no good sentences, create a generic question
            return [{
                'question': "What was the main topic of the content?",
                'options': ["The provided content", "Something else", "None of the above", "Cannot be determined"],
                'correct_answer': 0,
                'explanation': "This is a fallback question because the content couldn't be parsed properly."
            }]

        questions = []
        for i in range(min(num_questions, len(sentences))):
            sentence = sentences[i]
            # Create a simple recall question
            questions.append({
                'question': f"According to the content, which of the following statements is true?",
                'options': [
                    sentence,
                    f"The opposite of: {sentence}",
                    "None of the statements in the content",
                    "Cannot be determined from the content"
                ],
                'correct_answer': 0,
                'explanation': f"This statement appears in the original content."
            })

        return questions

    # Close client on shutdown
    async def close(self):
        """Close the HTTP client to release resources"""
        if self.client:
            await self.client.aclose()
            self.client = None
            logger.info("Closed OllamaService HTTP client")