"""
Ollama service for handling LLM interactions
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import os
import json
import httpx

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama Python client not installed. Using direct API calls instead.")

class OllamaService:
    """
    Service for interacting with Ollama LLM using direct API calls
    """
    def __init__(self, model_name: str = "llama3:latest"):
        """
        Initialize the Ollama service with the specified model

        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        self.base_url = 'http://localhost:11434'
        self.api_verified = False

        # Try to verify connection and API endpoints
        self._verify_ollama_connection()

    def _verify_ollama_connection(self):
        """Verify connection to Ollama and determine API structure"""
        try:
            # Try direct HTTP request to check if Ollama is running
            response = httpx.get(f"{self.base_url}")
            if response.status_code == 200:
                logger.info("Ollama server is running.")

                # Try to get list of models using /api/tags
                try:
                    models_response = httpx.get(f"{self.base_url}/api/tags")
                    if models_response.status_code == 200:
                        data = models_response.json()
                        available_models = [model.get('name') for model in data.get('models', [])]
                        logger.info(f"Available Ollama models from /api/tags: {available_models}")

                    # Try also /api/ps for running models
                    ps_response = httpx.get(f"{self.base_url}/api/ps")
                    if ps_response.status_code == 200:
                        data = ps_response.json()
                        running_models = [model.get('name') for model in data.get('models', [])]
                        logger.info(f"Currently running models from /api/ps: {running_models}")

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

            logger.debug(f"Sending chat request to Ollama with model: {self.model_name}")

            # Check if we need to reconnect
            if not self.api_verified:
                self._verify_ollama_connection()

            # Direct API call using the documented format
            async with httpx.AsyncClient(timeout=60.0) as client:  # Increase timeout for longer generations
                payload = {
                    "model": self.model_name,
                    "messages": formatted_messages,
                    "stream": False,  # Don't stream the response
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }

                logger.debug(f"Sending request to {self.base_url}/api/chat with payload: {json.dumps(payload)}")
                response = await client.post(f"{self.base_url}/api/chat", json=payload)

                if response.status_code == 200:
                    response_data = response.json()
                    logger.debug(f"Received response: {json.dumps(response_data)}")
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

                    # Check if it's a model not found error
                    if "model not found" in response.text.lower():
                        return f"Error: The model '{self.model_name}' was not found. Available models are shown in the logs. Try using 'llama3:latest' instead or run 'ollama pull {self.model_name}' to download it."

                    return f"Error calling Ollama API: {response.status_code}. Please check that Ollama is running with the correct model available."

        except Exception as e:
            logger.error(f"Error generating chat response: {str(e)}")
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
            # Create a prompt for the summarization task
            prompt = f"""Please summarize the following content.

Style: {style}
Length: {length}
"""

            if focus_topics:
                prompt += f"Focus on these topics: {', '.join(focus_topics)}\n"

            prompt += f"\nContent to summarize:\n{content}\n\nSummary:"

            logger.debug(f"Sending generate request to Ollama with model: {self.model_name}")

            # Check if we need to reconnect
            if not self.api_verified:
                self._verify_ollama_connection()

            # Direct API call
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 1000
                    }
                }

                logger.debug(f"Sending request to {self.base_url}/api/generate with payload: {json.dumps(payload)}")
                response = await client.post(f"{self.base_url}/api/generate", json=payload)

                if response.status_code == 200:
                    response_data = response.json()
                    logger.debug(f"Received generate response: {json.dumps(response_data)}")
                    return response_data.get("response", "")
                else:
                    error_msg = f"Ollama API returned error: {response.status_code}, {response.text}"
                    logger.error(error_msg)
                    return f"Error calling Ollama API: {response.status_code}. Please check that Ollama is running with the correct model available."

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
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