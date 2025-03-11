"""
Ollama service for handling LLM interactions
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import os
import json
import httpx
import re

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

    async def create_quiz(self,
                         content: str,
                         num_questions: int = 5,
                         focus_topics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Generate a quiz from the provided content

        Args:
            content: The text content to create a quiz from
            num_questions: Number of questions to generate
            focus_topics: Optional list of topics to focus on

        Returns:
            List of quiz questions with multiple choice options
        """
        # Try the two-stage approach first
        try:
            logger.info("Attempting two-stage quiz generation for improved accuracy")
            questions = await self._generate_quiz_two_stage(content, num_questions, focus_topics)
            if questions and len(questions) > 0:
                logger.info(f"Successfully generated {len(questions)} questions using two-stage approach")
                return questions
        except Exception as e:
            logger.warning(f"Two-stage quiz generation failed: {str(e)}. Falling back to standard method.")

        # Fall back to the original method if the two-stage approach fails
        try:
            # Create a prompt for the quiz generation task
            prompt = f"""Create a multiple choice quiz with exactly {num_questions} questions based on the following content.
IMPORTANT: Each question MUST have 4 distinct answer options. Every option must be a complete phrase or sentence.

For each question, provide:
1. The question text asking about a specific fact or concept
2. Four possible answers labeled as options in an array
3. The correct answer index (0 for the first option, 1 for the second, etc.)
4. A brief explanation of why the correct answer is right

For example, a good question would be:
"What is the capital of France?"
With options: ["Paris", "London", "Berlin", "Madrid"]
Correct answer: 0 (for Paris)
Explanation: "Paris is the capital city of France."

IMPORTANT: For each question, you MUST create four distinct options that are plausible but with only one correct answer.

Use this exact JSON format in your response:
```json
[
  {{
    "question": "What is the capital of France?",
    "options": ["Paris", "London", "Berlin", "Madrid"],
    "correct_answer": 0,
    "explanation": "Paris is the capital city of France."
  }},
  // more questions...
]
```

The explanation MUST mention which option is correct and why it's correct.
Do not include any text outside of the JSON structure. Make sure all questions have complete options.
"""

            if focus_topics:
                prompt += f"\nFocus on these topics: {', '.join(focus_topics)}\n"

            prompt += f"\nContent to create quiz from:\n{content}\n\nQuiz:"

            logger.debug(f"Sending quiz generation request to Ollama with model: {self.model_name}")

            # Check if we need to reconnect
            if not self.api_verified:
                self._verify_ollama_connection()

            # Direct API call
            async with httpx.AsyncClient(timeout=90.0) as client:  # Increased timeout for complex generations
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.5,
                        "num_predict": 2500  # Increased token limit for longer responses
                    }
                }

                try:
                    logger.debug(f"Sending request to {self.base_url}/api/generate with payload")
                    response = await client.post(f"{self.base_url}/api/generate", json=payload, timeout=90.0)
                except httpx.ReadTimeout:
                    logger.error("Request to Ollama timed out")
                    return self._create_fallback_questions(num_questions, focus_topics, "Request timed out")
                except Exception as req_e:
                    logger.error(f"Error making request to Ollama: {str(req_e)}")
                    return self._create_fallback_questions(num_questions, focus_topics, str(req_e))

                if response.status_code == 200:
                    try:
                        logger.debug(f"Response from Ollama: '{response.text[:100]}...'")
                        response_data = response.json()
                        response_text = response_data.get("response", "")
                        logger.debug(f"Response from Ollama: '{response_text[:100]}...'")

                        # Extract options from text
                        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
                        logger.debug(f"Extracted {len(lines)} lines for options")

                        # Extract JSON from response
                        try:
                            # Find JSON content within ``` blocks or just parse directly
                            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
                            if match:
                                json_str = match.group(1)
                                logger.debug("Found JSON within code blocks")
                            else:
                                # Try to find a JSON array anywhere in the text
                                array_match = re.search(r'\[\s*\{\s*"question"', response_text)
                                if array_match:
                                    # Extract from the [ to the last ]
                                    start_idx = array_match.start()
                                    # Find the matching closing bracket
                                    bracket_count = 0
                                    end_idx = start_idx
                                    for i in range(start_idx, len(response_text)):
                                        if response_text[i] == '[':
                                            bracket_count += 1
                                        elif response_text[i] == ']':
                                            bracket_count -= 1
                                            if bracket_count == 0:
                                                end_idx = i + 1
                                                break

                                    json_str = response_text[start_idx:end_idx]
                                    logger.debug(f"Extracted JSON array from position {start_idx} to {end_idx}")
                                else:
                                    json_str = response_text
                                    logger.debug("Using full response text for JSON parsing")

                            # Clean up the JSON string
                            json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)  # Remove comments
                            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas
                            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas in objects

                            # Log the cleaned JSON string for debugging
                            logger.debug(f"Cleaned JSON: {json_str[:100]}...")

                            questions = json.loads(json_str)
                            logger.debug(f"Successfully parsed JSON with {len(questions)} questions")

                            # Process and generate options for each question
                            processed_questions = await self._process_and_ensure_options(questions, content)

                            if processed_questions:
                                logger.debug(f"Returning {len(processed_questions)} processed questions")
                                return processed_questions
                            else:
                                logger.warning("No valid questions after processing")
                                return self._create_fallback_questions(num_questions, focus_topics, "No valid questions found")

                        except json.JSONDecodeError as json_err:
                            logger.error(f"JSON decode error: {str(json_err)}")
                            logger.debug(f"Raw response was: {response_text[:200]}...")

                            # Attempt fallback parsing
                            questions = self._parse_questions_fallback(response_text, num_questions)
                            if questions:
                                # Process and generate options for parsed questions
                                processed_questions = await self._process_and_ensure_options(questions, content)
                                if processed_questions:
                                    logger.debug(f"Fallback parsing successful: {len(processed_questions)} questions")
                                    return processed_questions

                            return self._create_fallback_questions(num_questions, focus_topics, "JSON parsing failed")
                    except Exception as parse_e:
                        logger.error(f"Error processing Ollama response: {str(parse_e)}")
                        return self._create_fallback_questions(num_questions, focus_topics, str(parse_e))
                else:
                    error_msg = f"Ollama API returned error: {response.status_code}"
                    logger.error(error_msg)
                    if response.status_code == 404:
                        return self._create_fallback_questions(num_questions, focus_topics, "Model not found")
                    elif response.status_code == 500:
                        return self._create_fallback_questions(num_questions, focus_topics, "Ollama server error")
                    else:
                        return self._create_fallback_questions(num_questions, focus_topics, f"HTTP {response.status_code}")

        except Exception as e:
            logger.error(f"Error generating quiz: {str(e)}")
            return self._create_fallback_questions(num_questions, focus_topics, str(e))

    async def _generate_quiz_two_stage(self,
                                    content: str,
                                    num_questions: int = 5,
                                    focus_topics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Advanced two-stage quiz generation approach:
        1. First stage: Generate factually correct questions with single correct answers
        2. Second stage: Verify and align the correct answer with the explanation

        Args:
            content: The text content to create a quiz from
            num_questions: Number of questions to generate
            focus_topics: Optional list of topics to focus on

        Returns:
            List of quiz questions with verified correct answers
        """
        try:
            # Stage 1: Generate questions and factually correct options
            logger.debug("Stage 1: Generating questions and factually correct options")

            stage1_prompt = f"""Based on the following content, create {num_questions} quiz questions.
For each question:
1. Create a clear, factual question
2. Provide ONE correct answer that is FACTUALLY ACCURATE
3. Do NOT create incorrect options yet

Content:
---
{content[:1500]}
---

{f"Focus on these topics: {', '.join(focus_topics)}" if focus_topics else ""}

Format your response as a JSON array:
```json
[
  {{
    "question": "What is the tallest mountain in the world?",
    "correct_answer": "Mount Everest, standing at 8,848.86 meters (29,031.7 feet) above sea level"
  }},
  // more questions...
]
```

IMPORTANT: Ensure absolute factual accuracy in your questions and answers. Only include information that is definitively contained in the content.
"""

            # Execute stage 1
            questions_with_correct_answers = await self._execute_ollama_prompt(stage1_prompt)

            if not questions_with_correct_answers:
                logger.warning("Stage 1 failed to generate valid questions")
                return []

            # Parse stage 1 results
            try:
                questions_data = self._parse_json_response(questions_with_correct_answers)
                if not questions_data or not isinstance(questions_data, list) or len(questions_data) == 0:
                    raise ValueError("Invalid stage 1 response format")

                logger.debug(f"Stage 1 succeeded: Generated {len(questions_data)} question bases")
            except Exception as e:
                logger.error(f"Failed to parse stage 1 results: {str(e)}")
                return []

            # Stage 2: Process each question to generate distractors and finalize
            final_questions = []

            for idx, q_base in enumerate(questions_data):
                if len(final_questions) >= num_questions:
                    break

                if 'question' not in q_base or 'correct_answer' not in q_base:
                    logger.warning(f"Skipping invalid question at index {idx}")
                    continue

                question = q_base['question']
                correct_answer = q_base['correct_answer']

                # Generate distractors for this question
                final_q = await self._generate_question_with_distractors(
                    question,
                    correct_answer,
                    content
                )

                if final_q:
                    final_questions.append(final_q)

            return final_questions

        except Exception as e:
            logger.error(f"Two-stage quiz generation failed: {str(e)}")
            return []

    async def _generate_question_with_distractors(self,
                                              question: str,
                                              correct_answer: str,
                                              context: str) -> Optional[Dict[str, Any]]:
        """
        Generate a complete quiz question with distractors

        Args:
            question: The question text
            correct_answer: The known correct answer
            context: The content for context

        Returns:
            Completed question dictionary or None if failed
        """
        try:
            # Use a dedicated prompt to generate plausible distractors
            prompt = f"""For this quiz question:
Question: "{question}"
Correct answer: "{correct_answer}"

Generate THREE incorrect but plausible alternative options that would serve as good distractors.
These should be different enough from the correct answer but still plausible to someone who doesn't know the answer.
Then, place the correct answer and the three distractors in a random order in an array.
Tell me the index (0-3) of where the correct answer appears in this randomized array.
Finally, provide a brief explanation that mentions the correct answer and why it's correct.

Return your response in this JSON format:
```json
{{
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "correct_index": 0,
  "explanation": "Option A is correct because..."
}}
```

Make sure the explanation explicitly mentions which option is the correct answer and why.
"""

            # Execute the prompt
            response = await self._execute_ollama_prompt(prompt)
            if not response:
                return None

            # Parse the response
            try:
                distractor_data = self._parse_json_response(response)

                if not distractor_data or not isinstance(distractor_data, dict):
                    return None

                if 'options' not in distractor_data or not isinstance(distractor_data['options'], list) or len(distractor_data['options']) != 4:
                    logger.warning("Invalid options in distractor generation")
                    return None

                if 'correct_index' not in distractor_data or not isinstance(distractor_data['correct_index'], int):
                    logger.warning("Missing or invalid correct_index in distractor generation")
                    return None

                # Validate the correct answer is actually in the options at the specified index
                correct_index = distractor_data['correct_index']
                if correct_index < 0 or correct_index >= len(distractor_data['options']):
                    logger.warning(f"Correct index {correct_index} out of bounds for options list")
                    return None

                # Check if the explanation mentions the correct option
                explanation = distractor_data.get('explanation', '')
                if not explanation:
                    explanation = f"The correct answer is: {distractor_data['options'][correct_index]}"

                # Create the final question object
                return {
                    "question": question,
                    "options": distractor_data['options'],
                    "correct_answer": correct_index,
                    "explanation": explanation
                }

            except Exception as parse_e:
                logger.error(f"Failed to parse distractor response: {str(parse_e)}")
                return None

        except Exception as e:
            logger.error(f"Failed to generate distractors: {str(e)}")
            return None

    async def _execute_ollama_prompt(self, prompt: str) -> Optional[str]:
        """
        Execute a prompt using Ollama

        Args:
            prompt: The prompt to send to Ollama

        Returns:
            Response text or None if failed
        """
        try:
            # Check if we need to reconnect
            if not self.api_verified:
                self._verify_ollama_connection()

            # Execute API call
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,  # Lower temperature for more factual responses
                        "num_predict": 2000
                    }
                }

                response = await client.post(f"{self.base_url}/api/generate", json=payload, timeout=60.0)

                if response.status_code == 200:
                    try:
                        response_data = response.json()
                        return response_data.get("response", "")
                    except Exception as e:
                        logger.error(f"Failed to parse Ollama response: {str(e)}")
                        return None
                else:
                    logger.error(f"Ollama API returned error code: {response.status_code}")
                    return None

        except Exception as e:
            logger.error(f"Error executing Ollama prompt: {str(e)}")
            return None

    def _parse_json_response(self, response: str) -> Optional[Any]:
        """
        Parse JSON from an Ollama response

        Args:
            response: Text response from Ollama

        Returns:
            Parsed JSON object or None if parsing failed
        """
        try:
            # Try to extract JSON content from the response
            # First, look for JSON in code blocks
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if match:
                json_str = match.group(1)
            else:
                # Try to find JSON object/array directly
                json_match = re.search(r'(\[|\{)[\s\S]*?(\]|\})', response)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response

            # Clean up the JSON string
            json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)  # Remove comments
            json_str = re.sub(r',\s*[\]\}]', r'\1', json_str)  # Remove trailing commas

            # Try to parse the JSON
            return json.loads(json_str)

        except Exception as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            logger.debug(f"Response was: {response[:200]}...")
            return None

    async def _process_and_ensure_options(self, questions: List[Dict[str, Any]], content: str) -> List[Dict[str, Any]]:
        """
        Process questions and ensure all have valid options and explanations

        Args:
            questions: List of question dictionaries
            content: Original content for generating options if needed

        Returns:
            List of validated and enhanced question dictionaries
        """
        try:
            validated_questions = []

            for question in questions:
                if "question" not in question or not question["question"].strip():
                    continue

                # Ensure the question has a valid format
                processed_q = {
                    "question": question["question"].strip(),
                    "options": [],
                    "correct_answer": question.get("correct_answer", 0),
                    "explanation": question.get("explanation")
                }

                # Check if we need to generate options
                needs_options = (
                    "options" not in question or
                    not isinstance(question.get("options"), list) or
                    len(question.get("options", [])) < 4 or
                    all(not opt.strip() for opt in question.get("options", []))
                )

                if needs_options:
                    # Generate options for this question
                    generated_options = await self._generate_question_options(processed_q["question"], content)
                    processed_q["options"] = generated_options
                    # If we had to generate all options, ensure correct_answer is valid
                    processed_q["correct_answer"] = min(processed_q["correct_answer"], 3)
                else:
                    # Use the existing options but ensure we have exactly 4
                    options = question.get("options", [])
                    # Filter out empty options
                    options = [opt for opt in options if opt.strip()]
                    # Pad if needed
                    if len(options) < 4:
                        additional_options = await self._generate_question_options(
                            processed_q["question"],
                            content,
                            4 - len(options)
                        )
                        options.extend(additional_options)
                    # Trim if too many
                    processed_q["options"] = options[:4]

                # Ensure we have an explanation
                if not processed_q.get("explanation"):
                    processed_q["explanation"] = f"The correct answer is {processed_q['options'][processed_q['correct_answer']]}"

                # Verify consistency between correct_answer index and explanation
                if processed_q.get("explanation") and processed_q.get("options"):
                    correct_option = processed_q["options"][processed_q["correct_answer"]]

                    # Make sure the explanation mentions the correct option
                    if correct_option not in processed_q["explanation"]:
                        processed_q["explanation"] = f"The correct answer is {correct_option}. {processed_q['explanation']}"

                validated_questions.append(processed_q)

                # Limit to reasonable number of questions
                if len(validated_questions) >= 10:
                    break

            return validated_questions

        except Exception as e:
            logger.error(f"Error processing questions: {str(e)}")
            return []

    async def _generate_question_options(self, question: str, content: str, num_options: int = 4) -> List[str]:
        """
        Generate options for a quiz question using the model

        Args:
            question: The question to generate options for
            content: The content context for generating relevant options
            num_options: Number of options to generate

        Returns:
            List of option strings
        """
        try:
            # Create a prompt to generate just the options
            # Limit content to prevent excessively long prompts
            trimmed_content = content[:1000] if len(content) > 1000 else content
            prompt = f"""Based on this content snippet:
---
{trimmed_content}
---

For this question: "{question}"

Generate exactly {num_options} different multiple-choice options that are plausible answers.
Make one option clearly correct, and the others plausible but incorrect.
Format each option as a complete sentence or phrase that directly answers the question.
Return only the {num_options} options as a simple list, one per line.
"""

            logger.debug(f"Generating options for question: {question[:50]}...")

            # Direct API call with shorter timeout
            async with httpx.AsyncClient(timeout=30.0) as client:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,  # Slightly higher for creativity
                        "num_predict": 500   # Short limit for options only
                    }
                }

                try:
                    response = await client.post(f"{self.base_url}/api/generate", json=payload, timeout=30.0)

                    if response.status_code == 200:
                        response_data = response.json()
                        response_text = response_data.get("response", "")

                        # Extract options from text
                        lines = [line.strip() for line in response_text.split('\n') if line.strip()]

                        # Process options - remove numbers, bullets, etc.
                        processed_options = []
                        for line in lines:
                            # Remove numbering and bullets
                            line = re.sub(r'^[\d\-\*\•\.\)]+\s*', '', line)
                            # Remove option lettering (A., B:, etc)
                            line = re.sub(r'^[A-D][\s\.\:\)]+', '', line)
                            if line.strip():
                                processed_options.append(line.strip())

                        # Ensure we have enough options
                        if len(processed_options) >= num_options:
                            return processed_options[:num_options]

                        # If we don't have enough, add some generic ones
                        while len(processed_options) < num_options:
                            processed_options.append(f"Option {len(processed_options) + 1}")

                        return processed_options

                except Exception as e:
                    logger.error(f"Error generating options: {str(e)}")

            # Fallback options if generation fails
            return [
                f"Option 1 for '{question[:20]}...'",
                f"Option 2 for '{question[:20]}...'",
                f"Option 3 for '{question[:20]}...'",
                f"Option 4 for '{question[:20]}...'"
            ][:num_options]

        except Exception as e:
            logger.error(f"Error in option generation: {str(e)}")
            return [f"Option {i+1}" for i in range(num_options)]

    def _parse_questions_fallback(self, text: str, num_questions: int) -> List[Dict[str, Any]]:
        """
        Fallback method to parse quiz questions from non-JSON formatted text

        Args:
            text: Response text to parse
            num_questions: Maximum number of questions to extract

        Returns:
            List of question dictionaries
        """
        try:
            # Try to manually extract and format questions
            questions = []
            current_q = {}
            lines = text.split('\n')

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Question detection (looks for "Q1" or "Question 1" patterns, or lines ending with "?")
                if (re.match(r'^[Qq](?:uestion)?\s*\d+', line) or line.rstrip().endswith('?')) and not line.startswith(('A)', 'B)', 'C)', 'D)')):
                    if current_q and "question" in current_q:
                        questions.append(current_q)
                    current_q = {"question": line.split(":", 1)[-1].strip() if ':' in line else line, "options": []}

                # Option detection
                elif re.match(r'^[A-D][\s:\.\)]+', line) and current_q:
                    option_text = line.split(":", 1)[-1].strip() if ':' in line else line[2:].strip()
                    current_q["options"].append(option_text)

                # Correct answer detection
                elif any(marker in line.lower() for marker in ["correct answer", "answer:", "correct:"]) and current_q:
                    answer_match = re.search(r'[A-D]', line)
                    if answer_match:
                        answer_letter = answer_match.group(0)
                        current_q["correct_answer"] = ord(answer_letter) - ord('A')

                # Explanation detection
                elif any(marker in line.lower() for marker in ["explanation:", "reason:"]) and current_q:
                    current_q["explanation"] = line.split(":", 1)[-1].strip()

            # Add the last question
            if current_q and "question" in current_q:
                # Ensure all questions have required fields
                if "correct_answer" not in current_q:
                    current_q["correct_answer"] = 0
                if len(current_q.get("options", [])) < 4:
                    current_q["options"] = current_q.get("options", []) + [""] * (4 - len(current_q.get("options", [])))
                questions.append(current_q)

            # Limit to requested number of questions
            questions = questions[:num_questions]

            return questions
        except Exception as e:
            logger.error(f"Fallback parsing failed: {str(e)}")
            return []

    def _create_fallback_questions(self, num_questions: int, focus_topics: Optional[List[str]], error_reason: str) -> List[Dict[str, Any]]:
        """
        Create basic fallback questions when generation fails

        Args:
            num_questions: Number of questions to create
            focus_topics: Topics to focus on
            error_reason: Reason for fallback

        Returns:
            List of basic question dictionaries
        """
        topic = focus_topics[0] if focus_topics else "the content"
        logger.warning(f"Using fallback questions due to: {error_reason}")

        return [
            {
                "question": f"Question about {topic}? (Note: Quiz generation encountered an error: {error_reason})",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_answer": 0,
                "explanation": f"Generated question due to error: {error_reason}"
            }
        ] * min(num_questions, 3)