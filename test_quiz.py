import requests
import json
import time

# Test data
test_text = """
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.

The ideal characteristic of artificial intelligence is its ability to rationalize and take actions that have the best chance of achieving a specific goal. Machine Learning is a subset of AI where machines are designed to learn from data, identify patterns and make decisions with minimal human intervention.

Deep Learning is a further subset of Machine Learning that uses neural networks with many layers (hence "deep") to analyze various factors of data. Neural networks are computing systems inspired by the biological neural networks that constitute animal brains.

Natural Language Processing (NLP) allows machines to understand, interpret, and generate human language. Computer Vision enables machines to derive meaningful information from digital images, videos and other visual inputs.
"""

# API endpoint
url = "http://localhost:8000/api/v1/summarize/text"

# Request payload
payload = {
    "text": test_text,
    "output_type": "quiz",
    "num_quiz_questions": 3,
    "focus_topics": ["artificial intelligence", "machine learning", "deep learning"]
}

print("Sending request to API...")
start_time = time.time()

# Send POST request
response = requests.post(url, json=payload)

elapsed_time = time.time() - start_time
print(f"Request completed in {elapsed_time:.2f} seconds")

# Process response
if response.status_code == 200:
    quiz_data = response.json()
    print("\nQuiz generated successfully!\n")

    # Print the questions
    for i, q in enumerate(quiz_data["questions"], 1):
        print(f"Question {i}: {q['question']}")
        for j, option in enumerate(q["options"]):
            print(f"  {chr(65+j)}. {option}")
        print(f"Correct Answer: {chr(65+q['correct_answer'])}")
        if "explanation" in q and q["explanation"]:
            print(f"Explanation: {q['explanation']}")
        print()

    print(f"Total questions: {len(quiz_data['questions'])}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)