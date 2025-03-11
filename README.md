# FastAPI Chatbot

A RESTful API for a chatbot built with FastAPI, providing structured endpoints for chat interactions and content summarization.

## Project Structure

```
.
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── dependencies.py
│   │   ├── errors.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── chat.py
│   │       └── summarizer.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logger.py
│   │   └── errors.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schema.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── chatbot.py
│   │   └── summarizer.py
│   └── __init__.py
├── logs/
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── .env
├── .gitignore
├── main.py
├── requirements.txt
└── README.md
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- pip (Python package installer)
- Ollama (for LLM functionality)

### Installing and Setting up Ollama

Ollama is required to run the local LLM used by the chatbot and summarization features.

#### Windows

```bash
# Download and install Ollama from the official website
# https://ollama.com/download/windows

# After installation, pull the required model
ollama pull llama3:latest
```

#### macOS

```bash
# Install Ollama using Homebrew
brew install ollama

# Or download from the official website
# https://ollama.com/download/mac

# Pull the required model
ollama pull llama3:latest
```

#### Linux

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the required model
ollama pull llama3:latest
```

After installing, make sure the Ollama service is running:
```bash
# Start Ollama service
ollama serve
```
The Ollama API should be available at http://localhost:11434

### Setting Up a Virtual Environment

#### Windows

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate

# If you're using PowerShell instead of Command Prompt
# venv\Scripts\Activate.ps1
```

#### macOS/Linux

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### Installing Dependencies

Once your virtual environment is activated, install the required packages:

```bash
pip install -r requirements.txt

# Download SpaCy language model
python -m spacy download en_core_web_sm
```

## Running the Application

### Development Server

```bash
uvicorn main:app --reload
```

The API will be available at http://127.0.0.1:8000

### API Documentation

FastAPI automatically generates API documentation using Swagger UI and ReDoc:

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```
APP_ENV=development
DEBUG=true
LOG_LEVEL=INFO
```

## API Endpoints

### Chat Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/chat/message` | POST | Send a message to the chatbot |
| `/api/v1/health` | GET | Check API health status |

### Summarization Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/summarize/youtube` | POST | Summarize a YouTube video from its URL |
| `/api/v1/summarize/pdf` | POST | Summarize a PDF document from an uploaded file |
| `/api/v1/summarize/text` | POST | Summarize plain text content |

## Browser-Based Session Management

The chatbot uses cookies to automatically track user sessions:

- No login or user accounts required
- Sessions are created and managed automatically via browser cookies
- Cookies expire after 90 days
- When a user visits for the first time, a unique session ID is generated
- All subsequent requests from the same browser reuse the same session
- No need to manually provide or track session IDs

This approach simplifies the user experience while maintaining conversation context across visits.

## Summarization Features

The API provides content summarization with the following options:

### Summary Length Options
- **Short**: 1-2 paragraphs
- **Medium**: 3-5 paragraphs
- **Long**: 5-7 paragraphs

### Summary Style Options
- **Narrative**: Flowing text format
- **Bullet Points**: Bullet point format
- **Academic**: Formal, structured format
- **Simplified**: Easy to understand, non-technical language

### YouTube Video Summarization
Extracts and summarizes transcripts from YouTube videos, with options to include timestamped sections. Features:
- Automatic transcript extraction and translation if needed
- Video metadata extraction
- Key points identification
- Timestamped section detection

### PDF Document Summarization
Extracts and summarizes text from PDF documents, with options to specify page ranges.

### Text Summarization
Summarizes plain text content with various style and length options.

## Form-Based API

All endpoints use form-based input with the following benefits:
- Easy to test directly in the Swagger UI
- Dropdown selectors for summary length and style
- File upload for PDF documents
- Optional fields with clear documentation

## Technical Implementation

- Text processing and summarization use SpaCy for natural language processing
- YouTube transcripts are extracted using the YouTube Transcript API
- PDF text extraction uses pdfplumber
- No authentication required - completely open API
- Cookie-based session tracking

## Error Handling

This project includes comprehensive error handling with:
- Custom exception classes
- Proper HTTP status codes
- Structured error responses
- Request validation

## Logging

Logs are stored in the `logs/` directory with different levels (INFO, WARNING, ERROR) and automatic rotation.
