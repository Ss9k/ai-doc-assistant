# AI Document Assistant

A FastAPI-based application for uploading documents, storing document metadata in PostgreSQL, generating AI-powered responses, and extracting structured information from resumes using LLMs.

## Features

- Health check endpoint
- Upload PDF, TXT, and DOCX files
- Validate file extensions and reject empty or invalid files
- Enforce a maximum file size of 10 MiB
- Generate unique document IDs and upload timestamps
- Persist document metadata in PostgreSQL
- List uploaded document metadata
- AI-powered document question answering using Gemini and Ollama
- Streaming LLM responses
- Track LLM usage, token consumption, latency, and estimated API costs
- Extract structured resume information using Pydantic validation
- Retry handling for transient LLM API failures
- Automated API tests using pytest

## Tech Stack

- **Language:** Python
- **Backend:** FastAPI
- **Data Validation:** Pydantic
- **Database:** PostgreSQL
- **ORM:** SQLModel
- **Migrations:** Alembic
- **LLM Providers:** Google Gemini, Ollama
- **HTTP Client:** HTTPX
- **Testing:** Pytest
- **Package Management:** uv
- **Infrastructure:** Docker Compose

## Prerequisites

- Python version compatible with the project requirements
- [uv](https://docs.astral.sh/uv/)
- Docker Desktop (for PostgreSQL through Docker Compose)
- Gemini API key (for Gemini integration)
- Ollama (if using the local Ollama provider)

## Setup

Clone the repository and navigate to the project directory.

Install dependencies:

```bash
uv sync
```

Configure environment variables in a `.env` file:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/ai_doc_assistant

GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

Start the PostgreSQL database using Docker Compose:

```bash
docker compose up -d
```

Apply database migrations:

```bash
uv run python -m alembic upgrade head
```

## Run the Application

```bash
uv run uvicorn app.main:app --reload --port 8001
```

The API will be available at:

http://127.0.0.1:8001

## API Documentation

Explore and test the API using Swagger UI:

http://127.0.0.1:8001/docs

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check API health |
| POST | `/documents` | Upload a document |
| GET | `/documents` | List uploaded document metadata |
| POST | `/ask` | Ask questions using an LLM provider |
| POST | `/extract` | Extract structured information from a resume |

## Upload Restrictions

- Supported extensions: `.pdf`, `.txt`, `.docx`
- Maximum file size: 10 MiB
- Empty files are rejected
- Invalid or unsupported document formats are rejected

## Storage

Document metadata is persisted in PostgreSQL using SQLModel.

Stored metadata includes:

- Document ID
- Filename
- Content type
- File size
- Upload timestamp

Database schema changes are managed through Alembic migrations.

**Note:** Uploaded file contents are not persisted in PostgreSQL. The database currently stores document metadata only.

## LLM Usage Tracking

The application supports Google Gemini and Ollama as LLM providers.

LLM usage information is tracked, including:

- Provider and model
- Input and output token counts
- Request latency
- Estimated API cost

Usage records are stored in PostgreSQL.

## Resume Extraction Evaluation

The resume extraction functionality was evaluated using a dataset of 20 resumes.

Evaluation results:

- Resumes evaluated: 20
- Valid API responses: 20/20
- Correct fields: 80/80
- Field accuracy: 100% on the evaluation dataset
- Prompt-injection test: Passed

**Note:** These results represent the tested evaluation dataset and do not guarantee 100% accuracy on arbitrary resumes.

## Prompt Caching

Prompt caching reuses previously processed prompt prefixes across repeated API requests.

It can reduce latency and input-token costs when system instructions and examples remain unchanged. Cache writes may have additional costs, while cache reads can be cheaper than regular input tokens.

It is particularly useful for applications that repeatedly send large, consistent prompts.

**Note:** Prompt caching is currently documented as a potential optimization and is not implemented in this application.

## Run Tests

Run the automated test suite:

```bash
uv run pytest -v
```

## Database Migrations

Check the current migration revision:

```bash
uv run python -m alembic current
```

View migration history:

```bash
uv run python -m alembic history
```

Apply pending migrations:

```bash
uv run python -m alembic upgrade head
```