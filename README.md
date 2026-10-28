# AI Document Assistant

A FastAPI application that accepts document uploads and returns document metadata.

## Features

- Health check endpoint
- Upload PDF, TXT, and DOCX files
- Validate file extensions and reject empty files
- Enforce a maximum file size of 10 MiB
- Generate a unique ID and upload timestamp for each document
- List uploaded document metadata
- Automated API tests using pytest

## Tech Stack

- Python
- FastAPI
- Pydantic
- Uvicorn
- Pytest
- HTTPX

## Prerequisites

- Python version compatible with the project requirements
- [uv](https://docs.astral.sh/uv/)

## Setup

Clone the repository and navigate to the project directory.

Install the project dependencies:

```bash
uv sync
```

## Run the Application

```bash
uv run uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## API Documentation

Open `http://127.0.0.1:8000/docs` in your browser to explore and test the API using Swagger UI.

## Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check API health |
| POST | `/documents` | Upload a document |
| GET | `/documents` | List uploaded document metadata |

## Upload Restrictions

- Supported extensions: `.pdf`, `.txt`, `.docx`
- Maximum file size: 10 MiB
- Empty files are rejected

## Run Tests

```bash
uv run pytest -v
```

## Storage

Document metadata is stored in memory. Uploaded metadata is cleared when the application restarts. This version does not persist uploaded file contents or use a database.
