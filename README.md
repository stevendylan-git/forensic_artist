# Forensic Face Composite Generator (PS-01)

AI-assisted eyewitness composite generation system adhering to forensic evidentiary standards, structured cognitive interviewing, region-scoped edits, and tamper-evident audit logging.

## Core Rules & Invariants
- **Backend Secrets Only**: `GEMINI_API_KEY` is loaded strictly in `backend/.env`. Never transmitted to the client.
- **Model IDs from Config**: `GEMINI_TEXT_MODEL` and `GEMINI_IMAGE_MODEL` are managed exclusively in `backend/app/config.py`.
- **FaceSpec Source of Truth**: Images render the structured JSON FaceSpec. Unstated features stay `"unknown"`.
- **Region-Scoped Locality**: Edits preserve locked regions with pixel-identical integrity via masked compositing.
- **Audit Logging**: Append-only, SHA-256 hash-chained audit logging for every endpoint and action.
- **Anti-Leading & Safety**: Open questions and candidate grids only; real named individuals are rejected immediately and logged.

---

## Quickstart

### Prerequisites
- Node.js 18+ (Node 20+ recommended)
- Python 3.11+
- PostgreSQL or Docker Compose

### 1. Environment Setup
```bash
# Copy environment configuration
cp .env.example backend/.env
# Edit backend/.env and populate your GEMINI_API_KEY
```

### 2. Run with Docker Compose
```bash
docker compose up --build
```
- Frontend: `http://localhost:5173`
- Backend API Docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/v1/health`

### 3. Local Development (Without Docker)

#### Backend
```bash
cd backend
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
```

### 4. Running Backend Tests
```bash
cd backend
pytest -v
```
