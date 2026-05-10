<div align="center">
  <h1>🎯 SHL Conversational Assessment AI</h1>
  <p>A production-grade, stateless API that acts as an expert consultant for SHL's Assessment Catalog.</p>
</div>

---

## 📌 Project Overview
This project fulfills the SHL AI Intern Assignment requirements. It provides a robust, low-latency FastAPI microservice that guides hiring managers from vague intents to grounded shortlists of SHL Individual Test Solutions.

**Key Capabilities:**
* **Stateless Memory:** Evaluates full conversation context on every turn without relying on fragile server-side session memory.
* **Semantic Retrieval Engine:** Utilizes an embedded FAISS vector database to prevent AI hallucinations, ensuring 100% adherence to the SHL catalog.
* **Strict Output Structuring:** Employs Pydantic and native LLM JSON schema injection to mathematically guarantee valid API responses.
* **Defensive Guardrails:** Heavily prompt-engineered to refuse out-of-scope inquiries and demand clarification before recommending.

---

## 🏗️ Architecture

```text
SHL/
├── app/                              # Core API Logic
│   ├── main.py                       # FastAPI Endpoints
│   ├── agent.py                      # Gemini LLM Agent & Prompts
│   ├── models.py                     # Pydantic Schemas
│   └── retriever.py                  # FAISS Vector Search Engine
├── data/                             # RAG Context
│   └── shl_product_catalog_clean.json
├── frontend/                         # User Interface
│   └── index.html                    # Chat UI for testing
├── tests/                            # Evaluation Harness
│   └── evaluate.py
├── Dockerfile                        # Deployment Configuration
└── requirements.txt                  # Dependencies
```

---

## 🚀 Local Development Setup

### Prerequisites
* Python 3.11+
* Gemini API Key

### Installation

1. **Clone the repository & enter the directory**
2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: .\venv\Scripts\activate
   ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set Environment Variable**
   ```bash
   export GEMINI_API_KEY="your_api_key_here"  # Windows: $env:GEMINI_API_KEY="your_key"
   ```
5. **Start the API Server**
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

### Testing the UI
Once the server is running, simply open `frontend/index.html` in your web browser.

---

## 🌐 API Documentation

### `GET /health`
Validates that the service is running.
**Response:** `{"status": "ok"}`

### `POST /chat`
Accepts a stateless conversation history and returns the next agent reply.
**Request Payload:**
```json
{
  "messages": [
    {"role": "user", "content": "I am hiring a Java Developer."}
  ]
}
```
**Response Payload:**
```json
{
  "reply": "Are you looking for an assessment to test their technical skills or their cultural fit?",
  "recommendations": [],
  "end_of_conversation": false
}
```

---

## ☁️ Deployment Guide (Render)

This application is containerized and ready for 1-click deployment on [Render](https://render.com/).

1. Push this code to a public/private GitHub Repository.
2. Log into Render and click **New+** -> **Web Service**.
3. Connect your GitHub repository.
4. Render will automatically detect the `Dockerfile`.
5. Under **Environment Variables**, add:
   * Key: `GEMINI_API_KEY`
   * Value: `your_actual_api_key`
6. Click **Deploy Web Service**. Render will build the container and provide you with a live `https://...onrender.com` URL.

---

## 📊 Evaluation & Metrics
You can test the agent's performance against the provided SHL traces by running:
```bash
python tests/evaluate.py
```
This script acts as the automated grader, validating HTTP status codes, schema compliance, and end-of-conversation behavior.
