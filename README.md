# AI Nexus — Machine Learning Projects

A collection of four production-ready AI/ML microservices, each packaged as a Docker container with a FastAPI backend and a web frontend.

## Projects

### NexusGuard — Anomaly & Fraud Detection
> Port `8000`

Detects anomalies and potential fraud in transactional data using **Scikit-Learn's Isolation Forest**. Trains on demand and exposes a REST API for real-time scoring.

- Algorithm: Isolation Forest (unsupervised)
- Stack: FastAPI · Scikit-Learn · Pandas · NumPy
- RAM: ~500 MB

---

### DeepInsight — Sentiment & Topic Analysis
> Port `8001`

Multilingual sentiment analysis and topic extraction powered by **DistilBERT** (PyTorch / HuggingFace Transformers). Supports Portuguese, English, and other languages out of the box.

- Model: `lxyuan/distilbert-base-multilingual-cased-sentiments-student`
- Stack: FastAPI · PyTorch · HuggingFace Transformers
- RAM: ~2–3 GB

---

### VisionStock — Intelligent Inventory Prediction
> Port `8002`

Predicts future stock levels and demand using **XGBoost** and Scikit-Learn ensemble models. Accepts historical sales data and returns forecasts with confidence intervals.

- Algorithms: XGBoost · Scikit-Learn
- Stack: FastAPI · XGBoost · Pandas · NumPy · Joblib
- RAM: ~1 GB

---

### RealState — AI Agent for Real Estate Market
> Port `8003`

An agentic system with three specialised agents (valuation, market analysis, geospatial) that collaborate to answer real estate queries. Integrates with a local **Ollama** model (Granite 3.1) and optional official connectors.

- Architecture: Multi-agent orchestration
- Stack: FastAPI · Ollama (`granite3.1-dense:2b`) · HTTPX · BeautifulSoup4
- RAM: ~896 MB

---

## Getting Started

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- For RealState: [Ollama](https://ollama.com/) running locally with `granite3.1-dense:2b` pulled

### Run all services

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| NexusGuard | http://localhost:8000 |
| DeepInsight | http://localhost:8001 |
| VisionStock | http://localhost:8002 |
| RealState | http://localhost:8003 |

### RealState configuration (optional)

Copy the example connector file and fill in your credentials:

```bash
cp realstate/.env.connector.example realstate/.env.connector
```

Edit `.env.connector` with your API base URL and token. Leave blank to run without the official connector.

---

## Architecture

```
apps.ainexuspt.com/
├── docker-compose.yml      # Orchestrates all 4 services
├── nexusguard/             # Anomaly & fraud detection
├── deepinsight/            # Sentiment & topic analysis
├── visionstock/            # Inventory forecasting
└── realstate/              # Real estate AI agent
```

Each service follows the same structure:
```
<service>/
├── Dockerfile
├── requirements.txt
├── app/          # FastAPI application
├── frontend/     # Web UI
├── models/       # Persisted model artifacts (Docker volume)
└── static/       # Static assets
```

---

## License

MIT
