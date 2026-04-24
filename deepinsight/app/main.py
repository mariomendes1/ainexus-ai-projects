"""DeepInsight – Analisador de Sentimento e Tópicos com IA Local."""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import io
import os

app = FastAPI(title="DeepInsight API", version="0.1.0", root_path="/deepinsight")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = os.getenv(
    "SENTIMENT_MODEL", "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = None
sentiment_model = None


class TextInput(BaseModel):
    text: str


class BatchTextInput(BaseModel):
    texts: list[str]


class TopicRequest(BaseModel):
    texts: list[str]
    n_topics: int = 5
    n_words: int = 8


def _binary_scores_from_raw(raw_scores: dict[str, float]) -> dict[str, float]:
    labels = {str(k).upper(): float(v) for k, v in raw_scores.items()}
    pos = 0.0
    neg = 0.0
    neu = 0.0

    for key, value in labels.items():
        if "POS" in key:
            pos += value
        elif "NEG" in key:
            neg += value
        elif "NEU" in key:
            neu += value
        elif "STAR" in key:
            if "1" in key or "2" in key:
                neg += value
            elif "4" in key or "5" in key:
                pos += value
            elif "3" in key:
                neu += value
        elif key == "LABEL_0":
            neg += value
        elif key == "LABEL_1":
            neu += value
        elif key == "LABEL_2":
            pos += value

    if pos == 0 and neg == 0 and neu == 0:
        # Safe fallback: split uniform probability.
        pos = 0.5
        neg = 0.5
    else:
        # Project neutral into binary space for UI compatibility.
        pos += neu * 0.5
        neg += neu * 0.5
        total = pos + neg
        if total > 0:
            pos /= total
            neg /= total

    return {"POSITIVE": round(pos, 4), "NEGATIVE": round(neg, 4)}


def _model_sentiment(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    labels = sentiment_model.config.id2label
    raw_scores = {labels[i]: float(probs[i]) for i in range(len(probs))}
    scores = _binary_scores_from_raw(raw_scores)
    predicted = "POSITIVE" if scores["POSITIVE"] >= scores["NEGATIVE"] else "NEGATIVE"
    confidence = round(max(scores["POSITIVE"], scores["NEGATIVE"]), 4)
    return {"sentiment": predicted, "scores": scores, "confidence": confidence}


def _analyze_sentiment(text: str):
    return _model_sentiment(text)


@app.on_event("startup")
def load_model():
    global tokenizer, sentiment_model
    cache_dir = "/app/models/transformers"
    os.makedirs(cache_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, cache_dir=cache_dir
    )
    sentiment_model.to(DEVICE)
    sentiment_model.eval()


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}


@app.post("/sentiment")
def analyze_sentiment(inp: TextInput):
    """Analyze sentiment of a single text."""
    result = _analyze_sentiment(inp.text)
    return {"sentiment": result["sentiment"], "scores": result["scores"]}


@app.post("/sentiment/batch")
def analyze_sentiment_batch(inp: BatchTextInput):
    """Analyze sentiment for multiple texts."""
    results = []
    for text in inp.texts:
        result = _analyze_sentiment(text)
        results.append(
            {
                "text": text[:80],
                "sentiment": result["sentiment"],
                "confidence": result["confidence"],
                "scores": result["scores"],
            }
        )
    return {"results": results, "total": len(results)}


@app.post("/topics")
def extract_topics(req: TopicRequest):
    """Extract topics from a list of texts using LDA."""
    texts = [t.strip() for t in req.texts if isinstance(t, str) and t.strip()]
    if len(texts) < 2:
        raise HTTPException(
            status_code=400,
            detail="Para extrair tópicos, insere pelo menos 2 textos.",
        )

    min_df = 2 if len(texts) >= 5 else 1
    vectorizer = CountVectorizer(max_df=0.95, min_df=min_df)
    try:
        dtm = vectorizer.fit_transform(texts)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Não foi possível extrair tópicos com os textos fornecidos. Tenta adicionar mais textos variados.",
        ) from exc

    n_components = max(1, min(req.n_topics, dtm.shape[1]))
    lda = LatentDirichletAllocation(
        n_components=n_components, random_state=42, max_iter=20
    )
    lda.fit(dtm)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for i, topic in enumerate(lda.components_):
        top_words = [feature_names[j] for j in topic.argsort()[-req.n_words :]]
        topics.append({"topic_id": i, "words": top_words})
    return {"topics": topics, "n_documents": len(texts)}


@app.post("/analyze/file")
async def analyze_file(file: UploadFile = File(...)):
    """Upload CSV/TXT and get sentiment + topics."""
    try:
        content = await file.read()
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
            text_col = df.select_dtypes(include="object").columns[0]
            texts = df[text_col].dropna().tolist()
        else:
            texts = content.decode("utf-8").strip().split("\n")

        # Sentiment
        sentiments = []
        for text in texts[:200]:  # Limit to 200 for performance
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = sentiment_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            labels = sentiment_model.config.id2label
            predicted = labels[int(torch.argmax(probs))]
            sentiments.append(predicted)

        # Topics (LDA)
        if len(texts) >= 5:
            vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
            dtm = vectorizer.fit_transform(texts)
            lda = LatentDirichletAllocation(n_components=5, random_state=42, max_iter=20)
            lda.fit(dtm)
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            for i, topic in enumerate(lda.components_):
                top_words = [feature_names[j] for j in topic.argsort()[-8:]]
                topics.append({"topic_id": i, "words": top_words})
        else:
            topics = []

        from collections import Counter

        sentiment_counts = dict(Counter(sentiments))

        return {
            "total_texts": len(texts),
            "analyzed": len(sentiments),
            "sentiment_distribution": sentiment_counts,
            "topics": topics,
        }
    finally:
        await file.close()
