# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a FastAPI-based sentiment analysis API that uses a Hybrid Fusion model combining VADER (lexicon-based) and SVM (machine learning) for sentiment classification. The model classifies text into Positive, Neutral, or Negative sentiments with confidence scores.

## Common Commands

```bash
# Install dependencies (use virtual environment in commentSense/)
pip install -r requirements.txt

# Run the API server
python main.py
# Or with uvicorn directly:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Test the API (requires server running)
python test_api.py
```

## Architecture

### Core Files

- **main.py** - FastAPI application with endpoints:
  - `GET /health` - Health check
  - `POST /predict` - Single text prediction
  - `POST /predict/batch` - Multiple texts prediction
  - `POST /predict/array` - Array input with aggregated analysis response

- **model_predictor.py** - Model loading and prediction logic:
  - `HybridFusionPredictor` class wraps the ML pipeline
  - `get_predictor()` provides singleton access to loaded model
  - Combines TF-IDF + SVM predictions with VADER lexicon scores using weighted fusion

### Model Pipeline

The hybrid fusion approach:
1. Text is vectorized using TF-IDF
2. SVM provides probability distribution over classes
3. VADER provides lexicon-based sentiment compound score (converted to class probabilities)
4. Final prediction = `fusion_weight * SVM + (1 - fusion_weight) * VADER` (default: 90% SVM, 10% VADER)

### Key Files

- `outputs/models/hybrid_fusion_model.pkl` - Trained model artifacts (TF-IDF vectorizer, calibrated SVM, label encoder, config)
- `Hybrid_Fusion_VADER_SVM_Optimized(mine)_1765589021657.ipynb` - Jupyter notebook for training the model
- `commentSense/` - Python virtual environment

## Environment Variables

- `MODEL_PATH` - Override default model path (default: `outputs/models/hybrid_fusion_model.pkl`)

## API Response Formats

### `/predict` and `/predict/batch` (per-comment)
- `text`: Original comment text
- `prediction`: "Positive"/"Neutral"/"Negative"
- `confidence`: Overall confidence score
- `probabilities`: Distribution across all classes
- `vader_score`: Raw VADER compound score (-1 to 1)
- `svm_confidence`/`vader_confidence`: Component model confidences
- `fusion_weights`: Weights used in fusion

### `/predict/array` (aggregated + per-comment)
- `sentimentScore`: Average confidence across all comments
- `aspectScore`: Calculated from probability distributions
- `effectivenessScore`: Based on VADER/SVM agreement
- `positiveAspects`/`negativeAspects`: Derived aspect tags with scores
- `commentAnalysis`: Array of per-comment predictions (same format as `/predict`)

### `/health`
- `status`: "healthy"/"unhealthy"
- `model_loaded`: Boolean
- `fusion_weights`: Current SVM/VADER weight configuration
