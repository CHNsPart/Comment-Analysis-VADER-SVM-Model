"""
FastAPI Application for Hybrid Fusion VADER + SVM Sentiment Analysis Model
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Union, Optional, Dict, Tuple
import uvicorn
import os
import re
import traceback
from collections import defaultdict

from model_predictor import get_predictor, DEFAULT_MODEL_PATH

# =============================================================================
# ASPECT EXTRACTION CONFIGURATION
# =============================================================================
# Define aspect categories with associated keywords for YouTube video analysis
ASPECT_KEYWORDS: Dict[str, List[str]] = {
    # Content & Quality
    "content quality": ["content", "quality", "production", "professional", "polished", "well-made", "high quality", "low quality"],
    "information": ["informative", "information", "learned", "educational", "knowledge", "facts", "research", "detailed"],
    "clarity": ["clear", "clarity", "understand", "easy to follow", "well explained", "simple", "straightforward", "concise"],
    "explanation": ["explain", "explanation", "tutorial", "how to", "guide", "instructions", "step by step", "walkthrough"],

    # Presentation
    "presenter": ["presenter", "host", "speaker", "voice", "personality", "charisma", "energy", "enthusiasm"],
    "pacing": ["pacing", "pace", "speed", "fast", "slow", "rushed", "too long", "too short", "length", "duration"],
    "engagement": ["engaging", "entertaining", "interesting", "boring", "dull", "captivating", "fun", "enjoyable"],

    # Technical
    "audio quality": ["audio", "sound", "volume", "music", "background music", "noise", "mic", "microphone"],
    "video quality": ["video", "visual", "graphics", "animation", "editing", "effects", "resolution", "4k", "hd"],

    # Value & Impact
    "usefulness": ["useful", "helpful", "practical", "valuable", "worth", "recommend", "must watch", "waste of time"],
    "originality": ["original", "unique", "creative", "innovative", "fresh", "new perspective", "different"],

    # Emotional Response
    "inspiration": ["inspired", "inspiring", "motivated", "motivation", "encouraging", "uplifting"],
    "humor": ["funny", "hilarious", "humor", "laugh", "comedy", "jokes", "witty"],

    # Criticism
    "confusion": ["confused", "confusing", "unclear", "lost", "don't understand", "doesn't make sense", "hard to follow"],
    "disappointment": ["disappointed", "disappointing", "expected more", "letdown", "overhyped", "clickbait"],
    "accuracy": ["accurate", "incorrect", "wrong", "mistake", "error", "misinformation", "outdated"],
}

# Keywords that indicate positive sentiment when found with aspects
POSITIVE_MODIFIERS = ["great", "amazing", "excellent", "awesome", "fantastic", "perfect", "best", "love", "loved",
                      "good", "nice", "wonderful", "brilliant", "superb", "outstanding", "incredible", "impressive"]

# Keywords that indicate negative sentiment when found with aspects
NEGATIVE_MODIFIERS = ["bad", "terrible", "awful", "horrible", "worst", "hate", "hated", "poor", "disappointing",
                      "boring", "annoying", "frustrating", "useless", "waste", "lacking", "missing", "needs improvement"]


def extract_aspects_from_comments(
    texts: List[str],
    predictions: List[dict]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Extract aspects from comments based on keyword matching and sentiment.

    Returns:
        Tuple of (positive_aspects, negative_aspects) dictionaries with aspect names and scores
    """
    positive_aspects: Dict[str, List[float]] = defaultdict(list)
    negative_aspects: Dict[str, List[float]] = defaultdict(list)

    for text, pred in zip(texts, predictions):
        text_lower = text.lower()
        sentiment = pred["prediction"]
        confidence = pred["confidence"]

        # Check for aspect keywords in the comment
        for aspect_name, keywords in ASPECT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Determine if this mention is positive or negative
                    has_positive_modifier = any(mod in text_lower for mod in POSITIVE_MODIFIERS)
                    has_negative_modifier = any(mod in text_lower for mod in NEGATIVE_MODIFIERS)

                    # Use both explicit modifiers and overall sentiment
                    if has_positive_modifier or (sentiment == "Positive" and not has_negative_modifier):
                        positive_aspects[aspect_name].append(confidence)
                    elif has_negative_modifier or (sentiment == "Negative" and not has_positive_modifier):
                        negative_aspects[aspect_name].append(confidence)
                    # For neutral or ambiguous, use the overall sentiment
                    elif sentiment == "Positive":
                        positive_aspects[aspect_name].append(confidence * 0.5)
                    elif sentiment == "Negative":
                        negative_aspects[aspect_name].append(confidence * 0.5)
                    break  # Only count each aspect once per comment

    # Calculate average scores for each aspect (frequency * avg confidence)
    def aggregate_scores(aspects_dict: Dict[str, List[float]]) -> Dict[str, float]:
        result = {}
        total_mentions = sum(len(scores) for scores in aspects_dict.values())
        if total_mentions == 0:
            return result

        for aspect, scores in aspects_dict.items():
            if scores:
                # Score = frequency weight * average confidence
                frequency_weight = len(scores) / total_mentions
                avg_confidence = sum(scores) / len(scores)
                # Combine frequency and confidence (both matter)
                result[aspect] = round((frequency_weight * 0.4 + avg_confidence * 0.6), 3)
        return result

    return aggregate_scores(positive_aspects), aggregate_scores(negative_aspects)

# Response Models for AspectTag and CommentAnalysisResponse
class AspectTag(BaseModel):
    """Aspect tag model for sentiment aspects."""
    aspect: str = Field(..., description="The aspect/topic")
    score: float = Field(..., description="Score for this aspect")


class CommentAnalysisResponse(BaseModel):
    """Comment analysis response model."""
    sentimentScore: float = Field(..., description="Overall sentiment score")
    aspectScore: float = Field(..., description="Aspect-based score")
    effectivenessScore: float = Field(..., description="Effectiveness score")
    positiveAspects: List[AspectTag] = Field(default_factory=list, description="List of positive aspects")
    negativeAspects: List[AspectTag] = Field(default_factory=list, description="List of negative aspects")
    commentAnalysis: List['PredictionResponse'] = Field(default_factory=list, description="Per-comment sentiment analysis")

# Initialize FastAPI app
app = FastAPI(
    title="Hybrid Fusion Sentiment Analysis API",
    description="API for sentiment analysis using Hybrid Fusion (VADER + SVM) model",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class TextInput(BaseModel):
    """Single text input model."""
    text: str = Field(..., description="Text string to analyze", min_length=1)


class BatchTextInput(BaseModel):
    """Batch text input model."""
    texts: List[str] = Field(..., description="List of text strings to analyze", min_items=1)


class WordContribution(BaseModel):
    """Word-level sentiment contribution."""
    word: str
    score: float
    sentiment: str
    impact: str


class FeatureContribution(BaseModel):
    """SVM feature contribution."""
    feature: str
    tfidf_weight: float
    contributions: Dict[str, float]
    supports: str
    impact: float


class VaderExplanation(BaseModel):
    """VADER model explanation."""
    compound_score: float
    component_scores: Dict[str, float]
    word_contributions: List[WordContribution] = []
    positive_words: List[WordContribution] = []
    negative_words: List[WordContribution] = []
    total_sentiment_words: int = 0


class SvmExplanation(BaseModel):
    """SVM model explanation."""
    class_probabilities: Dict[str, float]
    predicted_class: str
    top_features: List[FeatureContribution] = []
    total_features_used: int = 0


class FusionExplanation(BaseModel):
    """Fusion decision explanation."""
    models_agree: bool
    vader_prediction: str
    svm_prediction: str
    final_prediction: str
    svm_weight: float
    vader_weight: float
    svm_contribution: float
    vader_contribution: float
    dominant_model: str
    fusion_reason: str


class ExplanationDetails(BaseModel):
    """Complete XAI explanation."""
    summary: str
    vader: Optional[VaderExplanation] = None
    svm: Optional[SvmExplanation] = None
    fusion: Optional[FusionExplanation] = None


class PredictionResponse(BaseModel):
    """Single prediction response model."""
    text: str
    prediction: str
    confidence: float
    probabilities: dict
    vader_score: float
    svm_confidence: float
    vader_confidence: float
    fusion_weights: dict


class PredictionWithExplanation(BaseModel):
    """Prediction response with XAI explanation."""
    text: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    vader_score: float
    svm_confidence: float
    vader_confidence: float
    fusion_weights: Dict[str, float]
    explanation: ExplanationDetails


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""
    predictions: List[PredictionResponse]
    total: int


class BatchPredictionWithExplanation(BaseModel):
    """Batch prediction response with XAI explanations."""
    predictions: List[PredictionWithExplanation]
    total: int


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    message: Optional[str] = None
    model_loaded: Optional[bool] = None
    fusion_weights: Optional[dict] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    try:
        # Try to load the model
        model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
        predictor = get_predictor(model_path)
        print(f"✅ Model loaded successfully on startup from {model_path}")
    except Exception as e:
        print(f"⚠️  Warning: Could not load model on startup: {e}")
        print(f"   Model path: {os.getenv('MODEL_PATH', DEFAULT_MODEL_PATH)}")
        print("   Model will be loaded on first prediction request")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    return {
        "status": "healthy",
        "message": "Hybrid Fusion Sentiment Analysis API is running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns model status, loaded state, and fusion weights.

    **Example Response:**
    ```json
    {
        "status": "healthy",
        "model_loaded": true,
        "fusion_weights": {
            "svm": 0.9,
            "vader": 0.1
        }
    }
    ```
    """
    try:
        predictor = get_predictor()
        return {
            "status": "healthy",
            "model_loaded": True,
            "fusion_weights": {
                "svm": float(predictor.fusion_weight),
                "vader": float(1 - predictor.fusion_weight)
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "message": f"Model not available: {str(e)}"
        }


@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(input_data: TextInput):
    """
    Predict sentiment for a single text string.
    
    **Example Request:**
    ```json
    {
        "text": "This video is amazing! Very helpful tutorial."
    }
    ```
    
    **Example Response:**
    ```json
    {
        "text": "This video is amazing! Very helpful tutorial.",
        "prediction": "Positive",
        "confidence": 0.95,
        "probabilities": {
            "Negative": 0.02,
            "Neutral": 0.03,
            "Positive": 0.95
        },
        "vader_score": 0.75,
        "svm_confidence": 0.92,
        "vader_confidence": 1.0,
        "fusion_weights": {
            "svm": 0.9,
            "vader": 0.1
        }
    }
    ```
    """
    try:
        predictor = get_predictor()
        result = predictor.predict_single(input_data.text)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_sentiment_batch(input_data: BatchTextInput):
    """
    Predict sentiment for multiple text strings.
    
    **Example Request:**
    ```json
    {
        "texts": [
            "This video is amazing!",
            "I don't understand this at all.",
            "It's okay, nothing special."
        ]
    }
    ```
    
    **Example Response:**
    ```json
    {
        "predictions": [
            {
                "text": "This video is amazing!",
                "prediction": "Positive",
                "confidence": 0.95,
                ...
            },
            ...
        ],
        "total": 3
    }
    ```
    """
    try:
        predictor = get_predictor()
        results = predictor.predict_batch(input_data.texts)
        return BatchPredictionResponse(
            predictions=[PredictionResponse(**r) for r in results],
            total=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


# =============================================================================
# XAI (EXPLAINABLE AI) ENDPOINTS
# =============================================================================

@app.post("/predict/explain", response_model=PredictionWithExplanation)
async def predict_with_explanation(input_data: TextInput):
    """
    Predict sentiment for a single text with detailed XAI explanation.

    Returns word-level contributions, model agreement analysis, and human-readable summary.

    **Example Request:**
    ```json
    {
        "text": "This video is amazing! Very helpful and clear explanation."
    }
    ```

    **Example Response includes:**
    - `explanation.summary`: Human-readable explanation
    - `explanation.vader`: Word-level sentiment analysis
    - `explanation.svm`: Feature importance from SVM
    - `explanation.fusion`: How the models were combined
    """
    try:
        predictor = get_predictor()
        result = predictor.predict_with_explanation(input_data.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch/explain", response_model=BatchPredictionWithExplanation)
async def predict_batch_with_explanation(input_data: BatchTextInput):
    """
    Predict sentiment for multiple texts with detailed XAI explanations.

    **Example Request:**
    ```json
    {
        "texts": [
            "This video is amazing!",
            "I don't understand this at all."
        ]
    }
    ```
    """
    try:
        predictor = get_predictor()
        results = predictor.predict_batch_with_explanation(input_data.texts)
        return BatchPredictionWithExplanation(
            predictions=results,
            total=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.post("/predict/array", response_model=CommentAnalysisResponse)
async def predict_sentiment_array(texts: List[str]):
    """
    Alternative endpoint that accepts a simple array of strings and returns aggregated comment analysis
    with per-comment predictions.

    **Example Request:**
    ```json
    ["This video is amazing!", "I don't understand this at all."]
    ```

    **Example Response:**
    ```json
    {
        "sentimentScore": 0.78,
        "aspectScore": 0.54,
        "effectivenessScore": 0.90,
        "positiveAspects": [
            {"aspect": "clarity", "score": 0.95},
            {"aspect": "content", "score": 0.90}
        ],
        "negativeAspects": [
            {"aspect": "complexity", "score": 0.85}
        ],
        "commentAnalysis": [
            {
                "text": "This video is amazing!",
                "prediction": "Positive",
                "confidence": 0.95,
                "probabilities": {"Negative": 0.02, "Neutral": 0.03, "Positive": 0.95},
                "vader_score": 0.75,
                "svm_confidence": 0.92,
                "vader_confidence": 1.0,
                "fusion_weights": {"svm": 0.9, "vader": 0.1}
            },
            {
                "text": "I don't understand this at all.",
                "prediction": "Negative",
                "confidence": 0.88,
                "probabilities": {"Negative": 0.88, "Neutral": 0.10, "Positive": 0.02},
                "vader_score": -0.65,
                "svm_confidence": 0.85,
                "vader_confidence": 1.0,
                "fusion_weights": {"svm": 0.9, "vader": 0.1}
            }
        ]
    }
    ```
    """
    try:
        if not texts or len(texts) == 0:
            raise HTTPException(status_code=400, detail="Texts array cannot be empty")
        
        predictor = get_predictor()
        predictions = predictor.predict_batch(texts)

        # Aggregate predictions into overall scores
        overall_sentiment = 0.0
        overall_aspect = 0.0
        overall_effectiveness = 0.0

        for pred in predictions:
            # Accumulate scores
            overall_sentiment += pred["confidence"]

            # Calculate aspect score as average of probabilities
            probs = pred["probabilities"]
            aspect_score = (probs["Positive"] + probs["Neutral"]) / 2 if probs["Neutral"] > 0 else probs["Positive"]
            overall_aspect += aspect_score

            # Calculate effectiveness score: weighted combination of aspect and sentiment
            # W_aspect = 60%, W_sentiment = 40%
            effectiveness_score = (0.6 * aspect_score) + (0.4 * pred["confidence"])
            overall_effectiveness += effectiveness_score

        # Calculate averages
        num_predictions = len(predictions)
        overall_sentiment /= num_predictions
        overall_aspect /= num_predictions
        overall_effectiveness /= num_predictions

        # Extract real aspects from comment text using keyword analysis
        pos_aspects_dict, neg_aspects_dict = extract_aspects_from_comments(texts, predictions)

        # Convert to AspectTag lists, sorted by score (top 8 each)
        positive_aspects = sorted(
            [AspectTag(aspect=k, score=v) for k, v in pos_aspects_dict.items()],
            key=lambda x: x.score,
            reverse=True
        )[:8]

        negative_aspects = sorted(
            [AspectTag(aspect=k, score=v) for k, v in neg_aspects_dict.items()],
            key=lambda x: x.score,
            reverse=True
        )[:8]

        # Fallback if no aspects were extracted
        if not positive_aspects and overall_sentiment > 0.5:
            positive_aspects = [AspectTag(aspect="overall sentiment", score=overall_sentiment)]
        if not negative_aspects and overall_sentiment < 0.5:
            negative_aspects = [AspectTag(aspect="overall sentiment", score=1 - overall_sentiment)]
        
        return CommentAnalysisResponse(
            sentimentScore=overall_sentiment,
            aspectScore=overall_aspect,
            effectivenessScore=overall_effectiveness,
            positiveAspects=positive_aspects,
            negativeAspects=negative_aspects,
            commentAnalysis=[PredictionResponse(**pred) for pred in predictions]
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n❌ ERROR in /predict/array:")
        print(f"   Number of texts: {len(texts) if texts else 0}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Set to False in production
    )

