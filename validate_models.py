"""
Validation script to test Pydantic models and JSON serialization.
"""

from pydantic import ValidationError
from main import (
    AspectTag,
    CommentAnalysisResponse,
    PredictionResponse,
    HealthResponse
)
import json


def validate_health_response():
    """Validate HealthResponse model."""
    print("Testing HealthResponse model...")

    # Test healthy response
    healthy = HealthResponse(
        status="healthy",
        model_loaded=True,
        fusion_weights={"svm": 0.9, "vader": 0.1}
    )
    print(f"  Healthy: {json.dumps(healthy.model_dump(), indent=2)}")

    # Test unhealthy response
    unhealthy = HealthResponse(
        status="unhealthy",
        model_loaded=False,
        message="Model not available"
    )
    print(f"  Unhealthy: {json.dumps(unhealthy.model_dump(), indent=2)}")
    print()


def validate_prediction_response():
    """Validate PredictionResponse model."""
    print("Testing PredictionResponse model...")

    pred = PredictionResponse(
        text="This is a test",
        prediction="Positive",
        confidence=0.95,
        probabilities={"Negative": 0.02, "Neutral": 0.03, "Positive": 0.95},
        vader_score=0.75,
        svm_confidence=0.92,
        vader_confidence=1.0,
        fusion_weights={"svm": 0.9, "vader": 0.1}
    )
    print(f"  {json.dumps(pred.model_dump(), indent=2)}")
    print()


def validate_comment_analysis_response():
    """Validate CommentAnalysisResponse model with commentAnalysis field."""
    print("Testing CommentAnalysisResponse model...")

    # Create sample predictions
    predictions = [
        PredictionResponse(
            text="This video is amazing!",
            prediction="Positive",
            confidence=0.95,
            probabilities={"Negative": 0.02, "Neutral": 0.03, "Positive": 0.95},
            vader_score=0.75,
            svm_confidence=0.92,
            vader_confidence=1.0,
            fusion_weights={"svm": 0.9, "vader": 0.1}
        ),
        PredictionResponse(
            text="I don't understand this.",
            prediction="Negative",
            confidence=0.88,
            probabilities={"Negative": 0.88, "Neutral": 0.10, "Positive": 0.02},
            vader_score=-0.65,
            svm_confidence=0.85,
            vader_confidence=1.0,
            fusion_weights={"svm": 0.9, "vader": 0.1}
        )
    ]

    # Create comment analysis response
    response = CommentAnalysisResponse(
        sentimentScore=0.78,
        aspectScore=0.54,
        effectivenessScore=0.90,
        positiveAspects=[
            AspectTag(aspect="clarity", score=0.95),
            AspectTag(aspect="content", score=0.90)
        ],
        negativeAspects=[
            AspectTag(aspect="complexity", score=0.85)
        ],
        commentAnalysis=predictions
    )

    # Convert to dict and JSON
    response_dict = response.model_dump()
    json_str = json.dumps(response_dict, indent=2)

    print(f"  Response has {len(response_dict['commentAnalysis'])} comment analyses")
    print(f"  JSON length: {len(json_str)} characters")
    print(f"\n  Sample JSON:\n{json_str}")
    print()

    # Verify JSON is valid
    parsed = json.loads(json_str)
    print(f"  JSON is valid: {isinstance(parsed, dict)}")
    print(f"  Has commentAnalysis field: {'commentAnalysis' in parsed}")
    print(f"  Number of comments: {len(parsed.get('commentAnalysis', []))}")
    print()


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("Pydantic Model Validation Tests")
    print("=" * 70)
    print()

    try:
        validate_health_response()
        validate_prediction_response()
        validate_comment_analysis_response()

        print("=" * 70)
        print("All validation tests passed!")
        print("=" * 70)

    except ValidationError as e:
        print(f"Validation Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
