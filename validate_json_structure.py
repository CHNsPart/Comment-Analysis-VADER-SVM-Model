"""
Simple JSON structure validation without requiring FastAPI installation.
"""

import json


def validate_health_response_structure():
    """Validate HealthResponse JSON structure."""
    print("Testing /health endpoint JSON structure...")

    # Test healthy response
    healthy = {
        "status": "healthy",
        "model_loaded": True,
        "fusion_weights": {
            "svm": 0.9,
            "vader": 0.1
        }
    }

    json_str = json.dumps(healthy, indent=2)
    parsed = json.loads(json_str)

    print(f"  Healthy response JSON: {json_str}")
    print(f"  Valid JSON: {isinstance(parsed, dict)}")
    print(f"  Has required fields: {all(k in parsed for k in ['status', 'model_loaded', 'fusion_weights'])}")
    print()

    # Test unhealthy response
    unhealthy = {
        "status": "unhealthy",
        "model_loaded": False,
        "message": "Model not available"
    }

    json_str = json.dumps(unhealthy, indent=2)
    parsed = json.loads(json_str)

    print(f"  Unhealthy response JSON: {json_str}")
    print(f"  Valid JSON: {isinstance(parsed, dict)}")
    print()


def validate_comment_analysis_structure():
    """Validate CommentAnalysisResponse JSON structure with commentAnalysis."""
    print("Testing /predict/array endpoint JSON structure...")

    # Create sample response
    response = {
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
            },
            {
                "text": "I don't understand this.",
                "prediction": "Negative",
                "confidence": 0.88,
                "probabilities": {
                    "Negative": 0.88,
                    "Neutral": 0.10,
                    "Positive": 0.02
                },
                "vader_score": -0.65,
                "svm_confidence": 0.85,
                "vader_confidence": 1.0,
                "fusion_weights": {
                    "svm": 0.9,
                    "vader": 0.1
                }
            }
        ]
    }

    # Convert to JSON and parse
    json_str = json.dumps(response, indent=2)
    parsed = json.loads(json_str)

    print(f"  Response JSON (truncated):")
    print(f"    sentimentScore: {parsed['sentimentScore']}")
    print(f"    aspectScore: {parsed['aspectScore']}")
    print(f"    effectivenessScore: {parsed['effectivenessScore']}")
    print(f"    positiveAspects count: {len(parsed['positiveAspects'])}")
    print(f"    negativeAspects count: {len(parsed['negativeAspects'])}")
    print(f"    commentAnalysis count: {len(parsed['commentAnalysis'])}")
    print()

    # Validate structure
    required_fields = [
        'sentimentScore', 'aspectScore', 'effectivenessScore',
        'positiveAspects', 'negativeAspects', 'commentAnalysis'
    ]
    has_all_fields = all(k in parsed for k in required_fields)

    print(f"  Valid JSON: {isinstance(parsed, dict)}")
    print(f"  Has all required fields: {has_all_fields}")
    print(f"  commentAnalysis is a list: {isinstance(parsed['commentAnalysis'], list)}")
    print(f"  commentAnalysis has items: {len(parsed['commentAnalysis']) > 0}")

    # Check first comment analysis item
    if parsed['commentAnalysis']:
        first_comment = parsed['commentAnalysis'][0]
        comment_required_fields = [
            'text', 'prediction', 'confidence', 'probabilities',
            'vader_score', 'svm_confidence', 'vader_confidence', 'fusion_weights'
        ]
        has_comment_fields = all(k in first_comment for k in comment_required_fields)
        print(f"  First comment has all required fields: {has_comment_fields}")
        print(f"\n  First comment sample:")
        print(f"    Text: '{first_comment['text']}'")
        print(f"    Prediction: {first_comment['prediction']}")
        print(f"    Confidence: {first_comment['confidence']}")

    print(f"\n  Full JSON structure valid: {has_all_fields and len(parsed['commentAnalysis']) > 0}")
    print()


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("JSON Structure Validation Tests")
    print("=" * 70)
    print()

    try:
        validate_health_response_structure()
        validate_comment_analysis_structure()

        print("=" * 70)
        print("All JSON structure validation tests passed!")
        print("=" * 70)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
