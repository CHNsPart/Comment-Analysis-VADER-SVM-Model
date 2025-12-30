"""
Simple test script for the FastAPI sentiment analysis endpoint.
Run this after starting the API server to test the endpoints.
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"


def  test_health():
    """Test health check endpoint."""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    if 'fusion_weights' in result:
        print(f"\nModel Status: {'Loaded' if result['model_loaded'] else 'Not Loaded'}")
        if result.get('fusion_weights'):
            print(f"Fusion Weights - SVM: {result['fusion_weights']['svm']}, VADER: {result['fusion_weights']['vader']}")
    print()


def test_single_prediction():
    """Test single text prediction."""
    print("Testing /predict endpoint (single text)...")
    
    test_texts = [
        "This video is amazing! Very helpful tutorial.",
        "I don't understand this at all. Very confusing.",
        "It's okay, nothing special."
    ]
    
    for text in test_texts:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": text}
        )
        result = response.json()
        print(f"Text: {text[:50]}...")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Probabilities: {result['probabilities']}")
        print()


def test_batch_prediction():
    """Test batch prediction endpoint."""
    print("Testing /predict/batch endpoint...")
    
    texts = [
        "This video is amazing!",
        "I don't understand this at all.",
        "It's okay, nothing special.",
        "Best tutorial ever!",
        "Waste of time, terrible explanation."
    ]
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json={"texts": texts}
    )
    result = response.json()
    
    print(f"Total predictions: {result['total']}")
    print("\nResults:")
    for pred in result['predictions']:
        print(f"  '{pred['text'][:40]}...' -> {pred['prediction']} ({pred['confidence']:.2f})")
    print()


def test_array_input():
    """Test array input endpoint."""
    print("Testing /predict/array endpoint...")

    texts = [
        "This video is amazing!",
        "I don't understand this at all.",
        "Great explanation, very clear.",
        "Confusing and poorly structured.",
        "It's okay, nothing special."
    ]

    response = requests.post(
        f"{BASE_URL}/predict/array",
        json=texts
    )
    result = response.json()

    print(f"Overall Sentiment Score: {result['sentimentScore']:.2f}")
    print(f"Overall Aspect Score: {result['aspectScore']:.2f}")
    print(f"Overall Effectiveness Score: {result['effectivenessScore']:.2f}")
    print(f"\nPositive Aspects:")
    for aspect in result['positiveAspects']:
        print(f"  - {aspect['aspect']}: {aspect['score']:.2f}")
    print(f"\nNegative Aspects:")
    if result['negativeAspects']:
        for aspect in result['negativeAspects']:
            print(f"  - {aspect['aspect']}: {aspect['score']:.2f}")
    else:
        print("  (none)")

    # Test new per-comment analysis feature
    if 'commentAnalysis' in result:
        print(f"\nPer-Comment Analysis ({len(result['commentAnalysis'])} comments):")
        for i, comment in enumerate(result['commentAnalysis'], 1):
            print(f"  {i}. '{comment['text'][:40]}...'")
            print(f"     Prediction: {comment['prediction']} (confidence: {comment['confidence']:.2f})")
            print(f"     VADER: {comment['vader_score']:.2f}, SVM: {comment['svm_confidence']:.2f}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("FastAPI Sentiment Analysis API Test")
    print("=" * 60)
    print()
    
    try:
        # Test health check
        test_health()
        
        # Test single prediction
        test_single_prediction()
        
        # Test batch prediction
        test_batch_prediction()
        
        # Test array input
        test_array_input()
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API.")
        print("   Make sure the FastAPI server is running:")
        print("   python main.py")
    except Exception as e:
        print(f"❌ Error: {e}")

