# Hybrid Fusion Sentiment Analysis API

FastAPI application for the Hybrid Fusion VADER + SVM sentiment analysis model.

## Features

- **Single Text Prediction**: Analyze sentiment of individual text strings
- **Batch Prediction**: Process multiple texts at once
- **Array Input Support**: Accept simple array of strings as input
- **Detailed Results**: Returns predictions with confidence scores, probabilities, and component scores

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Ensure Model is Trained

Make sure you have run the Jupyter notebook and the model has been saved to:
```
outputs/models/hybrid_fusion_model.pkl
```

If the model file doesn't exist, you'll need to:
1. Run the notebook `Hybrid_Fusion_VADER_SVM_Optimized(mine)_1765589021657.ipynb`
2. Execute all cells to train and save the model

### 3. Run the API

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## API Endpoints

### 1. Health Check

**GET** `/health`

Check if the API and model are ready.

**Response:**
```json
{
  "status": "healthy",
  "message": "Model is loaded and ready"
}
```

### 2. Single Text Prediction

**POST** `/predict`

Predict sentiment for a single text string.

**Request:**
```json
{
  "text": "This video is amazing! Very helpful tutorial."
}
```

**Response:**
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

### 3. Batch Prediction

**POST** `/predict/batch`

Predict sentiment for multiple texts.

**Request:**
```json
{
  "texts": [
    "This video is amazing!",
    "I don't understand this at all.",
    "It's okay, nothing special."
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "text": "This video is amazing!",
      "prediction": "Positive",
      "confidence": 0.95,
      ...
    },
    {
      "text": "I don't understand this at all.",
      "prediction": "Negative",
      "confidence": 0.88,
      ...
    },
    {
      "text": "It's okay, nothing special.",
      "prediction": "Neutral",
      "confidence": 0.76,
      ...
    }
  ],
  "total": 3
}
```

### 4. Array Input (Alternative)

**POST** `/predict/array`

Alternative endpoint that accepts a simple array of strings directly.

**Request:**
```json
["This video is amazing!", "I don't understand this at all."]
```

**Response:**
```json
{
  "predictions": [...],
  "total": 2
}
```

## Frontend Integration Example

### JavaScript/Fetch

```javascript
// Single prediction
async function predictSentiment(text) {
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text: text })
  });
  
  const result = await response.json();
  return result;
}

// Batch prediction
async function predictBatch(texts) {
  const response = await fetch('http://localhost:8000/predict/batch', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ texts: texts })
  });
  
  const result = await response.json();
  return result;
}

// Array input (simpler)
async function predictArray(texts) {
  const response = await fetch('http://localhost:8000/predict/array', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(texts)
  });
  
  const result = await response.json();
  return result;
}

// Usage
predictSentiment("This is great!").then(result => {
  console.log(`Prediction: ${result.prediction}`);
  console.log(`Confidence: ${result.confidence}`);
});
```

### Python Client

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "This video is amazing!"}
)
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={"texts": ["Text 1", "Text 2", "Text 3"]}
)
results = response.json()
```

## Response Fields

- **text**: Original input text
- **prediction**: Predicted sentiment ("Positive", "Neutral", or "Negative")
- **confidence**: Overall confidence score (0-1)
- **probabilities**: Probability distribution across all classes
- **vader_score**: Raw VADER compound score (-1 to 1)
- **svm_confidence**: SVM model confidence
- **vader_confidence**: VADER model confidence
- **fusion_weights**: Weights used in the hybrid fusion

## CORS Configuration

The API is configured to allow CORS from all origins by default. For production, update the `allow_origins` in `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],  # Specific domain
    ...
)
```

## Production Deployment

For production deployment, consider:

1. **Use a production ASGI server**: Use Gunicorn with Uvicorn workers
   ```bash
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

2. **Environment variables**: Use environment variables for configuration
3. **Model caching**: The model is loaded once on startup for efficiency
4. **Error handling**: Add logging and monitoring
5. **Rate limiting**: Add rate limiting to prevent abuse
6. **HTTPS**: Use HTTPS in production

## Troubleshooting

### Model Not Found Error

If you get an error about the model file not being found:
1. Ensure you've run the training notebook
2. Check that `outputs/models/hybrid_fusion_model.pkl` exists
3. Verify the path in `model_predictor.py` if your model is in a different location

### Import Errors

If you get import errors:
1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Check Python version (3.8+ recommended)

## License

Same as the original model project.

