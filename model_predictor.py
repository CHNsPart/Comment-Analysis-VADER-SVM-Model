"""
Model Predictor Module
Handles loading and prediction using the Hybrid Fusion VADER + SVM model
With Explainable AI (XAI) capabilities for transparent predictions
"""

import pickle
import os
import re
import numpy as np
from typing import List, Union, Dict, Any, Tuple
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import LabelEncoder

# Default model path (can be overridden by environment variable)
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "outputs/models/hybrid_fusion_model.pkl")


# =============================================================================
# XAI HELPER FUNCTIONS
# =============================================================================

def tokenize_simple(text: str) -> List[str]:
    """Simple tokenization for word-level analysis."""
    # Remove punctuation and split
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return [w for w in text.split() if len(w) > 1]


def get_sentiment_label(score: float) -> str:
    """Convert numeric score to sentiment label."""
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    return "neutral"


class HybridFusionPredictor:
    """Wrapper class for the Hybrid Fusion model prediction pipeline."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the predictor by loading the saved model artifacts.
        
        Args:
            model_path: Path to the saved model pickle file (defaults to MODEL_PATH env var or default path)
        """
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                "Please ensure the model has been trained and saved first."
            )
        
        print(f"Loading model from {model_path}...")
        with open(model_path, 'rb') as f:
            self.artifacts = pickle.load(f)
        
        # Load components
        self.tfidf_vectorizer = self.artifacts['tfidf_vectorizer']
        self.svm_model = self.artifacts.get('svm_calibrated', self.artifacts['svm_model'])
        self.label_encoder = self.artifacts.get('label_encoder')
        self.config = self.artifacts.get('config', {})
        self.fusion_weight = self.config.get('fusion_weight', 0.9)
        
        # Initialize VADER
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # VADER thresholds from config
        self.vader_threshold_pos = 0.05
        self.vader_threshold_neg = -0.05
        
        # Initialize label encoder if not present
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(['Negative', 'Neutral', 'Positive'])
        
        print("✅ Model loaded successfully!")
        print(f"   Fusion weight (SVM): {self.fusion_weight:.2f}")
        print(f"   Fusion weight (VADER): {1 - self.fusion_weight:.2f}")

        # Cache feature names for XAI
        self._feature_names = self.tfidf_vectorizer.get_feature_names_out()

    # =========================================================================
    # XAI METHODS - Explainable AI for transparent predictions
    # =========================================================================

    def _explain_vader(self, text: str) -> Dict[str, Any]:
        """
        Get word-level VADER sentiment explanation.

        Returns:
            Dictionary with word scores, key phrases, and overall breakdown
        """
        # Get detailed VADER scores
        scores = self.vader_analyzer.polarity_scores(text)

        # VADER's lexicon for word-level analysis
        lexicon = self.vader_analyzer.lexicon

        # Tokenize and find sentiment-bearing words
        words = tokenize_simple(text)
        word_sentiments = []

        for word in words:
            if word in lexicon:
                score = lexicon[word]
                word_sentiments.append({
                    "word": word,
                    "score": round(score, 3),
                    "sentiment": get_sentiment_label(score),
                    "impact": "high" if abs(score) > 1.5 else "medium" if abs(score) > 0.5 else "low"
                })

        # Sort by absolute impact
        word_sentiments.sort(key=lambda x: abs(x["score"]), reverse=True)

        # Identify key positive and negative words
        positive_words = [w for w in word_sentiments if w["score"] > 0][:5]
        negative_words = [w for w in word_sentiments if w["score"] < 0][:5]

        return {
            "compound_score": round(scores["compound"], 3),
            "component_scores": {
                "positive": round(scores["pos"], 3),
                "negative": round(scores["neg"], 3),
                "neutral": round(scores["neu"], 3)
            },
            "word_contributions": word_sentiments[:10],  # Top 10 sentiment words
            "positive_words": positive_words,
            "negative_words": negative_words,
            "total_sentiment_words": len(word_sentiments)
        }

    def _explain_svm(self, text: str, top_n: int = 10) -> Dict[str, Any]:
        """
        Get SVM feature importance explanation.

        Returns:
            Dictionary with top contributing features for each class
        """
        # Transform text to TF-IDF
        text_tfidf = self.tfidf_vectorizer.transform([text])
        tfidf_array = text_tfidf.toarray()[0]

        # Get non-zero features
        non_zero_indices = np.where(tfidf_array > 0)[0]

        # Get SVM coefficients (for linear SVM) or use decision function
        try:
            # For CalibratedClassifierCV, access base estimator
            if hasattr(self.svm_model, 'calibrated_classifiers_'):
                base_svm = self.svm_model.calibrated_classifiers_[0].estimator
                if hasattr(base_svm, 'coef_'):
                    coef = base_svm.coef_
                else:
                    coef = None
            elif hasattr(self.svm_model, 'coef_'):
                coef = self.svm_model.coef_
            else:
                coef = None
        except:
            coef = None

        feature_contributions = []

        if coef is not None:
            # Calculate feature contributions
            for idx in non_zero_indices:
                feature_name = self._feature_names[idx]
                tfidf_value = tfidf_array[idx]

                # Get contribution to each class
                contributions = {}
                for class_idx, class_name in enumerate(["Negative", "Neutral", "Positive"]):
                    if class_idx < coef.shape[0]:
                        contrib = tfidf_value * coef[class_idx, idx]
                        contributions[class_name] = round(float(contrib), 4)

                # Determine which class this feature supports most
                max_class = max(contributions, key=lambda k: contributions[k])
                max_contrib = contributions[max_class]

                feature_contributions.append({
                    "feature": feature_name,
                    "tfidf_weight": round(float(tfidf_value), 4),
                    "contributions": contributions,
                    "supports": max_class if max_contrib > 0 else f"against {max_class}",
                    "impact": abs(max_contrib)
                })

            # Sort by impact
            feature_contributions.sort(key=lambda x: x["impact"], reverse=True)

        # Get probabilities
        proba = self.svm_model.predict_proba(text_tfidf)[0]

        return {
            "class_probabilities": {
                "Negative": round(float(proba[0]), 4),
                "Neutral": round(float(proba[1]), 4),
                "Positive": round(float(proba[2]), 4)
            },
            "predicted_class": self.label_encoder.classes_[np.argmax(proba)],
            "top_features": feature_contributions[:top_n],
            "total_features_used": len(non_zero_indices)
        }

    def _explain_fusion(
        self,
        vader_score: float,
        vader_proba: np.ndarray,
        svm_proba: np.ndarray,
        fused_proba: np.ndarray
    ) -> Dict[str, Any]:
        """
        Explain how VADER and SVM predictions were fused.
        """
        vader_pred = self.label_encoder.classes_[np.argmax(vader_proba)]
        svm_pred = self.label_encoder.classes_[np.argmax(svm_proba)]
        final_pred = self.label_encoder.classes_[np.argmax(fused_proba)]

        models_agree = vader_pred == svm_pred

        # Calculate how much each model contributed
        svm_contribution = self.fusion_weight * np.max(svm_proba)
        vader_contribution = (1 - self.fusion_weight) * np.max(vader_proba)

        # Determine which model was more influential for final decision
        if svm_pred == final_pred and vader_pred != final_pred:
            dominant_model = "SVM"
            reason = f"SVM prediction ({svm_pred}) overrode VADER ({vader_pred}) due to higher weight"
        elif vader_pred == final_pred and svm_pred != final_pred:
            dominant_model = "VADER"
            reason = f"VADER sentiment ({vader_pred}) was strong enough to influence despite lower weight"
        elif models_agree:
            dominant_model = "Both"
            reason = f"Both models agreed on {final_pred}, reinforcing confidence"
        else:
            dominant_model = "SVM"
            reason = f"SVM's higher weight (90%) drove the final {final_pred} prediction"

        return {
            "models_agree": models_agree,
            "vader_prediction": vader_pred,
            "svm_prediction": svm_pred,
            "final_prediction": final_pred,
            "svm_weight": self.fusion_weight,
            "vader_weight": 1 - self.fusion_weight,
            "svm_contribution": round(float(svm_contribution), 4),
            "vader_contribution": round(float(vader_contribution), 4),
            "dominant_model": dominant_model,
            "fusion_reason": reason
        }

    def _generate_summary(
        self,
        text: str,
        prediction: str,
        confidence: float,
        vader_explanation: Dict,
        svm_explanation: Dict,
        fusion_explanation: Dict
    ) -> str:
        """
        Generate a human-readable summary of the prediction.
        """
        summary_parts = []

        # Main prediction statement
        conf_level = "high" if confidence > 0.8 else "moderate" if confidence > 0.6 else "low"
        summary_parts.append(
            f"This text is classified as **{prediction}** with {conf_level} confidence ({confidence:.0%})."
        )

        # Model agreement
        if fusion_explanation["models_agree"]:
            summary_parts.append(
                f"Both VADER and SVM models agree on this classification."
            )
        else:
            summary_parts.append(
                f"VADER predicted {fusion_explanation['vader_prediction']} while SVM predicted {fusion_explanation['svm_prediction']}. "
                f"{fusion_explanation['fusion_reason']}."
            )

        # Key sentiment words from VADER
        pos_words = vader_explanation.get("positive_words", [])
        neg_words = vader_explanation.get("negative_words", [])

        if pos_words:
            words = ", ".join([f"'{w['word']}'" for w in pos_words[:3]])
            summary_parts.append(f"Positive indicators: {words}.")

        if neg_words:
            words = ", ".join([f"'{w['word']}'" for w in neg_words[:3]])
            summary_parts.append(f"Negative indicators: {words}.")

        # Top SVM features
        top_features = svm_explanation.get("top_features", [])[:3]
        if top_features:
            features = ", ".join([f"'{f['feature']}'" for f in top_features])
            summary_parts.append(f"Key terms for classification: {features}.")

        return " ".join(summary_parts)

    def predict_with_explanation(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment with full XAI explanation.

        Args:
            text: Input text string

        Returns:
            Dictionary with prediction, confidence, and detailed explanations
        """
        # Preprocess
        processed_text = self._preprocess_text(text)

        if len(processed_text) < 3:
            return {
                "text": text,
                "prediction": "Neutral",
                "confidence": 0.0,
                "probabilities": {"Negative": 0.33, "Neutral": 0.34, "Positive": 0.33},
                "explanation": {
                    "summary": "Text too short for meaningful analysis.",
                    "vader": {},
                    "svm": {},
                    "fusion": {}
                }
            }

        # Get VADER analysis
        vader_score = self._get_vader_score(processed_text)
        vader_proba = self._vader_score_to_proba(vader_score)

        # Get SVM prediction
        text_tfidf = self.tfidf_vectorizer.transform([processed_text])
        svm_proba = self.svm_model.predict_proba(text_tfidf)[0]

        # Apply fusion
        fused_proba = (
            self.fusion_weight * svm_proba +
            (1 - self.fusion_weight) * vader_proba
        )

        # Get prediction
        pred_idx = np.argmax(fused_proba)
        prediction = self.label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(fused_proba[pred_idx])

        # Generate explanations
        vader_explanation = self._explain_vader(processed_text)
        svm_explanation = self._explain_svm(processed_text)
        fusion_explanation = self._explain_fusion(vader_score, vader_proba, svm_proba, fused_proba)

        # Generate human-readable summary
        summary = self._generate_summary(
            processed_text, prediction, confidence,
            vader_explanation, svm_explanation, fusion_explanation
        )

        return {
            "text": text,
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "probabilities": {
                "Negative": round(float(fused_proba[0]), 4),
                "Neutral": round(float(fused_proba[1]), 4),
                "Positive": round(float(fused_proba[2]), 4)
            },
            "vader_score": round(float(vader_score), 4),
            "svm_confidence": round(float(np.max(svm_proba)), 4),
            "vader_confidence": round(float(np.max(vader_proba)), 4),
            "fusion_weights": {
                "svm": float(self.fusion_weight),
                "vader": float(1 - self.fusion_weight)
            },
            "explanation": {
                "summary": summary,
                "vader": vader_explanation,
                "svm": svm_explanation,
                "fusion": fusion_explanation
            }
        }

    def predict_batch_with_explanation(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment with explanations for a batch of texts.
        """
        return [self.predict_with_explanation(text) for text in texts]

    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        if not isinstance(text, str):
            text = str(text)
        return text.strip()
    
    def _get_vader_score(self, text: str) -> float:
        """Get VADER compound score for a single text."""
        sentiment_dict = self.vader_analyzer.polarity_scores(text)
        return sentiment_dict['compound']
    
    def _vader_score_to_proba(self, score: float) -> np.ndarray:
        """Convert VADER score to probability distribution."""
        proba = np.zeros(3)  # [Negative, Neutral, Positive]
        
        if score >= self.vader_threshold_pos:
            proba[2] = 1.0  # Positive
        elif score <= self.vader_threshold_neg:
            proba[0] = 1.0  # Negative
        else:
            proba[1] = 1.0  # Neutral
        
        return proba
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment for a single text string.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with prediction, probabilities, and confidence scores
        """
        # Preprocess
        processed_text = self._preprocess_text(text)
        
        if len(processed_text) < 3:
            return {
                "text": text,
                "prediction": "Neutral",
                "confidence": 0.0,
                "probabilities": {
                    "Negative": 0.33,
                    "Neutral": 0.34,
                    "Positive": 0.33
                },
                "vader_score": 0.0,
                "svm_confidence": 0.0,
                "vader_confidence": 0.0
            }
        
        # Get VADER score
        vader_score = self._get_vader_score(processed_text)
        vader_proba = self._vader_score_to_proba(vader_score)
        
        # Get SVM prediction
        text_tfidf = self.tfidf_vectorizer.transform([processed_text])
        svm_proba = self.svm_model.predict_proba(text_tfidf)[0]
        
        # Apply fusion
        fused_proba = (
            self.fusion_weight * svm_proba + 
            (1 - self.fusion_weight) * vader_proba
        )
        
        # Get prediction
        pred_idx = np.argmax(fused_proba)
        prediction = self.label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(fused_proba[pred_idx])
        
        # Format probabilities
        proba_dict = {
            "Negative": float(fused_proba[0]),
            "Neutral": float(fused_proba[1]),
            "Positive": float(fused_proba[2])
        }
        
        return {
            "text": text,
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": proba_dict,
            "vader_score": float(vader_score),
            "svm_confidence": float(np.max(svm_proba)),
            "vader_confidence": float(np.max(vader_proba)),
            "fusion_weights": {
                "svm": float(self.fusion_weight),
                "vader": float(1 - self.fusion_weight)
            }
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for a batch of texts.
        
        Args:
            texts: List of input text strings
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for text in texts:
            results.append(self.predict_single(text))
        return results


# Global model instance (lazy loading)
_predictor_instance = None


def get_predictor(model_path: str = None) -> HybridFusionPredictor:
    """Get or create the global predictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        _predictor_instance = HybridFusionPredictor(model_path)
    return _predictor_instance

