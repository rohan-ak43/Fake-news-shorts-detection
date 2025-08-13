# app.py - Main Flask API
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import youtube_transcript_api
from youtube_transcript_api import YouTubeTranscriptApi
import requests
from urllib.parse import urlparse, parse_qs
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except:
    pass

app = Flask(__name__)
CORS(app)

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.is_trained = False
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
            words = word_tokenize(text)
            text = ' '.join([word for word in words if word not in stop_words])
        except:
            pass
            
        return text
    
    def train_model(self, dataset_path='WELFake_Dataset.csv'):
        """Train the fake news detection model"""
        try:
            # Load the dataset
            df = pd.read_csv(dataset_path)
            
            # Assuming columns: 'text' and 'label' (0=fake, 1=real)
            # Adjust column names based on your dataset
            if 'text' not in df.columns:
                # Try common column names
                text_cols = ['title', 'content', 'news', 'article']
                for col in text_cols:
                    if col in df.columns:
                        df['text'] = df[col]
                        break
            
            if 'label' not in df.columns:
                # Try common label column names
                label_cols = ['label', 'target', 'class', 'fake']
                for col in label_cols:
                    if col in df.columns:
                        df['label'] = df[col]
                        break
            
            # Preprocess text
            df['text'] = df['text'].apply(self.preprocess_text)
            
            # Remove empty texts
            df = df[df['text'].str.len() > 0]
            
            # Split data
            X = df['text']
            y = df['label']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Vectorize text
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words='english'
            )
            
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            # Train model
            self.model = LogisticRegression(random_state=42, max_iter=1000)
            self.model.fit(X_train_vec, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            
            # Save model
            with open('fake_news_model.pkl', 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'vectorizer': self.vectorizer
                }, f)
            
            print(f"Model trained successfully! Accuracy: {accuracy:.4f}")
            return accuracy
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return None
    
    def load_model(self):
        """Load pre-trained model"""
        try:
            with open('fake_news_model.pkl', 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.vectorizer = data['vectorizer']
                self.is_trained = True
            return True
        except:
            return False
    
    def predict_fake_news(self, text):
        """Predict if text is fake news"""
        if not self.is_trained:
            return None
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Vectorize
        text_vec = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.model.predict(text_vec)[0]
        confidence = self.model.predict_proba(text_vec)[0].max()
        
        return {
            'prediction': 'real' if prediction == 1 else 'fake',
            'confidence': float(confidence * 100)
        }
    
    def analyze_segments(self, sentences):
        """Analyze individual sentences for fake content"""
        fake_segments = []
        
        # Common fake news indicators
        fake_indicators = [
            {
                'patterns': [r'\b100%\b', r'\balways works\b', r'\bnever fails\b'],
                'reason': 'Absolute claims without evidence are typically unreliable'
            },
            {
                'patterns': [r'\bbig pharma\b', r'\bthey don\'t want you to know\b', r'\bsecret\b'],
                'reason': 'Conspiracy language is a common indicator of misinformation'
            },
            {
                'patterns': [r'\bmiracle cure\b', r'\binstant results\b', r'\bamazing discovery\b'],
                'reason': 'Sensational medical claims without scientific backing'
            },
            {
                'patterns': [r'\bscientists hate\b', r'\bdoctors shocked\b', r'\bbreaking\b'],
                'reason': 'Clickbait language often accompanies false information'
            }
        ]
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Check prediction for individual sentence
            pred_result = self.predict_fake_news(sentence)
            
            if pred_result and pred_result['prediction'] == 'fake' and pred_result['confidence'] > 70:
                # Find specific reason
                reason = "Content patterns suggest potential misinformation"
                
                for indicator in fake_indicators:
                    for pattern in indicator['patterns']:
                        if re.search(pattern, sentence_lower):
                            reason = indicator['reason']
                            break
                
                fake_segments.append({
                    'text': sentence,
                    'startTime': i * 5,  # Estimated timing
                    'endTime': (i + 1) * 5,
                    'reason': reason
                })
        
        return fake_segments

# Initialize detector
detector = FakeNewsDetector()

# YouTube API functions
def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    if 'youtube.com' in url:
        if 'shorts/' in url:
            return url.split('shorts/')[-1].split('?')[0]
        else:
            parsed_url = urlparse(url)
            return parse_qs(parsed_url.query)['v'][0]
    elif 'youtu.be' in url:
        return url.split('/')[-1].split('?')[0]
    else:
        return url

def get_video_transcript(video_id):
    """Get transcript from YouTube video"""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to get English transcript
        try:
            transcript = transcript_list.find_transcript(['en'])
        except:
            # Get any available transcript
            transcript = transcript_list.find_transcript(['en-US', 'en-GB'])
        
        transcript_data = transcript.fetch()
        
        # Combine all text
        full_text = ' '.join([entry['text'] for entry in transcript_data])
        
        # Get individual sentences for segment analysis
        sentences = sent_tokenize(full_text)
        
        return full_text, sentences
        
    except Exception as e:
        print(f"Error getting transcript: {str(e)}")
        return None, None

# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_trained': detector.is_trained})

@app.route('/train', methods=['POST'])
def train_model():
    """Train the model with the dataset"""
    accuracy = detector.train_model()
    if accuracy:
        return jsonify({
            'success': True,
            'accuracy': accuracy,
            'message': 'Model trained successfully'
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Failed to train model'
        }), 500

@app.route('/analyze', methods=['POST'])
def analyze_video():
    """Analyze YouTube video for fake news"""
    try:
        data = request.get_json()
        video_url = data.get('url')
        
        if not video_url:
            return jsonify({'error': 'URL is required'}), 400
        
        # Extract video ID
        video_id = extract_video_id(video_url)
        
        # Get transcript
        transcript, sentences = get_video_transcript(video_id)
        
        if not transcript:
            return jsonify({'error': 'Could not extract transcript from video'}), 400
        
        # Analyze full transcript
        overall_result = detector.predict_fake_news(transcript)
        
        if not overall_result:
            return jsonify({'error': 'Model not trained'}), 500
        
        # Analyze individual segments
        fake_segments = detector.analyze_segments(sentences)
        
        # Generate explanation and original news (mock for now)
        explanation = ""
        original_news = ""
        
        if overall_result['prediction'] == 'fake':
            explanation = "The video contains suspicious patterns commonly found in misinformation."
            if fake_segments:
                explanation += f" {len(fake_segments)} specific segments were flagged."
            original_news = "Please verify information through reliable news sources and fact-checking websites."
        else:
            explanation = "The content appears to follow reliable information patterns."
        
        return jsonify({
            'prediction': overall_result['prediction'],
            'confidence': overall_result['confidence'],
            'fakeSegments': fake_segments,
            'originalNews': original_news if overall_result['prediction'] == 'fake' else None,
            'explanation': explanation,
            'transcript': transcript
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    """Analyze text directly for fake news"""
    try:
        data = request.get_json()
        text = data.get('text')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        result = detector.predict_fake_news(text)
        
        if not result:
            return jsonify({'error': 'Model not trained'}), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Try to load existing model, otherwise train new one
    if not detector.load_model():
        print("No pre-trained model found. Training new model...")
        detector.train_model()
    
    app.run(host='0.0.0.0', port=5000, debug=True)
