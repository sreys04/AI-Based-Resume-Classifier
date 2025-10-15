"""
Resume Classification System using Machine Learning
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

class ResumeClassifier:
    """AI-powered resume classifier"""
    
    def __init__(self, model_path=None, vectorizer_path=None):
        """Initialize the classifier"""
        self.model = None
        self.vectorizer = None
        self.categories = [
            'Software Engineer',
            'Data Scientist', 
            'Product Manager',
            'Marketing Manager',
            'Sales Representative',
            'HR Manager',
            'Financial Analyst',
            'Operations Manager',
            'Designer',
            'Other'
        ]
        self.model_version = "1.0"
        
        # Initialize NLTK components
        self._download_nltk_dependencies()
        self.lemmatizer = WordNetLemmatizer()
        
        # Load existing model if paths provided
        if model_path and vectorizer_path:
            self.load_model(model_path, vectorizer_path)
        else:
            self._create_default_model()
    
    def _download_nltk_dependencies(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
    
    def preprocess_text(self, text):
        """Preprocess resume text for classification"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, emails, and phone numbers
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S*@\S*\s?', '', text)
        text = re.sub(r'\d{3}-\d{3}-\d{4}|\(\d{3}\)\s*\d{3}-\d{4}', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        stop_words = set(stopwords.words('english'))
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def _create_default_model(self):
        """Create a default model with sample data"""
        # Sample training data for demonstration
        sample_data = {
            'resume_text': [
                "python java programming software development machine learning algorithms data structures",
                "data analysis pandas numpy matplotlib statistics machine learning python sql database",
                "product management roadmap stakeholder communication agile scrum user stories market research",
                "marketing strategy digital marketing seo social media content marketing brand management",
                "sales revenue territory management crm customer relationship lead generation b2b sales",
                "human resources recruitment talent acquisition employee relations policy hr management",
                "financial analysis excel modeling budgeting forecasting accounting finance investment",
                "operations management supply chain logistics process improvement lean six sigma",
                "ui ux design graphic design adobe photoshop illustrator figma user experience",
                "javascript react nodejs html css frontend development web development responsive design"
            ],
            'category': [
                'Software Engineer', 'Data Scientist', 'Product Manager', 'Marketing Manager',
                'Sales Representative', 'HR Manager', 'Financial Analyst', 'Operations Manager',
                'Designer', 'Software Engineer'
            ]
        }
        
        df = pd.DataFrame(sample_data)
        
        # Preprocess text
        df['processed_text'] = df['resume_text'].apply(self.preprocess_text)
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=1,
            max_df=0.95
        )
        
        # Fit vectorizer and transform data
        X = self.vectorizer.fit_transform(df['processed_text'])
        y = df['category']
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X, y)
    
    def train_model(self, training_data_path):
        """Train the model with provided training data"""
        try:
            # Load training data
            df = pd.read_csv(training_data_path)
            
            # Ensure required columns exist
            if 'resume_text' not in df.columns or 'category' not in df.columns:
                raise ValueError("Training data must contain 'resume_text' and 'category' columns")
            
            # Preprocess text
            df['processed_text'] = df['resume_text'].apply(self.preprocess_text)
            
            # Create TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
            
            # Fit vectorizer and transform data
            X = self.vectorizer.fit_transform(df['processed_text'])
            y = df['category']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train multiple models and select best
            models = {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
                'LogisticRegression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
                'MultinomialNB': MultinomialNB(),
                'SVM': SVC(random_state=42, class_weight='balanced', probability=True)
            }
            
            best_score = 0
            best_model = None
            best_name = ""
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                
                print(f"{name} Accuracy: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_name = name
            
            self.model = best_model
            print(f"Best model: {best_name} with accuracy: {best_score:.4f}")
            
            return best_score
            
        except Exception as e:
            print(f"Training failed: {e}")
            return None
    
    def classify(self, resume_text):
        """Classify a resume"""
        if not self.model or not self.vectorizer:
            raise ValueError("Model not trained or loaded")
        
        # Preprocess text
        processed_text = self.preprocess_text(resume_text)
        
        if not processed_text:
            return {
                'predicted_category': 'Other',
                'confidence_score': 0.1,
                'all_scores': {'Other': 0.1},
                'features_info': {'processed_text_length': 0}
            }
        
        # Vectorize
        text_vector = self.vectorizer.transform([processed_text])
        
        # Get prediction probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(text_vector)[0]
            classes = self.model.classes_
            
            # Create scores dictionary
            all_scores = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
            
            # Get top prediction
            top_class_idx = np.argmax(probabilities)
            predicted_category = classes[top_class_idx]
            confidence_score = float(probabilities[top_class_idx])
        else:
            # For models without predict_proba
            predicted_category = self.model.predict(text_vector)[0]
            confidence_score = 0.8  # Default confidence
            all_scores = {predicted_category: confidence_score}
        
        return {
            'predicted_category': predicted_category,
            'confidence_score': confidence_score,
            'all_scores': all_scores,
            'features_info': {
                'processed_text_length': len(processed_text),
                'feature_vector_size': text_vector.shape[1]
            }
        }
    
    def save_model(self, model_path, vectorizer_path):
        """Save trained model and vectorizer"""
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
            
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            print(f"Model saved to {model_path}")
            print(f"Vectorizer saved to {vectorizer_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_path, vectorizer_path):
        """Load trained model and vectorizer"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            print(f"Model loaded from {model_path}")
            print(f"Vectorizer loaded from {vectorizer_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_model_version(self):
        """Get model version"""
        return self.model_version
    
    def get_feature_importance(self, top_n=20):
        """Get top feature importance if model supports it"""
        if hasattr(self.model, 'feature_importances_') and self.vectorizer:
            feature_names = self.vectorizer.get_feature_names_out()
            importance_scores = self.model.feature_importances_
            
            # Create feature importance pairs
            feature_importance = list(zip(feature_names, importance_scores))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            return feature_importance[:top_n]
        else:
            return None