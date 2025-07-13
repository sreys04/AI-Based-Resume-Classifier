import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud
import joblib
import warnings
warnings.filterwarnings('ignore')

class ResumeClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.label_encoder = LabelEncoder()
        self.stemmer = PorterStemmer()
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(random_state=42),
            'Naive Bayes': MultinomialNB()
        }
        self.best_model = None
        self.best_model_name = None
        
    def download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading NLTK data...")
            nltk.download('punkt')
            nltk.download('stopwords')
    
    def generate_sample_data(self, num_samples=1000):
        """Generate sample resume data for demonstration"""
        print("Generating sample resume data...")
        
        # Sample resume templates for different categories
        categories = {
            'Data Science': [
                'Python programming machine learning data analysis pandas numpy scikit-learn tensorflow',
                'Statistical analysis data visualization matplotlib seaborn plotly SQL database',
                'Deep learning neural networks computer vision natural language processing',
                'Big data analytics hadoop spark data mining predictive modeling',
                'Data engineer ETL pipeline data warehouse business intelligence tableau'
            ],
            'Software Engineering': [
                'Software development programming languages java python javascript web development',
                'Full stack development react nodejs angular database mysql postgresql',
                'Mobile app development android ios swift kotlin flutter',
                'DevOps CI/CD docker kubernetes aws cloud infrastructure',
                'Backend development API design microservices system architecture'
            ],
            'Marketing': [
                'Digital marketing social media marketing content marketing SEO SEM',
                'Brand management marketing strategy campaign management analytics',
                'Email marketing automation marketing research customer segmentation',
                'Product marketing go-to-market strategy competitive analysis',
                'Growth marketing conversion optimization A/B testing performance marketing'
            ],
            'Finance': [
                'Financial analysis investment banking risk management portfolio management',
                'Corporate finance financial modeling excel VBA financial reporting',
                'Accounting auditing tax preparation financial planning wealth management',
                'Quantitative analysis derivatives trading fixed income equity research',
                'Banking operations credit analysis loan underwriting compliance'
            ],
            'Human Resources': [
                'Talent acquisition recruitment hiring employee relations performance management',
                'HR policies procedures compliance training development succession planning',
                'Compensation benefits payroll HRIS human capital management',
                'Organizational development change management culture transformation',
                'Employee engagement retention diversity inclusion workplace wellness'
            ]
        }
        
        # Generate synthetic resumes
        resumes = []
        labels = []
        
        for category, templates in categories.items():
            for _ in range(num_samples // len(categories)):
                # Create a resume by combining random templates with additional context
                resume_text = np.random.choice(templates)
                # Add some random professional experience keywords
                experience_words = ['experience', 'years', 'project', 'team', 'management', 'leadership', 
                                  'development', 'implementation', 'analysis', 'optimization', 'design']
                resume_text += ' ' + ' '.join(np.random.choice(experience_words, 3))
                
                resumes.append(resume_text)
                labels.append(category)
        
        df = pd.DataFrame({
            'Resume': resumes,
            'Category': labels
        })
        
        return df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Stem words
        tokens = [self.stemmer.stem(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def prepare_data(self, df):
        """Prepare data for training"""
        print("Cleaning and preprocessing text data...")
        
        # Clean resume text
        df['Cleaned_Resume'] = df['Resume'].apply(self.clean_text)
        
        # Remove empty resumes
        df = df[df['Cleaned_Resume'].str.len() > 0]
        
        return df
    
    def extract_features(self, df):
        """Extract features using TF-IDF"""
        print("Extracting features using TF-IDF...")
        
        # Fit and transform the text data
        X = self.vectorizer.fit_transform(df['Cleaned_Resume'])
        
        # Encode labels
        y = self.label_encoder.fit_transform(df['Category'])
        
        return X, y
    
    def train_models(self, X, y):
        """Train multiple classification models"""
        print("Training classification models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'y_test': y_test
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\nBest model: {best_model_name}")
        
        return results, (X_train, X_test, y_train, y_test)
    
    def evaluate_model(self, results):
        """Evaluate and visualize model performance"""
        print(f"\nDetailed evaluation for {self.best_model_name}:")
        
        best_result = results[self.best_model_name]
        y_test = best_result['y_test']
        y_pred = best_result['predictions']
        
        # Classification report
        target_names = self.label_encoder.classes_
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot model comparison
        self.plot_model_comparison(results)
    
    def plot_model_comparison(self, results):
        """Plot comparison of different models"""
        models = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in models]
        cv_means = [results[model]['cv_mean'] for model in models]
        cv_stds = [results[model]['cv_std'] for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Test accuracy comparison
        bars1 = ax1.bar(models, accuracies, color='skyblue', alpha=0.7)
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Cross-validation scores
        bars2 = ax2.bar(models, cv_means, yerr=cv_stds, color='lightgreen', alpha=0.7)
        ax2.set_title('Cross-Validation Scores')
        ax2.set_ylabel('CV Score')
        ax2.set_ylim(0, 1)
        for bar, mean in zip(bars2, cv_means):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_resume_category(self, resume_text):
        """Predict category for a new resume"""
        if self.best_model is None:
            raise ValueError("Model not trained yet!")
        
        # Clean the text
        cleaned_text = self.clean_text(resume_text)
        
        # Transform using fitted vectorizer
        text_vector = self.vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = self.best_model.predict(text_vector)[0]
        prediction_proba = self.best_model.predict_proba(text_vector)[0]
        
        # Get category name
        category = self.label_encoder.inverse_transform([prediction])[0]
        
        # Get probabilities for all categories
        categories = self.label_encoder.classes_
        probabilities = dict(zip(categories, prediction_proba))
        
        return category, probabilities
    
    def generate_wordcloud(self, df):
        """Generate word clouds for each category"""
        print("Generating word clouds...")
        
        categories = df['Category'].unique()
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, category in enumerate(categories):
            if i < len(axes):
                category_text = ' '.join(df[df['Category'] == category]['Cleaned_Resume'])
                
                wordcloud = WordCloud(width=400, height=300, 
                                    background_color='white',
                                    max_words=100).generate(category_text)
                
                axes[i].imshow(wordcloud, interpolation='bilinear')
                axes[i].set_title(f'{category}', fontsize=14, fontweight='bold')
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(categories), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('wordclouds.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filename='resume_classifier_model.pkl'):
        """Save the trained model"""
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'stemmer': self.stemmer
        }
        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='resume_classifier_model.pkl'):
        """Load a saved model"""
        model_data = joblib.load(filename)
        self.best_model = model_data['best_model']
        self.best_model_name = model_data['best_model_name']
        self.vectorizer = model_data['vectorizer']
        self.label_encoder = model_data['label_encoder']
        self.stemmer = model_data['stemmer']
        print(f"Model loaded from {filename}")


def main():
    """Main function to run the resume classifier"""
    print("=== AI-Based Resume Classifier ===\n")
    
    # Initialize classifier
    classifier = ResumeClassifier()
    
    # Download NLTK data
    classifier.download_nltk_data()
    
    # Generate sample data (or load from CSV if available)
    try:
        df = pd.read_csv('resume_dataset.csv')
        print("Loaded existing resume dataset")
    except FileNotFoundError:
        df = classifier.generate_sample_data(1000)
        print("Generated sample resume data")
        # Save for future use
        df.to_csv('resume_dataset.csv', index=False)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Categories: {df['Category'].unique()}")
    print(f"Category distribution:\n{df['Category'].value_counts()}\n")
    
    # Prepare data
    df = classifier.prepare_data(df)
    
    # Extract features
    X, y = classifier.extract_features(df)
    
    # Train models
    results, data_splits = classifier.train_models(X, y)
    
    # Evaluate best model
    classifier.evaluate_model(results)
    
    # Generate word clouds
    classifier.generate_wordcloud(df)
    
    # Save model
    classifier.save_model()
    
    # Test prediction with sample resume
    print("\n=== Testing Prediction ===")
    sample_resume = """
    Experienced data scientist with 5 years of experience in machine learning, 
    python programming, statistical analysis, and data visualization. 
    Skilled in pandas, numpy, scikit-learn, and tensorflow. 
    Strong background in predictive modeling and big data analytics.
    """
    
    category, probabilities = classifier.predict_resume_category(sample_resume)
    print(f"\nSample Resume: {sample_resume[:100]}...")
    print(f"Predicted Category: {category}")
    print("\nProbabilities:")
    for cat, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {prob:.3f}")


if __name__ == "__main__":
    main()
