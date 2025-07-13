# AI-Based Resume Classifier

An intelligent resume classification system built with Python, pandas, numpy, and scikit-learn. This tool automatically categorizes resumes into different job categories based on their content using natural language processing and machine learning techniques.

## Features

- **Text Preprocessing**: Advanced text cleaning and normalization
- **Feature Extraction**: TF-IDF vectorization with n-gram support
- **Multiple Models**: Comparison of Logistic Regression, Random Forest, SVM, and Naive Bayes
- **Visualization**: Confusion matrices, model comparison charts, and word clouds
- **Model Persistence**: Save and load trained models
- **Interactive Demo**: Command-line interface for testing predictions
- **Batch Processing**: Process multiple resumes at once

## Job Categories

The classifier can categorize resumes into:
- Data Science
- Software Engineering
- Marketing
- Finance
- Human Resources

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-resume-classifier
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data (automatically handled by the script):
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Usage

### 1. Train the Model

Run the main script to train the classifier:
```bash
python main.py
```

This will:
- Generate sample resume data (or use existing `resume_dataset.csv`)
- Train multiple classification models
- Evaluate model performance
- Generate visualizations
- Save the best model

### 2. Interactive Demo

Use the demo script for testing:
```bash
python demo.py
```

Choose from:
- Individual predictions demo
- Batch processing demo
- Interactive classification

### 3. Programmatic Usage

```python
from main import ResumeClassifier

# Initialize classifier
classifier = ResumeClassifier()

# Load pre-trained model
classifier.load_model('resume_classifier_model.pkl')

# Classify a resume
resume_text = "Data scientist with Python and machine learning experience"
category, probabilities = classifier.predict_resume_category(resume_text)

print(f"Category: {category}")
print(f"Probabilities: {probabilities}")
```

## Files Structure

- `main.py` - Main classifier implementation and training script
- `demo.py` - Interactive demo script
- `requirements.txt` - Python dependencies
- `README.md` - This file
- `resume_dataset.csv` - Generated sample data (created on first run)
- `resume_classifier_model.pkl` - Trained model (created after training)

## Model Performance

The classifier uses multiple algorithms and selects the best performing one:

- **Logistic Regression**: Fast and interpretable
- **Random Forest**: Robust ensemble method
- **Support Vector Machine**: Effective for text classification
- **Naive Bayes**: Traditional text classification approach

Performance metrics include:
- Accuracy scores
- Cross-validation results
- Confusion matrices
- Classification reports

## Customization

### Adding New Categories

1. Modify the `generate_sample_data()` method in `main.py`
2. Add new category templates
3. Retrain the model

### Using Your Own Data

Replace the sample data generation with your own CSV file:
```python
df = pd.read_csv('your_resume_data.csv')
# Ensure columns: 'Resume', 'Category'
```

### Hyperparameter Tuning

Modify the model parameters in the `ResumeClassifier.__init__()` method:
```python
self.models = {
    'Logistic Regression': LogisticRegression(C=1.0, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    # ... other models
}
```

## Dependencies

- pandas >= 2.1.4
- numpy >= 1.24.3
- scikit-learn >= 1.3.2
- nltk >= 3.8.1
- matplotlib >= 3.7.2
- seaborn >= 0.13.0
- wordcloud >= 1.9.2
- joblib >= 1.3.2

## Output Files

The script generates several output files:
- `resume_dataset.csv` - Sample dataset
- `resume_classifier_model.pkl` - Trained model
- `confusion_matrix.png` - Confusion matrix visualization
- `model_comparison.png` - Model performance comparison
- `wordclouds.png` - Word clouds for each category
- `batch_predictions.csv` - Batch processing results

## Model Workflow

1. **Data Preprocessing**
   - Remove special characters and digits
   - Convert to lowercase
   - Tokenization
   - Stop word removal
   - Stemming

2. **Feature Extraction**
   - TF-IDF vectorization
   - N-gram features (1-2 grams)
   - Maximum 5000 features

3. **Model Training**
   - Train multiple algorithms
   - 5-fold cross-validation
   - Select best performing model

4. **Evaluation**
   - Confusion matrix
   - Classification report
   - Accuracy metrics
   - Visualization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Future Enhancements

- Support for more job categories
- Deep learning models (BERT, transformers)
- Real-time web interface
- API endpoint for integration
- Advanced feature engineering
- Multi-language support
