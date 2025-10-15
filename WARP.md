# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Development Commands

### Local Development
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "from app import create_app; app = create_app(); app.app_context().push(); from backend.database import init_db; init_db()"

# Run development server
python app.py
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=backend --cov-report=html

# Run specific test file
pytest tests/test_api.py -v

# API testing with curl
curl -X POST -F "file=@sample_resume.pdf" http://localhost:5000/api/upload
curl -X POST http://localhost:5000/api/classify/1
curl http://localhost:5000/api/results/1
```

### Code Quality
```bash
# Install development tools
pip install black flake8 isort

# Format code
black .
isort .

# Check code style
flake8 .
```

### Docker Development
```bash
# Build and run locally
docker build -t resume-classifier-ai .
docker run -p 5000:5000 resume-classifier-ai

# Using deployment script
python deploy.py docker
```

## Architecture Overview

### Core Components

**Flask Application Factory** (`app.py`)
- Uses application factory pattern with `create_app()`
- Registers blueprints, initializes extensions, and handles error routes
- Main entry point serves at `0.0.0.0:5000` for containerized deployment

**Machine Learning Pipeline** (`backend/classifier.py`)
- `ResumeClassifier` class handles text preprocessing and classification
- Supports multiple ML algorithms: Random Forest (default), Logistic Regression, Naive Bayes, SVM
- Uses TF-IDF vectorization with n-gram features (1,2)
- NLTK integration for text preprocessing (tokenization, stopwords, lemmatization)

**File Processing System** (`backend/file_processor.py`)
- Multi-format support: PDF (pdfplumber + PyPDF2 fallback), DOCX, TXT
- Text extraction with fallback mechanisms for robust document parsing
- Content validation ensures minimum 50 characters extracted

**Database Layer** (`backend/database.py`)
- SQLAlchemy models: `Resume` (main records) and `ClassificationLog` (analytics)
- Supports SQLite (development) and PostgreSQL (production)
- Tracks processing status, confidence scores, and classification results

**RESTful API** (`backend/api.py`)
- Blueprint-based organization with `/api` prefix
- Key endpoints: upload, classify, results, stats, categories
- Proper error handling and status tracking throughout processing pipeline

### Data Flow

1. **File Upload** → Secure filename generation → Database record creation
2. **Text Extraction** → Format-specific processors → Content validation
3. **ML Classification** → Text preprocessing → TF-IDF vectorization → Model prediction
4. **Result Storage** → Database update with scores → Classification logging

### Configuration Management

**Environment-based Config** (`backend/config.py`)
- Separate classes for Development, Production, Testing
- Centralized category definitions in `RESUME_CATEGORIES`
- File upload constraints and ML model paths

**Deployment Support**
- Multi-platform deployment script (`deploy.py`) with Docker, Heroku support
- Procfile for Heroku, railway.json for Railway
- Health checks and proper WSGI configuration

### Key Design Patterns

- **Repository Pattern**: Database operations abstracted through SQLAlchemy models
- **Factory Pattern**: Application creation and ML model initialization
- **Strategy Pattern**: Multiple ML algorithms with consistent interface
- **Blueprint Pattern**: Modular API organization

### Performance Considerations

- **File Processing**: Async-ready design for background task integration
- **ML Inference**: Model caching and reuse across requests
- **Database**: Proper indexing on frequently queried fields (status, category)
- **Memory Management**: Cleanup of uploaded files after processing

### Security Features

- File type validation and secure filename generation
- Input sanitization in text preprocessing
- Environment variable protection for secrets
- CSRF protection via Flask configuration
- File size limits (16MB default)

## Development Notes

### Adding New Job Categories
Update `RESUME_CATEGORIES` list in `backend/config.py` and retrain model if needed.

### Model Training
Use custom dataset by calling `classifier.train_model('path/to/dataset.csv')` with columns: `resume_text`, `category`.

### Database Migrations
For production, implement Flask-Migrate for schema changes:
```bash
flask db init
flask db migrate -m "Description"
flask db upgrade
```

### Testing Strategy
- Unit tests for API endpoints (`test_api.py`)
- Integration tests needed for file processing and ML pipeline
- Mock external dependencies (file system, ML models) in tests