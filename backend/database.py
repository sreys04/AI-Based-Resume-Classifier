"""
Database models and initialization for Resume Classifier AI
"""
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class Resume(db.Model):
    """Resume model for storing uploaded resumes and classifications"""
    __tablename__ = 'resumes'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.Integer)
    content_text = db.Column(db.Text)
    
    # Classification results
    predicted_category = db.Column(db.String(100))
    confidence_score = db.Column(db.Float)
    classification_scores = db.Column(db.Text)  # JSON string of all scores
    
    # Metadata
    upload_timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    processing_time = db.Column(db.Float)  # seconds
    status = db.Column(db.String(50), default='uploaded')  # uploaded, processing, completed, error
    
    def __repr__(self):
        return f'<Resume {self.filename}>'
    
    def to_dict(self):
        """Convert resume object to dictionary"""
        return {
            'id': self.id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'predicted_category': self.predicted_category,
            'confidence_score': self.confidence_score,
            'classification_scores': json.loads(self.classification_scores) if self.classification_scores else None,
            'upload_timestamp': self.upload_timestamp.isoformat() if self.upload_timestamp else None,
            'processing_time': self.processing_time,
            'status': self.status
        }

class ClassificationLog(db.Model):
    """Log of classification operations for analytics"""
    __tablename__ = 'classification_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    resume_id = db.Column(db.Integer, db.ForeignKey('resumes.id'), nullable=False)
    model_version = db.Column(db.String(50))
    features_used = db.Column(db.Text)  # JSON string
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    resume = db.relationship('Resume', backref=db.backref('logs', lazy=True))

def init_db():
    """Initialize the database"""
    db.create_all()
    print("Database initialized successfully!")