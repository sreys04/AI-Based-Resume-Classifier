"""
Configuration settings for the Resume Classifier AI application
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration class"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///resume_classifier.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = 'data/uploads'
    ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}
    
    # ML Model settings
    MODEL_PATH = 'models/resume_classifier.pkl'
    VECTORIZER_PATH = 'models/vectorizer.pkl'
    
    # Classification categories
    RESUME_CATEGORIES = [
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

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    ENV = 'development'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    ENV = 'production'
    SECRET_KEY = os.environ.get('SECRET_KEY')

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'