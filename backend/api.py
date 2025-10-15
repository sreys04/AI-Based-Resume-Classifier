"""
API endpoints for Resume Classifier AI
"""
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import time
import uuid
from .database import db, Resume, ClassificationLog
from .file_processor import process_resume_file
from .classifier import ResumeClassifier
import json

api_bp = Blueprint('api', __name__)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@api_bp.route('/upload', methods=['POST'])
def upload_resume():
    """Upload and process a resume file"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Generate unique filename
        original_filename = file.filename
        file_extension = original_filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
        
        # Save file
        upload_folder = current_app.config['UPLOAD_FOLDER']
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, unique_filename)
        file.save(file_path)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Create resume record
        resume = Resume(
            filename=unique_filename,
            original_filename=original_filename,
            file_path=file_path,
            file_size=file_size,
            status='uploaded'
        )
        
        db.session.add(resume)
        db.session.commit()
        
        return jsonify({
            'message': 'File uploaded successfully',
            'resume_id': resume.id,
            'filename': original_filename
        }), 201
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/classify/<int:resume_id>', methods=['POST'])
def classify_resume(resume_id):
    """Classify an uploaded resume"""
    try:
        start_time = time.time()
        
        # Get resume record
        resume = Resume.query.get_or_404(resume_id)
        
        if resume.status == 'processing':
            return jsonify({'error': 'Resume is already being processed'}), 400
        
        # Update status
        resume.status = 'processing'
        db.session.commit()
        
        try:
            # Process file to extract text
            content_text = process_resume_file(resume.file_path)
            resume.content_text = content_text
            
            # Initialize classifier
            classifier = ResumeClassifier()
            
            # Classify resume
            prediction_result = classifier.classify(content_text)
            
            # Update resume with classification results
            resume.predicted_category = prediction_result['predicted_category']
            resume.confidence_score = prediction_result['confidence_score']
            resume.classification_scores = json.dumps(prediction_result['all_scores'])
            resume.processing_time = time.time() - start_time
            resume.status = 'completed'
            
            # Log classification
            log_entry = ClassificationLog(
                resume_id=resume.id,
                model_version=classifier.get_model_version(),
                features_used=json.dumps(prediction_result.get('features_info', {}))
            )
            
            db.session.add(log_entry)
            db.session.commit()
            
            return jsonify({
                'message': 'Resume classified successfully',
                'result': {
                    'predicted_category': resume.predicted_category,
                    'confidence_score': resume.confidence_score,
                    'all_scores': prediction_result['all_scores'],
                    'processing_time': resume.processing_time
                }
            }), 200
            
        except Exception as processing_error:
            # Update status to error
            resume.status = 'error'
            resume.processing_time = time.time() - start_time
            db.session.commit()
            raise processing_error
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/results/<int:resume_id>', methods=['GET'])
def get_resume_result(resume_id):
    """Get classification results for a specific resume"""
    try:
        resume = Resume.query.get_or_404(resume_id)
        return jsonify(resume.to_dict()), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/results', methods=['GET'])
def get_all_results():
    """Get all resume classification results"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        resumes = Resume.query.order_by(Resume.upload_timestamp.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'resumes': [resume.to_dict() for resume in resumes.items],
            'total': resumes.total,
            'pages': resumes.pages,
            'current_page': page
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/categories', methods=['GET'])
def get_categories():
    """Get available resume categories"""
    return jsonify({
        'categories': current_app.config['RESUME_CATEGORIES']
    }), 200

@api_bp.route('/stats', methods=['GET'])
def get_statistics():
    """Get classification statistics"""
    try:
        total_resumes = Resume.query.count()
        completed_resumes = Resume.query.filter_by(status='completed').count()
        
        # Category distribution
        category_stats = db.session.query(
            Resume.predicted_category, 
            db.func.count(Resume.id)
        ).filter(
            Resume.status == 'completed'
        ).group_by(Resume.predicted_category).all()
        
        category_distribution = {category: count for category, count in category_stats}
        
        return jsonify({
            'total_resumes': total_resumes,
            'completed_resumes': completed_resumes,
            'category_distribution': category_distribution
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500