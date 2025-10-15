"""
Unit tests for Resume Classifier AI API endpoints
"""

import unittest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from backend.database import db

class APITestCase(unittest.TestCase):
    """Test cases for API endpoints"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.client = self.app.test_client()
        
        with self.app.app_context():
            db.create_all()
    
    def tearDown(self):
        """Clean up after tests"""
        with self.app.app_context():
            db.session.remove()
            db.drop_all()
    
    def test_index_route(self):
        """Test the main index route"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Resume Classifier AI', response.data)
    
    def test_upload_route(self):
        """Test the upload page route"""
        response = self.client.get('/upload')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Upload Your Resume', response.data)
    
    def test_results_route(self):
        """Test the results page route"""
        response = self.client.get('/results')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Classification Results', response.data)
    
    def test_api_categories(self):
        """Test the categories API endpoint"""
        response = self.client.get('/api/categories')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('categories', data)
        self.assertIsInstance(data['categories'], list)
        self.assertIn('Software Engineer', data['categories'])
    
    def test_api_stats_empty(self):
        """Test the stats API endpoint with empty database"""
        response = self.client.get('/api/stats')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('total_resumes', data)
        self.assertEqual(data['total_resumes'], 0)
    
    def test_api_results_empty(self):
        """Test the results API endpoint with empty database"""
        response = self.client.get('/api/results')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('resumes', data)
        self.assertEqual(len(data['resumes']), 0)
        self.assertEqual(data['total'], 0)
    
    def test_upload_no_file(self):
        """Test upload endpoint without file"""
        response = self.client.post('/api/upload')
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'No file provided')
    
    def test_upload_empty_filename(self):
        """Test upload endpoint with empty filename"""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"Test resume content")
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, 'rb') as f:
                response = self.client.post('/api/upload', data={
                    'file': (f, '')  # Empty filename
                })
            
            self.assertEqual(response.status_code, 400)
            data = json.loads(response.data)
            self.assertEqual(data['error'], 'No file selected')
        finally:
            os.unlink(tmp_path)
    
    def test_upload_invalid_file_type(self):
        """Test upload endpoint with invalid file type"""
        with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as tmp:
            tmp.write(b"Not a resume")
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, 'rb') as f:
                response = self.client.post('/api/upload', data={
                    'file': (f, 'malware.exe')
                })
            
            self.assertEqual(response.status_code, 400)
            data = json.loads(response.data)
            self.assertEqual(data['error'], 'File type not allowed')
        finally:
            os.unlink(tmp_path)
    
    @patch('backend.api.process_resume_file')
    def test_upload_valid_file(self, mock_process):
        """Test upload endpoint with valid file"""
        mock_process.return_value = "Sample resume text"
        
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"Sample resume content")
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, 'rb') as f:
                response = self.client.post('/api/upload', data={
                    'file': (f, 'resume.txt')
                })
            
            self.assertEqual(response.status_code, 201)
            data = json.loads(response.data)
            self.assertIn('resume_id', data)
            self.assertEqual(data['filename'], 'resume.txt')
        finally:
            os.unlink(tmp_path)
    
    def test_classify_nonexistent_resume(self):
        """Test classify endpoint with non-existent resume ID"""
        response = self.client.post('/api/classify/999')
        self.assertEqual(response.status_code, 404)
    
    def test_get_result_nonexistent_resume(self):
        """Test get result endpoint with non-existent resume ID"""
        response = self.client.get('/api/results/999')
        self.assertEqual(response.status_code, 404)
    
    def test_404_handler(self):
        """Test 404 error handler"""
        response = self.client.get('/nonexistent-page')
        self.assertEqual(response.status_code, 404)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'Not found')

if __name__ == '__main__':
    unittest.main()