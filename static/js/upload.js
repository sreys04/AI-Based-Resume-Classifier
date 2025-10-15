// Resume Classifier AI - Upload Functionality

class ResumeUploader {
    constructor() {
        this.selectedFile = null;
        this.currentResumeId = null;
        this.initializeElements();
        this.attachEventListeners();
    }

    initializeElements() {
        // File input elements
        this.dropZone = document.getElementById('drop-zone');
        this.fileInput = document.getElementById('file-input');
        this.fileInfo = document.getElementById('file-info');
        this.fileName = document.getElementById('file-name');
        this.fileSize = document.getElementById('file-size');
        this.removeFileBtn = document.getElementById('remove-file');

        // Action buttons
        this.uploadBtn = document.getElementById('upload-btn');
        this.cancelBtn = document.getElementById('cancel-btn');
        this.newUploadBtn = document.getElementById('new-upload-btn');
        this.retryBtn = document.getElementById('retry-btn');

        // Container elements
        this.uploadCard = document.querySelector('.upload-card');
        this.progressContainer = document.getElementById('progress-container');
        this.resultsContainer = document.getElementById('results-container');
        this.errorContainer = document.getElementById('error-container');

        // Progress elements
        this.progressFill = document.getElementById('progress-fill');
        this.progressText = document.getElementById('progress-text');

        // Results elements
        this.predictedCategory = document.getElementById('predicted-category');
        this.confidenceFill = document.getElementById('confidence-fill');
        this.confidenceText = document.getElementById('confidence-text');
        this.scoresList = document.getElementById('scores-list');

        // Error elements
        this.errorMessage = document.getElementById('error-message');
    }

    attachEventListeners() {
        // Drop zone events
        this.dropZone.addEventListener('click', () => this.fileInput.click());
        this.dropZone.addEventListener('dragover', this.handleDragOver.bind(this));
        this.dropZone.addEventListener('dragleave', this.handleDragLeave.bind(this));
        this.dropZone.addEventListener('drop', this.handleDrop.bind(this));

        // File input change
        this.fileInput.addEventListener('change', this.handleFileSelect.bind(this));

        // Button events
        this.removeFileBtn.addEventListener('click', this.removeFile.bind(this));
        this.uploadBtn.addEventListener('click', this.uploadFile.bind(this));
        this.cancelBtn.addEventListener('click', this.cancelUpload.bind(this));
        this.newUploadBtn.addEventListener('click', this.resetUpload.bind(this));
        this.retryBtn.addEventListener('click', this.resetUpload.bind(this));
    }

    handleDragOver(e) {
        e.preventDefault();
        this.dropZone.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.dropZone.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.dropZone.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    processFile(file) {
        // Validate file type
        const allowedTypes = ['application/pdf', 'application/msword', 
                             'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 
                             'text/plain'];
        const allowedExtensions = ['.pdf', '.doc', '.docx', '.txt'];
        
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
            this.showError('Please select a valid file type (PDF, DOC, DOCX, or TXT)');
            return;
        }

        // Validate file size (16MB max)
        const maxSize = 16 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showError('File size too large. Please select a file smaller than 16MB.');
            return;
        }

        this.selectedFile = file;
        this.displayFileInfo();
    }

    displayFileInfo() {
        if (!this.selectedFile) return;

        this.fileName.textContent = this.selectedFile.name;
        this.fileSize.textContent = this.formatFileSize(this.selectedFile.size);
        
        this.dropZone.style.display = 'none';
        this.fileInfo.style.display = 'flex';
        this.uploadBtn.disabled = false;
    }

    removeFile() {
        this.selectedFile = null;
        this.fileInput.value = '';
        this.dropZone.style.display = 'block';
        this.fileInfo.style.display = 'none';
        this.uploadBtn.disabled = true;
    }

    async uploadFile() {
        if (!this.selectedFile) return;

        try {
            this.showProgress();
            this.updateProgress(10, 'Uploading file...');

            // Create form data
            const formData = new FormData();
            formData.append('file', this.selectedFile);

            // Upload file
            const uploadResponse = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!uploadResponse.ok) {
                const errorData = await uploadResponse.json();
                throw new Error(errorData.error || 'Upload failed');
            }

            const uploadResult = await uploadResponse.json();
            this.currentResumeId = uploadResult.resume_id;

            this.updateProgress(50, 'Processing resume...');

            // Classify resume
            const classifyResponse = await fetch(`/api/classify/${this.currentResumeId}`, {
                method: 'POST'
            });

            if (!classifyResponse.ok) {
                const errorData = await classifyResponse.json();
                throw new Error(errorData.error || 'Classification failed');
            }

            const classifyResult = await classifyResponse.json();
            
            this.updateProgress(100, 'Complete!');
            
            setTimeout(() => {
                this.showResults(classifyResult.result);
            }, 1000);

        } catch (error) {
            console.error('Upload error:', error);
            this.showError(error.message || 'An error occurred during upload');
        }
    }

    cancelUpload() {
        // In a real implementation, you might want to abort the fetch request
        this.resetUpload();
    }

    showProgress() {
        this.uploadCard.style.display = 'none';
        this.progressContainer.style.display = 'block';
        this.resultsContainer.style.display = 'none';
        this.errorContainer.style.display = 'none';
    }

    updateProgress(percentage, text) {
        this.progressFill.style.width = percentage + '%';
        this.progressText.textContent = text;
    }

    showResults(result) {
        this.uploadCard.style.display = 'none';
        this.progressContainer.style.display = 'none';
        this.resultsContainer.style.display = 'block';
        this.errorContainer.style.display = 'none';

        // Display main prediction
        this.predictedCategory.textContent = result.predicted_category;
        
        // Display confidence score
        const confidence = Math.round(result.confidence_score * 100);
        this.confidenceFill.style.width = confidence + '%';
        this.confidenceText.textContent = confidence + '%';

        // Display all scores
        this.displayAllScores(result.all_scores);
    }

    displayAllScores(allScores) {
        this.scoresList.innerHTML = '';

        // Sort scores by value (highest first)
        const sortedScores = Object.entries(allScores)
            .sort(([,a], [,b]) => b - a);

        sortedScores.forEach(([category, score]) => {
            const scoreItem = document.createElement('div');
            scoreItem.className = 'score-item';
            
            const percentage = Math.round(score * 100);
            
            scoreItem.innerHTML = `
                <span>${category}</span>
                <div style="display: flex; align-items: center;">
                    <span style="margin-right: 10px; font-weight: 500;">${percentage}%</span>
                    <div class="score-bar">
                        <div class="score-fill" style="width: ${percentage}%;"></div>
                    </div>
                </div>
            `;
            
            this.scoresList.appendChild(scoreItem);
        });
    }

    showError(message) {
        this.uploadCard.style.display = 'none';
        this.progressContainer.style.display = 'none';
        this.resultsContainer.style.display = 'none';
        this.errorContainer.style.display = 'block';

        this.errorMessage.textContent = message;
    }

    resetUpload() {
        this.selectedFile = null;
        this.currentResumeId = null;
        this.fileInput.value = '';
        
        this.uploadCard.style.display = 'block';
        this.progressContainer.style.display = 'none';
        this.resultsContainer.style.display = 'none';
        this.errorContainer.style.display = 'none';
        
        this.dropZone.style.display = 'block';
        this.fileInfo.style.display = 'none';
        this.uploadBtn.disabled = true;
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// Initialize uploader when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ResumeUploader();
});