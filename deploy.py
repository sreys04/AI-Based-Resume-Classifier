#!/usr/bin/env python3
"""
Deployment script for Resume Classifier AI
Supports multiple deployment platforms: Heroku, Railway, Render, and Docker
"""

import os
import subprocess
import sys
import json
from pathlib import Path

class DeploymentManager:
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.app_name = "resume-classifier-ai"
    
    def check_requirements(self):
        """Check if required tools are installed"""
        tools = {
            'git': 'Git is required for deployment',
            'docker': 'Docker is required for containerized deployment',
        }
        
        missing_tools = []
        for tool, description in tools.items():
            if not self.command_exists(tool):
                missing_tools.append(f"{tool}: {description}")
        
        if missing_tools:
            print("❌ Missing required tools:")
            for tool in missing_tools:
                print(f"  - {tool}")
            return False
        
        print("✅ All required tools are available")
        return True
    
    def command_exists(self, command):
        """Check if a command exists in PATH"""
        try:
            subprocess.run([command, '--version'], 
                          stdout=subprocess.DEVNULL, 
                          stderr=subprocess.DEVNULL, 
                          check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def create_env_file(self):
        """Create .env file with necessary environment variables"""
        env_content = """
# Resume Classifier AI Environment Variables
SECRET_KEY=your-super-secret-key-change-in-production
FLASK_ENV=production
DATABASE_URL=sqlite:///resume_classifier.db

# Optional: For production database
# DATABASE_URL=postgresql://user:password@host:port/dbname

# File upload settings
MAX_CONTENT_LENGTH=16777216
UPLOAD_FOLDER=data/uploads

# Model settings
MODEL_PATH=models/resume_classifier.pkl
VECTORIZER_PATH=models/vectorizer.pkl
"""
        
        env_file = self.project_dir / '.env'
        if not env_file.exists():
            with open(env_file, 'w') as f:
                f.write(env_content.strip())
            print(f"✅ Created .env file at {env_file}")
        else:
            print(f"ℹ️ .env file already exists at {env_file}")
    
    def create_procfile(self):
        """Create Procfile for Heroku deployment"""
        procfile_content = "web: gunicorn --bind 0.0.0.0:$PORT app:create_app()"
        
        procfile = self.project_dir / 'Procfile'
        with open(procfile, 'w') as f:
            f.write(procfile_content)
        print(f"✅ Created Procfile for Heroku deployment")
    
    def create_dockerignore(self):
        """Create .dockerignore file"""
        dockerignore_content = """
.git
.gitignore
README.md
.env
.venv
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.pytest_cache/
.coverage
htmlcov/
.tox/
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.DS_Store
.vscode/
.idea/
*.swp
*.swo
data/uploads/*
!data/uploads/.gitkeep
"""
        
        dockerignore = self.project_dir / '.dockerignore'
        with open(dockerignore, 'w') as f:
            f.write(dockerignore_content.strip())
        print(f"✅ Created .dockerignore file")
    
    def create_railway_config(self):
        """Create railway.json for Railway deployment"""
        railway_config = {
            "build": {
                "builder": "dockerfile"
            },
            "deploy": {
                "startCommand": "gunicorn --bind 0.0.0.0:$PORT app:create_app()",
                "healthcheckPath": "/"
            }
        }
        
        config_file = self.project_dir / 'railway.json'
        with open(config_file, 'w') as f:
            json.dump(railway_config, f, indent=2)
        print(f"✅ Created Railway configuration")
    
    def deploy_docker_local(self):
        """Deploy using Docker locally"""
        print("🚀 Building Docker image...")
        
        try:
            # Build Docker image
            subprocess.run([
                'docker', 'build', '-t', f'{self.app_name}:latest', '.'
            ], cwd=self.project_dir, check=True)
            
            print("✅ Docker image built successfully")
            print(f"🚀 To run the container:")
            print(f"   docker run -p 5000:5000 {self.app_name}:latest")
            print(f"📱 Access your app at: http://localhost:5000")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Docker build failed: {e}")
            return False
        
        return True
    
    def deploy_heroku(self):
        """Deploy to Heroku"""
        if not self.command_exists('heroku'):
            print("❌ Heroku CLI not found. Install it from https://devcenter.heroku.com/articles/heroku-cli")
            return False
        
        print("🚀 Deploying to Heroku...")
        
        try:
            # Create Heroku app
            subprocess.run([
                'heroku', 'create', self.app_name
            ], cwd=self.project_dir, check=True)
            
            # Set environment variables
            subprocess.run([
                'heroku', 'config:set', 
                'SECRET_KEY=your-production-secret-key-here',
                'FLASK_ENV=production'
            ], cwd=self.project_dir, check=True)
            
            # Deploy
            subprocess.run([
                'git', 'push', 'heroku', 'main'
            ], cwd=self.project_dir, check=True)
            
            print("✅ Successfully deployed to Heroku!")
            subprocess.run(['heroku', 'open'], cwd=self.project_dir)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Heroku deployment failed: {e}")
            return False
        
        return True
    
    def setup_git_repo(self):
        """Initialize git repository and make initial commit"""
        try:
            # Check if already a git repo
            if (self.project_dir / '.git').exists():
                print("ℹ️ Git repository already exists")
                return True
            
            # Initialize git repo
            subprocess.run(['git', 'init'], cwd=self.project_dir, check=True)
            subprocess.run(['git', 'add', '.'], cwd=self.project_dir, check=True)
            subprocess.run([
                'git', 'commit', '-m', 'Initial commit: Resume Classifier AI'
            ], cwd=self.project_dir, check=True)
            
            print("✅ Git repository initialized")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Git setup failed: {e}")
            return False
    
    def show_deployment_instructions(self):
        """Show deployment instructions for different platforms"""
        print("\n" + "="*60)
        print("🚀 DEPLOYMENT INSTRUCTIONS")
        print("="*60)
        
        print("\n📋 MANUAL DEPLOYMENT OPTIONS:")
        print("\n1. 🐳 DOCKER (Local)")
        print("   docker build -t resume-classifier-ai .")
        print("   docker run -p 5000:5000 resume-classifier-ai")
        print("   Access at: http://localhost:5000")
        
        print("\n2. 🟣 HEROKU")
        print("   heroku login")
        print("   heroku create your-app-name")
        print("   git push heroku main")
        
        print("\n3. 🚂 RAILWAY")
        print("   1. Visit https://railway.app")
        print("   2. Connect your GitHub repository")
        print("   3. Deploy automatically")
        
        print("\n4. 🟢 RENDER")
        print("   1. Visit https://render.com")
        print("   2. Connect your GitHub repository")
        print("   3. Select 'Web Service'")
        print("   4. Use: gunicorn --bind 0.0.0.0:$PORT app:create_app()")
        
        print("\n5. ⚡ VERCEL (Serverless)")
        print("   npm i -g vercel")
        print("   vercel --prod")
        
        print("\n📝 ENVIRONMENT VARIABLES TO SET:")
        print("   - SECRET_KEY: your-secret-key")
        print("   - FLASK_ENV: production")
        print("   - DATABASE_URL: your-database-url (optional)")
        
        print("\n" + "="*60)
    
    def run_deployment(self, platform='docker'):
        """Run complete deployment process"""
        print(f"🚀 Starting deployment for platform: {platform}")
        
        if not self.check_requirements():
            return False
        
        # Setup files
        self.create_env_file()
        self.create_procfile()
        self.create_dockerignore()
        self.create_railway_config()
        
        # Setup git if needed
        if platform != 'docker':
            if not self.setup_git_repo():
                return False
        
        # Deploy to specific platform
        if platform == 'docker':
            success = self.deploy_docker_local()
        elif platform == 'heroku':
            success = self.deploy_heroku()
        else:
            print(f"❌ Platform '{platform}' not supported for automated deployment")
            self.show_deployment_instructions()
            return False
        
        if success:
            print(f"✅ Deployment to {platform} completed successfully!")
        
        return success

def main():
    """Main function"""
    manager = DeploymentManager()
    
    if len(sys.argv) > 1:
        platform = sys.argv[1].lower()
        manager.run_deployment(platform)
    else:
        print("🔧 Resume Classifier AI - Deployment Setup")
        print("\nUsage: python deploy.py [platform]")
        print("Platforms: docker, heroku")
        print("\nOr run without arguments to see all instructions:")
        manager.show_deployment_instructions()

if __name__ == '__main__':
    main()