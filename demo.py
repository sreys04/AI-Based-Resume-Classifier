#!/usr/bin/env python3
"""
Demo script for the AI-based Resume Classifier
This script demonstrates how to use the trained model for predictions
"""

import pandas as pd
from main import ResumeClassifier

def demo_individual_predictions():
    """Demo individual resume predictions"""
    print("=== Individual Resume Prediction Demo ===\n")
    
    # Initialize and load the trained model
    classifier = ResumeClassifier()
    
    try:
        classifier.load_model('resume_classifier_model.pkl')
    except FileNotFoundError:
        print("Model not found. Please run main.py first to train the model.")
        return
    
    # Sample resumes for different categories
    sample_resumes = {
        "Data Science Resume": """
        Senior Data Scientist with 8 years of experience in machine learning, 
        statistical analysis, and big data processing. Expert in Python, R, 
        pandas, numpy, scikit-learn, tensorflow, and pytorch. Experience with 
        deep learning, computer vision, and natural language processing. 
        Strong background in predictive modeling and data visualization.
        """,
        
        "Software Engineering Resume": """
        Full Stack Software Engineer with 6 years of experience in web development. 
        Proficient in JavaScript, React, Node.js, Python, and Java. Experience with 
        microservices architecture, RESTful APIs, and cloud deployment on AWS. 
        Strong understanding of software design patterns and agile methodologies.
        """,
        
        "Marketing Resume": """
        Digital Marketing Manager with 5 years of experience in campaign management, 
        social media marketing, and content strategy. Skilled in SEO, SEM, Google 
        Analytics, and marketing automation tools. Experience with brand management, 
        customer segmentation, and performance marketing optimization.
        """,
        
        "Finance Resume": """
        Financial Analyst with 4 years of experience in investment banking and 
        portfolio management. Strong background in financial modeling, risk assessment, 
        and quantitative analysis. Proficient in Excel, VBA, and financial reporting. 
        Experience with derivatives trading and equity research.
        """,
        
        "HR Resume": """
        Human Resources Specialist with 7 years of experience in talent acquisition, 
        employee relations, and performance management. Skilled in recruitment, 
        training and development, and HR policy implementation. Experience with 
        HRIS systems, compensation planning, and organizational development.
        """
    }
    
    # Make predictions for each sample resume
    for title, resume_text in sample_resumes.items():
        print(f"\n{title}:")
        print("-" * 50)
        print(f"Resume excerpt: {resume_text[:100]}...")
        
        try:
            category, probabilities = classifier.predict_resume_category(resume_text)
            print(f"Predicted Category: {category}")
            print("Confidence Scores:")
            for cat, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                print(f"  {cat}: {prob:.3f}")
        except Exception as e:
            print(f"Error predicting category: {e}")
        
        print()

def demo_batch_predictions():
    """Demo batch processing of resumes"""
    print("=== Batch Resume Processing Demo ===\n")
    
    # Initialize classifier
    classifier = ResumeClassifier()
    
    try:
        classifier.load_model('resume_classifier_model.pkl')
    except FileNotFoundError:
        print("Model not found. Please run main.py first to train the model.")
        return
    
    # Create sample batch data
    batch_resumes = [
        "Python developer with machine learning experience using scikit-learn and pandas",
        "Marketing professional skilled in digital campaigns and social media strategy",
        "Financial analyst with expertise in investment banking and portfolio management",
        "HR manager with experience in talent acquisition and employee relations",
        "Software engineer proficient in React, Node.js, and full-stack development"
    ]
    
    # Process batch
    results = []
    for i, resume in enumerate(batch_resumes, 1):
        try:
            category, probabilities = classifier.predict_resume_category(resume)
            results.append({
                'Resume_ID': f'Resume_{i}',
                'Resume_Text': resume,
                'Predicted_Category': category,
                'Confidence': max(probabilities.values())
            })
        except Exception as e:
            print(f"Error processing resume {i}: {e}")
    
    # Display results
    if results:
        df_results = pd.DataFrame(results)
        print("Batch Processing Results:")
        print("=" * 80)
        for _, row in df_results.iterrows():
            print(f"ID: {row['Resume_ID']}")
            print(f"Text: {row['Resume_Text'][:60]}...")
            print(f"Category: {row['Predicted_Category']}")
            print(f"Confidence: {row['Confidence']:.3f}")
            print("-" * 40)
        
        # Save results
        df_results.to_csv('batch_predictions.csv', index=False)
        print(f"\nResults saved to 'batch_predictions.csv'")

def interactive_demo():
    """Interactive demo where user can input their own resume"""
    print("=== Interactive Resume Classification Demo ===\n")
    
    # Initialize classifier
    classifier = ResumeClassifier()
    
    try:
        classifier.load_model('resume_classifier_model.pkl')
    except FileNotFoundError:
        print("Model not found. Please run main.py first to train the model.")
        return
    
    print("Enter your resume text (or 'quit' to exit):")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\nYour resume text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                print("Please enter some text.")
                continue
            
            # Make prediction
            category, probabilities = classifier.predict_resume_category(user_input)
            
            print(f"\nPredicted Category: {category}")
            print("Confidence Scores:")
            for cat, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                print(f"  {cat}: {prob:.3f}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main demo function"""
    print("AI-Based Resume Classifier Demo")
    print("=" * 40)
    
    while True:
        print("\nChoose a demo option:")
        print("1. Individual Predictions Demo")
        print("2. Batch Processing Demo")
        print("3. Interactive Demo")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            demo_individual_predictions()
        elif choice == '2':
            demo_batch_predictions()
        elif choice == '3':
            interactive_demo()
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()