import os
import logging
from flask import Flask, render_template, request, session, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import PyPDF2
from io import BytesIO
import zipfile
import re
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Replace with a secure key in production

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'zip'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Sample job description
SAMPLE_JD = """
Job Title: Software Engineer
Responsibilities: Develop and maintain web applications, collaborate with cross-functional teams, write clean code.
Requirements: Bachelor’s degree in Computer Science, 3+ years of experience in Python and JavaScript, strong problem-solving skills.
Skills: Python, JavaScript, SQL, teamwork, leadership.
"""

# Utility functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ''

def extract_text_from_zip(zip_file):
    texts = []
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith('.pdf'):
                with zip_ref.open(file_name) as file:
                    text = extract_text_from_pdf(BytesIO(file.read()))
                    texts.append((file_name, text))
    return texts

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(tokens)

def extract_keywords(text, top_n=5):
    try:
        vectorizer = TfidfVectorizer(max_features=50)
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        keyword_scores = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
        return [keyword for keyword, score in keyword_scores[:top_n]]
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return []

def highlight_text(text, keywords):
    for keyword in keywords:
        text = re.sub(f'\\b{keyword}\\b', f'<mark>{keyword}</mark>', text, flags=re.IGNORECASE)
    return text

def analyze_jd(jd_text):
    sections = {
        'Responsibilities': '',
        'Requirements': '',
        'Skills': ''
    }
    current_section = None
    for line in jd_text.split('\n'):
        line = line.strip()
        if 'responsibilities' in line.lower():
            current_section = 'Responsibilities'
        elif 'requirements' in line.lower():
            current_section = 'Requirements'
        elif 'skills' in line.lower():
            current_section = 'Skills'
        elif current_section and line:
            sections[current_section] += line + ' '
    return sections

def score_resume(resume_text, jd_text, custom_keywords):
    resume_processed = preprocess_text(resume_text)
    jd_processed = preprocess_text(jd_text)
    
    documents = [jd_processed, resume_processed]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    similarity = (tfidf_matrix * tfidf_matrix.T).toarray()[0][1] * 100
    
    resume_words = set(resume_processed.split())
    keyword_matches = sum(1 for keyword in custom_keywords if keyword.lower() in resume_words)
    score = min(100, similarity + (keyword_matches * 5))
    
    section_scores = {
        'Education': max(0, min(100, similarity * 0.8 + np.random.uniform(-10, 10))),
        'Experience': max(0, min(100, similarity * 0.9 + np.random.uniform(-10, 10))),
        'Skills': max(0, min(100, similarity * 1.1 + np.random.uniform(-10, 10)))
    }
    
    jd_section_scores = {
        'Responsibilities': max(0, min(100, similarity * 0.9 + np.random.uniform(-5, 5))),
        'Requirements': max(0, min(100, similarity * 0.95 + np.random.uniform(-5, 5)))
    }
    
    jd_feedback = "Matches well with responsibilities but lacks some required skills."
    keyword_density = (len(resume_words), 'python, sql')
    length_feedback = (len(resume_text.split()), ['Too short', 'Add more details'])
    bias = ['Gendered language detected'] if 'he/she' in resume_text.lower() else []
    ats_issues = ['Missing keywords'] if keyword_matches < len(custom_keywords) / 2 else []
    suggestions = {
        'Education': 'Highlight relevant coursework.',
        'Experience': 'Quantify achievements.',
        'Skills': 'Add more technical skills.'
    }
    
    return {
        'score': score,
        'section_scores': section_scores,
        'jd_section_scores': jd_section_scores,
        'jd_feedback': jd_feedback,
        'keyword_density': keyword_density,
        'length_feedback': length_feedback,
        'bias': bias,
        'ats_issues': ats_issues,
        'suggestions': suggestions,
        'highlighted_text': highlight_text(resume_text, custom_keywords)
    }

# Define get_score_class to fix the UndefinedError
def get_score_class(score):
    try:
        score = float(score)  # Ensure score is a number
        if score >= 80:
            return 'score-good'
        elif score >= 50:
            return 'score-average'
        else:
            return 'score-poor'
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid score value: {score}, error: {e}")
        return 'score-poor'  # Fallback class

# Routes
@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    results = []
    error = None
    highlighted_jd = None
    jd_keywords = []
    jd_sections = {}
    jd_keyword_suggestions = []
    custom_keywords = []
    dark_mode = session.get('theme', 'theme-light') == 'theme-dark'

    if request.method == 'POST':
        logger.debug("Form submitted to /dashboard")
        logger.debug(f"Job Description (Text): {request.form.get('job_description')}")
        logger.debug(f"JD File: {request.files.get('jd_file')}")
        logger.debug(f"Resumes: {request.files.getlist('resumes')}")
        logger.debug(f"ZIP File: {request.files.get('zip_file')}")
        logger.debug(f"Custom Keywords: {request.form.get('custom_keywords')}")

        jd_text = request.form.get('job_description', '')
        jd_file = request.files.get('jd_file')
        if jd_file and allowed_file(jd_file.filename):
            jd_text = extract_text_from_pdf(jd_file)
            logger.debug(f"Extracted JD text from file: {jd_text[:100]}...")

        if not jd_text:
            error = "Please provide a job description (text or PDF)."
        else:
            custom_keywords = request.form.get('custom_keywords', '').split(',')
            custom_keywords = [keyword.strip().lower() for keyword in custom_keywords if keyword.strip()]
            logger.debug(f"Custom keywords: {custom_keywords}")

            jd_keywords = extract_keywords(jd_text)
            highlighted_jd = highlight_text(jd_text, jd_keywords)
            jd_sections = analyze_jd(jd_text)
            jd_keyword_suggestions = ['communication', 'problem-solving']

            resumes = []
            zip_file = request.files.get('zip_file')
            if zip_file and allowed_file(zip_file.filename):
                resumes.extend(extract_text_from_zip(zip_file))
            else:
                resume_files = request.files.getlist('resumes')
                for resume_file in resume_files:
                    if resume_file and allowed_file(resume_file.filename):
                        text = extract_text_from_pdf(resume_file)
                        resumes.append((resume_file.filename, text))

            if not resumes:
                error = "Please upload at least one resume or a ZIP file."
            else:
                resume_texts = [text for _, text in resumes]
                processed_texts = [preprocess_text(text) for text in resume_texts]
                vectorizer = TfidfVectorizer()
                try:
                    tfidf_matrix = vectorizer.fit_transform(processed_texts)
                    kmeans = KMeans(n_clusters=min(3, len(resumes)), random_state=42)
                    clusters = kmeans.fit_predict(tfidf_matrix)
                except Exception as e:
                    logger.error(f"Error in clustering: {e}")
                    clusters = [0] * len(resumes)

                for i, (resume_name, resume_text) in enumerate(resumes):
                    result = score_resume(resume_text, jd_text, custom_keywords)
                    result['resume'] = resume_name
                    result['cluster'] = clusters[i]
                    results.append(result)

                results.sort(key=lambda x: x['score'], reverse=True)

    # Line 698 (as per traceback): render_template call
    return render_template(
        'dashboard.html',
        results=results,
        error=error,
        highlighted_jd=highlighted_jd,
        jd_keywords=jd_keywords,
        jd_sections=jd_sections,
        jd_keyword_suggestions=jd_keyword_suggestions,
        sample_jd=SAMPLE_JD,
        dark_mode=dark_mode,
        custom_keywords=custom_keywords,
        getScoreClass=get_score_class  # Add this to fix the UndefinedError
    )

@app.route('/set_theme', methods=['POST'])
def set_theme():
    theme = request.form.get('theme', 'theme-light')
    session['theme'] = theme
    return '', 204

@app.route('/progress')
def progress():
    import random
    progress = random.randint(0, 100)
    return {'progress': progress}

@app.route('/download_report', methods=['POST'])
def download_report():
    buffer = BytesIO()
    buffer.write(b"Dummy PDF Report")
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name='report.pdf', mimetype='application/pdf')

@app.route('/download_csv', methods=['POST'])
def download_csv():
    buffer = BytesIO()
    buffer.write(b"resume,score,cluster\nResume1,85,0\nResume2,70,1")
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name='report.csv', mimetype='text/csv')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)