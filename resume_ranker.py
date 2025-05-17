import os
import pdfplumber
import spacy
import numpy as np
import re
import zipfile
import csv
import io
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from flask import Flask, request, render_template, redirect, url_for, session, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from io import BytesIO, StringIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import logging
import threading
import queue

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret-key-123'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minutes session timeout
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load SpaCy model with optimized settings
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

# Sample Job Description for testing
SAMPLE_JD = """
Job Title: Software Engineer

Responsibilities:
- Design, develop, and maintain web applications using Python, JavaScript, and React.
- Collaborate with cross-functional teams to define and implement new features.
- Write clean, scalable, and efficient code following best practices.
- Troubleshoot and debug issues in production environments.
- Participate in code reviews and ensure high-quality deliverables.

Requirements:
- Bachelor’s degree in Computer Science, Engineering, or related field.
- 3+ years of experience in software development.
- Strong proficiency in Python, JavaScript, and React.
- Experience with RESTful APIs and database systems like PostgreSQL.
- Familiarity with cloud platforms such as AWS or Azure.
- Excellent problem-solving and communication skills.

Skills:
- Python, JavaScript, React
- RESTful APIs, PostgreSQL
- AWS, Azure
- Problem-solving, teamwork
"""

# Common industry keywords for JD suggestion (Software Engineer role)
INDUSTRY_KEYWORDS = [
    'agile', 'scrum', 'docker', 'kubernetes', 'ci/cd', 'microservices', 'graphql',
    'typescript', 'node.js', 'sql', 'nosql', 'mongodb', 'redis', 'git', 'jenkins',
    'security', 'testing', 'unit testing', 'automation', 'devops', 'linux', 'networking'
]

# In-memory user storage
users = {}

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        logger.debug(f"Extracting text from PDF: {pdf_path}")
        with pdfplumber.open(pdf_path) as pdf:
            text = ''
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
        if not text.strip():
            logger.warning(f"No text extracted from PDF: {pdf_path}")
            return None
        logger.debug(f"Extracted text length: {len(text)}")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        return None

# Validate resume for ATS compatibility
def validate_resume_format(text):
    logger.debug("Validating resume format for ATS compatibility")
    issues = []
    
    # Check for special characters
    special_chars = re.findall(r'[^\w\s.,-]', text)
    if special_chars:
        issues.append(f"Special characters detected ({', '.join(set(special_chars[:5]))}). ATS systems may not parse these correctly; consider removing them.")
    
    # Check for potential images or tables (basic heuristic: look for low text density)
    lines = text.split('\n')
    empty_lines = sum(1 for line in lines if not line.strip())
    if empty_lines / len(lines) > 0.5:
        issues.append("High proportion of empty lines detected. This may indicate images, tables, or complex formatting that ATS systems struggle with. Use plain text formatting.")
    
    # Check for non-standard fonts (heuristic: pdfplumber might fail to extract some text if fonts are unsupported)
    if not text.strip():
        issues.append("No text extracted. This may indicate the use of unsupported fonts or image-based PDFs. Convert to a text-based PDF for better ATS compatibility.")
    
    return issues if issues else ["Resume format appears ATS-friendly."]

# Extract sections from resume
def extract_resume_sections(text):
    logger.debug("Extracting resume sections")
    sections = {'Education': '', 'Experience': '', 'Skills': ''}
    current_section = None
    lines = text.split('\n')
    
    for line in lines:
        line_lower = line.lower()
        if 'education' in line_lower:
            current_section = 'Education'
        elif 'experience' in line_lower or 'work' in line_lower:
            current_section = 'Experience'
        elif 'skills' in line_lower:
            current_section = 'Skills'
        elif current_section:
            sections[current_section] += line + ' '
    
    return sections

# Extract sections from JD
def extract_jd_sections(text):
    logger.debug("Extracting JD sections")
    sections = {'Responsibilities': '', 'Requirements': '', 'Skills': ''}
    current_section = None
    lines = text.split('\n')
    
    for line in lines:
        line_lower = line.lower()
        if 'responsibilities' in line_lower or 'duties' in line_lower:
            current_section = 'Responsibilities'
        elif 'requirements' in line_lower or 'qualifications' in line_lower:
            current_section = 'Requirements'
        elif 'skills' in line_lower:
            current_section = 'Skills'
        elif current_section:
            sections[current_section] += line + ' '
    
    return sections

# Preprocess text with SpaCy
def preprocess_text(text):
    logger.debug(f"Preprocessing text of length: {len(text)}")
    try:
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        return ' '.join(tokens)
    except Exception as e:
        logger.error(f"Error preprocessing text: {str(e)}")
        return ''

# Advanced bias detection
def detect_bias(text):
    logger.debug("Detecting bias in text")
    bias_indicators = []
    if re.search(r'\b(he|she|him|her|his|hers)\b', text.lower()):
        bias_indicators.append("Gender pronouns detected (e.g., he, she). Suggestion: Use gender-neutral terms like 'they' or rewrite the sentence.")
    if re.search(r'\b(born|age|years old|graduated in \d{4})\b', text.lower()):
        bias_indicators.append("Age-related terms detected (e.g., born, age). Suggestion: Remove age-related information unless necessary.")
    if re.search(r'\b(indian|chinese|american|african|european|asian|hispanic|latino)\b', text.lower()):
        bias_indicators.append("Ethnicity/nationality terms detected. Suggestion: Avoid mentioning ethnicity or nationality unless relevant to the role.")
    if re.search(r'\b(christian|muslim|hindu|buddhist|jewish|sikh)\b', text.lower()):
        bias_indicators.append("Religious terms detected. Suggestion: Remove references to religion unless relevant to the role.")
    return bias_indicators if bias_indicators else ["No bias indicators detected."]

# Extract keywords from JD or user input
def extract_keywords(text, custom_keywords=None):
    logger.debug("Extracting keywords")
    try:
        doc = nlp(text.lower())
        keywords = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        keywords = list(set(keywords))
        if custom_keywords:
            custom_keywords = [kw.strip().lower() for kw in custom_keywords.split(',')]
            keywords.extend(kw for kw in custom_keywords if kw not in keywords)
        return keywords
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        return []

# Suggest additional JD keywords based on industry standards
def suggest_jd_keywords(extracted_keywords):
    logger.debug("Suggesting additional JD keywords")
    missing_keywords = [kw for kw in INDUSTRY_KEYWORDS if kw not in extracted_keywords]
    return missing_keywords[:5]  # Limit to 5 suggestions

# Highlight keywords in text
def highlight_keywords(text, keywords):
    logger.debug("Highlighting keywords in text")
    highlighted_text = text
    for keyword in keywords:
        pattern = rf'\b{re.escape(keyword)}\b'
        highlighted_text = re.sub(pattern, f'<span class="bg-yellow-200">{keyword}</span>', highlighted_text, flags=re.IGNORECASE)
    return highlighted_text

# Analyze keyword density in resume
def analyze_keyword_density(resume_text, jd_keywords):
    logger.debug("Analyzing keyword density")
    word_count = len(resume_text.split())
    if word_count == 0:
        return 0.0, "Resume is empty."
    
    keyword_count = sum(resume_text.lower().count(kw) for kw in jd_keywords)
    density = (keyword_count / word_count) * 100
    density = round(density, 2)
    
    feedback = ""
    if density < 2.0:
        feedback = "Keyword density is too low. Aim for 2-3% to improve ATS ranking."
    elif density > 5.0:
        feedback = "Keyword density is too high. Reduce to 2-3% to avoid keyword stuffing."
    else:
        feedback = "Keyword density is optimal for ATS ranking."
    
    return density, feedback

# Provide feedback on resume length
def analyze_resume_length(text):
    logger.debug("Analyzing resume length")
    word_count = len(text.split())
    line_count = len(text.split('\n'))
    
    feedback = []
    # Word count feedback
    if word_count < 400:
        feedback.append(f"Resume is short ({word_count} words). Aim for 400-600 words for a comprehensive one-page resume.")
    elif word_count > 600:
        feedback.append(f"Resume is long ({word_count} words). Aim for 400-600 words to fit on one page.")
    else:
        feedback.append("Resume length is appropriate for a one-page resume.")
    
    # Page count approximation (heuristic: assume ~300 words per page with standard formatting)
    estimated_pages = word_count / 300
    if estimated_pages > 1.2:
        feedback.append("Resume may exceed one page. Consider reducing content or adjusting formatting.")
    
    return word_count, feedback

# Extract noun phrases using POS tags
def extract_noun_phrases(doc, missing_keywords):
    logger.debug("Extracting noun phrases using POS tags")
    phrases = []
    current_phrase = []
    
    for i, token in enumerate(doc):
        if token.pos_ in ('NOUN', 'PROPN'):
            current_phrase.append(token.text)
            if i > 0 and doc[i-1].pos_ == 'ADJ':
                current_phrase.insert(0, doc[i-1].text)
            if i < len(doc) - 1 and doc[i+1].pos_ in ('NOUN', 'PROPN'):
                continue
            phrase_text = ' '.join(current_phrase)
            if any(kw in phrase_text.lower() for kw in missing_keywords):
                phrases.append(phrase_text)
            current_phrase = []
        else:
            current_phrase = []
    
    return phrases[:2]

# Generate detailed improvement suggestions
def generate_improvement_suggestions(resume_text, jd_text, jd_keywords):
    logger.debug("Generating improvement suggestions")
    suggestions = {}
    resume_sections = extract_resume_sections(resume_text)
    jd_sections = extract_jd_sections(jd_text)
    
    for section, content in resume_sections.items():
        if not content:
            suggestions[section] = f"Section missing or empty. Consider adding relevant {section.lower()} details from the JD."
            continue
        
        resume_doc = nlp(content.lower())
        resume_keywords = set([token.lemma_ for token in resume_doc if not token.is_stop and token.is_alpha])
        missing_keywords = [kw for kw in jd_keywords if kw not in resume_keywords]
        
        if missing_keywords:
            suggestions[section] = f"Add keywords: {', '.join(missing_keywords[:3])}. "
            jd_section_content = jd_sections.get(section, '')
            if jd_section_content:
                jd_doc = nlp(jd_section_content.lower())
                jd_phrases = extract_noun_phrases(jd_doc, missing_keywords)
                if jd_phrases:
                    suggestions[section] += f"Consider including phrases like: {', '.join(jd_phrases)}."
                else:
                    suggestions[section] += "Align your content more closely with the JD."
        else:
            suggestions[section] = "Section aligns well with the JD."
    
    return suggestions

# Score resumes, sections, and cluster candidates
def score_resumes(job_description, resume_texts, resume_files, custom_keywords=None):
    logger.debug("Starting resume scoring")
    
    MAX_RESUMES = 5
    if len(resume_texts) > MAX_RESUMES:
        logger.warning(f"Too many resumes ({len(resume_texts)}), limiting to {MAX_RESUMES}")
        resume_texts = resume_texts[:MAX_RESUMES]
        resume_files = resume_files[:MAX_RESUMES]
    
    job_desc_processed = preprocess_text(job_description)
    resume_processed = [preprocess_text(text) for text in resume_texts if text]
    
    if not job_desc_processed or not resume_processed:
        logger.error("Preprocessing failed: Job description or resumes empty after preprocessing")
        return [], job_description, [], {}, [], [], []
    
    jd_sections = extract_jd_sections(job_description)
    jd_keywords = extract_keywords(job_description, custom_keywords)
    jd_keyword_suggestions = suggest_jd_keywords(jd_keywords)
    
    logger.debug("Vectorizing texts")
    try:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([job_desc_processed] + resume_processed)
        job_vector = vectors[0]
        resume_vectors = vectors[1:]
        scores = cosine_similarity(job_vector, resume_vectors).flatten()
    except Exception as e:
        logger.error(f"Error during vectorization or scoring: {str(e)}")
        return [], job_description, jd_keywords, jd_sections, [], [], jd_keyword_suggestions
    
    resume_section_scores = []
    for text in resume_texts:
        sections = extract_resume_sections(text)
        section_results = {}
        for section, content in sections.items():
            if content:
                section_processed = preprocess_text(content)
                if section_processed:
                    section_vector = vectorizer.transform([section_processed])
                    section_score = cosine_similarity(job_vector, section_vector).flatten()[0]
                    section_results[section] = round(section_score * 100, 2)
                else:
                    section_results[section] = 0.0
            else:
                section_results[section] = 0.0
        resume_section_scores.append(section_results)
    
    jd_section_scores = []
    for text in resume_texts:
        jd_section_results = {}
        for section, content in jd_sections.items():
            if content:
                section_processed = preprocess_text(content)
                if section_processed:
                    section_vector = vectorizer.transform([section_processed])
                    resume_processed = preprocess_text(text)
                    if resume_processed:
                        resume_vector = vectorizer.transform([resume_processed])
                        section_score = cosine_similarity(section_vector, resume_vector).flatten()[0]
                        jd_section_results[section] = round(section_score * 100, 2)
                    else:
                        jd_section_results[section] = 0.0
                else:
                    jd_section_results[section] = 0.0
            else:
                jd_section_results[section] = 0.0
        jd_section_scores.append(jd_section_results)
    
    logger.debug("Performing clustering")
    num_clusters = min(len(resume_texts), 3) if len(resume_texts) > 1 else 1
    if num_clusters > 1:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(resume_vectors)
    else:
        clusters = [0] * len(resume_texts)
    
    suggestions = [generate_improvement_suggestions(text, job_description, jd_keywords) for text in resume_texts]
    
    jd_feedback = []
    for resume_text in resume_texts:
        resume_doc = nlp(resume_text.lower())
        resume_keywords = set([token.lemma_ for token in resume_doc if not token.is_stop and token.is_alpha])
        missing_keywords = [kw for kw in jd_keywords if kw not in resume_keywords]
        if missing_keywords:
            jd_feedback.append(f"Missing key skills from JD: {', '.join(missing_keywords[:3])}. Consider adding these to improve alignment.")
        else:
            jd_feedback.append("Resume aligns well with JD.")
    
    biases = [detect_bias(text) for text in resume_texts]
    ats_issues = [validate_resume_format(text) for text in resume_texts]
    keyword_densities = [analyze_keyword_density(text, jd_keywords) for text in resume_texts]
    length_feedbacks = [analyze_resume_length(text) for text in resume_texts]
    
    highlighted_texts = [highlight_keywords(text, jd_keywords) for text in resume_texts]
    highlighted_jd = highlight_keywords(job_description, jd_keywords)
    
    target_score = 80.0
    improvements = []
    for score in scores:
        improvement_needed = max(0, target_score - (score * 100))
        improvements.append(round(improvement_needed, 2))
    
    results = []
    for i, score in enumerate(scores):
        results.append({
            'resume': resume_files[i].filename,
            'raw_text': resume_texts[i],
            'highlighted_text': highlighted_texts[i],
            'score': round(score * 100, 2),
            'section_scores': resume_section_scores[i],
            'jd_section_scores': jd_section_scores[i],
            'suggestions': suggestions[i],
            'cluster': clusters[i],
            'bias': biases[i],
            'ats_issues': ats_issues[i],
            'keyword_density': keyword_densities[i],
            'length_feedback': length_feedbacks[i],
            'improvement_needed': improvements[i],
            'jd_feedback': jd_feedback[i]
        })
    
    logger.debug(f"Scoring completed. Results: {len(results)}")
    return sorted(results, key=lambda x: x['score'], reverse=True), highlighted_jd, jd_keywords, jd_sections, ats_issues, keyword_densities, jd_keyword_suggestions

# Wrapper to run score_resumes with a timeout
def score_resumes_with_timeout(job_description, resume_texts, resume_files, custom_keywords=None, timeout=120):
    result_queue = queue.Queue()
    
    def target():
        try:
            result = score_resumes(job_description, resume_texts, resume_files, custom_keywords)
            result_queue.put(('success', result))
        except Exception as e:
            result_queue.put(('error', str(e)))
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        logger.error("Scoring timed out after 2 minutes")
        return 'timeout', "Processing timed out after 2 minutes. Please try with fewer or smaller resumes."
    
    status, result = result_queue.get()
    if status == 'success':
        return 'success', result
    else:
        return 'error', result

# Generate PDF report
def generate_pdf_report(results, job_description):
    logger.debug("Generating PDF report")
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    elements.append(Paragraph("Resume Ranking Report", styles['Title']))
    elements.append(Paragraph(f"Job Description: {job_description[:200]}...", styles['Normal']))
    elements.append(Paragraph(" ", styles['Normal']))
    
    data = [['Rank', 'Resume', 'Score', 'Education', 'Experience', 'Skills', 'Responsibilities', 'Requirements', 'Cluster', 'Bias Indicators']]
    for i, result in enumerate(results, 1):
        data.append([
            i, result['resume'], f"{result['score']}%",
            f"{result['section_scores']['Education']}%",
            f"{result['section_scores']['Experience']}%",
            f"{result['section_scores']['Skills']}%",
            f"{result['jd_section_scores']['Responsibilities']}%",
            f"{result['jd_section_scores']['Requirements']}%",
            result['cluster'],
            '; '.join(result['bias'])
        ])
    
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    
    doc.build(elements)
    buffer.seek(0)
    logger.debug("PDF report generated")
    return buffer

# Generate CSV report
def generate_csv_report(results):
    logger.debug("Generating CSV report")
    output = StringIO()
    writer = csv.writer(output)
    
    headers = ['Rank', 'Resume', 'Score', 'Education', 'Experience', 'Skills', 'Responsibilities', 'Requirements', 'Cluster', 'Bias Indicators', 'JD Feedback', 'Keyword Density', 'ATS Issues', 'Length Feedback']
    writer.writerow(headers)
    
    for i, result in enumerate(results, 1):
        writer.writerow([
            i,
            result['resume'],
            f"{result['score']}%",
            f"{result['section_scores']['Education']}%",
            f"{result['section_scores']['Experience']}%",
            f"{result['section_scores']['Skills']}%",
            f"{result['jd_section_scores']['Responsibilities']}%",
            f"{result['jd_section_scores']['Requirements']}%",
            result['cluster'],
            '; '.join(result['bias']),
            result['jd_feedback'],
            f"{result['keyword_density'][0]}% - {result['keyword_density'][1]}",
            '; '.join(result['ats_issues']),
            '; '.join(result['length_feedback'][1])
        ])
    
    output.seek(0)
    logger.debug("CSV report generated")
    return output

# Middleware to check session timeout
@app.before_request
def check_session_timeout():
    if 'user_id' in session:
        last_activity = session.get('last_activity')
        if last_activity:
            if time.time() - last_activity > app.config['PERMANENT_SESSION_LIFETIME']:
                session.clear()
                return redirect(url_for('login'))
        session['last_activity'] = time.time()

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users.get(username)
        if user and check_password_hash(user['password'], password):
            session['user_id'] = username
            session['dark_mode'] = session.get('dark_mode', False)  # Load saved preference
            session['theme'] = session.get('theme', 'theme-light')  # Load saved theme
            session['custom_keywords'] = session.get('custom_keywords', '')  # Load saved preference
            session['last_activity'] = time.time()  # Initialize last activity
            return redirect(url_for('dashboard'))
        return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            return render_template('register.html', error='Username exists')
        users[username] = {'password': generate_password_hash(password)}
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Initialize results as an empty list instead of None to prevent TypeError
    results = session.get('results', [])
    if results is None:
        logger.warning("Session 'results' was None, initializing as empty list")
        results = []
        session['results'] = results
    
    error = None
    highlighted_jd = SAMPLE_JD
    jd_keywords = extract_keywords(SAMPLE_JD)
    jd_sections = extract_jd_sections(SAMPLE_JD)
    jd_keyword_suggestions = suggest_jd_keywords(jd_keywords)
    dark_mode = session.get('dark_mode', False)
    custom_keywords = session.get('custom_keywords', '')
    
    if request.method == 'POST':
        logger.debug("Received POST request for dashboard")
        jd_file = request.files.get('jd_file')
        jd_text = request.form.get('job_description')
        custom_keywords = request.form.get('custom_keywords', '')
        session['custom_keywords'] = custom_keywords  # Save preference
        
        if jd_file and jd_file.filename.endswith('.pdf'):
            logger.debug(f"JD file uploaded: {jd_file.filename}")
            jd_path = os.path.join(app.config['UPLOAD_FOLDER'], jd_file.filename)
            jd_file.save(jd_path)
            job_description = extract_text_from_pdf(jd_path)
            if job_description is None:
                error = "Failed to extract text from the JD PDF. Please ensure it contains readable text."
                return render_template('dashboard.html', results=results, error=error, highlighted_jd=highlighted_jd, jd_keywords=jd_keywords, jd_sections=jd_sections, jd_keyword_suggestions=jd_keyword_suggestions, sample_jd=SAMPLE_JD, dark_mode=dark_mode, custom_keywords=custom_keywords)
        elif jd_text:
            logger.debug("JD text provided")
            job_description = jd_text
        else:
            logger.debug("Using sample JD")
            job_description = SAMPLE_JD
        
        if not job_description:
            logger.error("No job description provided or extracted")
            error = "Job description is empty. Please provide a valid JD."
            return render_template('dashboard.html', results=results, error=error, highlighted_jd=highlighted_jd, jd_keywords=jd_keywords, jd_sections=jd_sections, jd_keyword_suggestions=jd_keyword_suggestions, sample_jd=SAMPLE_JD, dark_mode=dark_mode, custom_keywords=custom_keywords)
        
        resume_files = []
        resume_texts = []
        total_files = len(request.files.getlist('resumes')) + (1 if request.files.get('zip_file') else 0)
        processed_files = 0
        
        resumes = request.files.getlist('resumes')
        for resume in resumes:
            if resume and resume.filename.endswith('.pdf'):
                logger.debug(f"Processing resume: {resume.filename}")
                path = os.path.join(app.config['UPLOAD_FOLDER'], resume.filename)
                resume.save(path)
                text = extract_text_from_pdf(path)
                if text:
                    resume_texts.append(text)
                    resume_files.append(resume)
                else:
                    logger.warning(f"Failed to extract text from resume: {resume.filename}")
                processed_files += 1
                session['progress'] = (processed_files / total_files) * 100 if total_files > 0 else 0
        
        zip_file = request.files.get('zip_file')
        if zip_file and zip_file.filename.endswith('.zip'):
            logger.debug(f"Processing ZIP file: {zip_file.filename}")
            zip_path = os.path.join(app.config['UPLOAD_FOLDER'], zip_file.filename)
            zip_file.save(zip_path)
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(app.config['UPLOAD_FOLDER'])
                    zip_files = [f for f in zip_ref.namelist() if f.endswith('.pdf')]
                    total_files += len(zip_files)
                    for file_name in zip_files:
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
                        text = extract_text_from_pdf(file_path)
                        if text:
                            resume_texts.append(text)
                            resume_files.append(type('File', (), {'filename': file_name})())
                        else:
                            logger.warning(f"Failed to extract text from ZIP file resume: {file_name}")
                        processed_files += 1
                        session['progress'] = (processed_files / total_files) * 100 if total_files > 0 else 0
            except Exception as e:
                logger.error(f"Error processing ZIP file: {str(e)}")
                error = "Error processing ZIP file. Please ensure it contains valid PDFs."
                session['progress'] = 0
                return render_template('dashboard.html', results=results, error=error, highlighted_jd=highlighted_jd, jd_keywords=jd_keywords, jd_sections=jd_sections, jd_keyword_suggestions=jd_keyword_suggestions, sample_jd=SAMPLE_JD, dark_mode=dark_mode, custom_keywords=custom_keywords)
        
        session['progress'] = 0
        
        if not resume_texts:
            logger.error("No valid resumes extracted")
            error = "No valid resumes were uploaded. Please ensure the PDFs contain readable text."
            return render_template('dashboard.html', results=results, error=error, highlighted_jd=highlighted_jd, jd_keywords=jd_keywords, jd_sections=jd_sections, jd_keyword_suggestions=jd_keyword_suggestions, sample_jd=SAMPLE_JD, dark_mode=dark_mode, custom_keywords=custom_keywords)
        
        if resume_texts and job_description:
            logger.debug(f"Scoring {len(resume_texts)} resumes")
            status, result = score_resumes_with_timeout(job_description, resume_texts, resume_files, custom_keywords=custom_keywords, timeout=120)
            
            if status == 'success':
                results, highlighted_jd, jd_keywords, jd_sections, ats_issues, keyword_densities, jd_keyword_suggestions = result
                if not results:
                    logger.warning("No results returned from scoring")
                    error = "No valid resumes were processed. Please ensure the uploaded files contain readable text."
                else:
                    session['results'] = results
                    session['job_description'] = job_description
                    logger.debug("Results stored in session")
            elif status == 'timeout':
                error = result
            else:
                logger.error(f"Error during scoring: {result}")
                error = "An error occurred while processing the resumes: " + result
        else:
            logger.error("No resumes or job description provided")
            error = "Please upload at least one PDF or ZIP file and provide a job description."
    
    return render_template('dashboard.html', results=results, error=error, highlighted_jd=highlighted_jd, jd_keywords=jd_keywords, jd_sections=jd_sections, jd_keyword_suggestions=jd_keyword_suggestions, sample_jd=SAMPLE_JD, dark_mode=dark_mode, custom_keywords=custom_keywords)

@app.route('/toggle_dark_mode', methods=['POST'])
def toggle_dark_mode():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    session['dark_mode'] = not session.get('dark_mode', False)
    session['last_activity'] = time.time()
    return redirect(url_for('dashboard'))

@app.route('/set_theme', methods=['POST'])
def set_theme():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    theme = request.form.get('theme')
    if theme in ['theme-light', 'theme-dark', 'theme-blue']:
        session['theme'] = theme
    session['last_activity'] = time.time()
    return '', 200

@app.route('/download_report', methods=['POST'])
def download_report():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    session['last_activity'] = time.time()
    results = session.get('results', [])
    job_description = session.get('job_description', SAMPLE_JD)
    pdf_buffer = generate_pdf_report(results, job_description)
    return send_file(pdf_buffer, download_name='resume_ranking_report.pdf', as_attachment=True)

@app.route('/download_csv', methods=['POST'])
def download_csv():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    session['last_activity'] = time.time()
    results = session.get('results', [])
    csv_buffer = generate_csv_report(results)
    return send_file(
        BytesIO(csv_buffer.getvalue().encode('utf-8')),
        download_name='resume_ranking_report.csv',
        as_attachment=True,
        mimetype='text/csv'
    )

@app.route('/progress')
def progress():
    if 'user_id' not in session:
        return {'progress': 0}
    session['last_activity'] = time.time()
    return {'progress': session.get('progress', 0)}

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)