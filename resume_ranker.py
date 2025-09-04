# Add these changes to your resume_ranker.py file:

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
logging.basicConfig(level=logging.INFO if os.environ.get('FLASK_ENV') == 'production' else logging.DEBUG)
logger = logging.getLogger(__name__)

# Download NLTK data with error handling
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.error(f"Error downloading NLTK data: {e}")

app = Flask(__name__)

# Use environment variable for secret key in production
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Configuration
UPLOAD_FOLDER = '/tmp/uploads' if os.environ.get('RENDER') else 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'zip'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
except Exception as e:
    logger.error(f"Error creating upload folder: {e}")

# Add this at the end of your file, replace the existing if __name__ == '__main__' block:
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug)
