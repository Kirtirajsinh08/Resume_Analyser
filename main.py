# ------------------------------------------
# 🔍 Necessary Libraries & Imports
# ------------------------------------------
# 🧠 Flask web framework for building the web app
from flask import Flask, request, render_template, send_from_directory

# 🤖 SentenceTransformer and cosine similarity for semantic comparison
from sentence_transformers import SentenceTransformer, util

# 🗂 For handling file paths and creating folders
import os

# 📄 Extracting text from PDF files
import PyPDF2

# 📃 Extracting text from .docx files
import docx2txt

# 🧹 Regular expressions for pattern matching (emails, phone, etc.)
import re

# ➗ NumPy for numerical operations (e.g., mean calculation)
import numpy as np

# 🧠 spaCy for skill extraction from job description (NLP processing)
import spacy

# 📊 For generating skill match bar charts
import matplotlib
matplotlib.use('Agg')  # 🖼 Disables GUI backend for matplotlib (useful for servers)

# 📈 Matplotlib's plotting functionality
import matplotlib.pyplot as plt

# 🧾 For converting .txt/.docx to PDF (reportlab)
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ------------------------------------------
# ⚙️ Flask App Initialization & Folder Setup
# ------------------------------------------
app = Flask(__name__)  # 🎯 Create Flask app instance

UPLOAD_FOLDER = 'uploads'  # 📁 Folder to store uploaded files
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # ✅ Create the folder if it doesn't exist
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # 🔗 Save path to Flask config

# ------------------------------------------
# 🤖 Load Pre-trained Models
# ------------------------------------------
# 🔍 Load Sentence Transformer for semantic similarity
model = SentenceTransformer('all-mpnet-base-v2')

# 🔎 Load spaCy NLP model for extracting skills
nlp = spacy.load("en_core_web_sm")

# ------------------------------------------
# 📄 extract_text: Extract text from resume files (.pdf, .docx, .txt)
# ------------------------------------------
def extract_text(file_path):
    if file_path.endswith('.pdf'):
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        return text

    elif file_path.endswith('.docx'):
        return docx2txt.process(file_path)

    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    return ""

# ------------------------------------------
# 📬 extract_contact_details: Extract name, email, mobile from resume
# ------------------------------------------
def extract_contact_details(text):
    text = text.replace('\n', ' ').replace('\r', ' ').strip()

    # 📧 Extract email using regex
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    email = email_match.group() if email_match else "Not found"

    # 📱 Extract mobile number using regex (handles +91 and spacing)
    mobile_match = re.search(r"(\+91[\-\s]*)?([6-9][0-9]{4})[\s\-]?([0-9]{5})", text)
    mobile = f"{mobile_match.group(2)} {mobile_match.group(3)}" if mobile_match else "Not found"

    # 🧑 Try to guess name from first 2 words in lines
    name = "Unknown"
    for line in text.splitlines():
        words = line.strip().split()
        if len(words) >= 2 and all(re.match(r"^[a-zA-Z]+$", w) for w in words[:2]):
            name = ' '.join(w.capitalize() for w in words[:2])
            break

    return name, email, mobile

# ------------------------------------------
# 🧠 extract_skills_from_jd: Extract keywords/skills from JD using spaCy
# ------------------------------------------
def extract_skills_from_jd(jd_text):
    doc = nlp(jd_text.lower())
    skills = set()

    for chunk in doc.noun_chunks:
        token = chunk.text.strip()
        if 2 <= len(token) <= 30 and re.match(r'^[a-zA-Z0-9\s\+\#\-\.]+$', token):
            skills.add(token.lower())

    return list(skills)

# ------------------------------------------
# ✂️ chunk_text: Divide text into smaller parts for similarity comparison
# ------------------------------------------
def chunk_text(text, chunk_size=3):
    lines = text.splitlines()
    return [
        " ".join(lines[i:i + chunk_size]).strip()
        for i in range(0, len(lines), chunk_size)
        if lines[i:i + chunk_size]
    ]

# ------------------------------------------
# 🎯 compute_skill_score: Match JD skills in resume & boost score with similarity
# ------------------------------------------
def compute_skill_score(resume_text, jd_skills, similarity_score):
    resume_text = resume_text.lower()
    matched = [skill for skill in jd_skills if skill in resume_text]

    base_score = (len(matched) / len(jd_skills)) * 10 if jd_skills else 0.0
    bonus = similarity_score * 0.035  # 🔥 Boost based on similarity
    return round(min(base_score + bonus, 10), 2)

# ------------------------------------------
# 📄 convert_to_pdf: Convert resume text (docx/txt) into PDF
# ------------------------------------------
def convert_to_pdf(text, output_path):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    text_object = c.beginText(40, height - 40)
    text_object.setFont("Helvetica", 10)

    for line in text.splitlines():
        text_object.textLine(line.strip())

    c.drawText(text_object)
    c.save()

# ------------------------------------------
# 📊 create_skill_score_chart: Bar chart of all candidates’ skill scores
# ------------------------------------------
def create_skill_score_chart(matches):
    names = [m['name'] for m in matches]
    scores = [m['skill_score'] for m in matches]

    plt.figure(figsize=(12, 5))
    plt.bar(names, scores, color='dodgerblue')
    plt.ylabel('Skill Score (out of 10)')
    plt.xlabel('Candidate')
    plt.xticks(rotation=90)
    plt.title('Skill Match Score by Candidate')
    plt.tight_layout()

    chart_filename = 'skill_score_chart.png'
    chart_path = os.path.join('static', chart_filename)
    plt.savefig(chart_path)
    plt.close()

    return chart_filename

# ------------------------------------------
# 📥 download_file: Route to allow downloading converted PDF resumes
# ------------------------------------------
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

# ------------------------------------------
# 🏠 home: Loads the HTML form on the homepage
# ------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

# ------------------------------------------
# 🚀 matcher: Core resume processing logic
# ------------------------------------------
@app.route('/resume-matcher', methods=['POST'])
def matcher():
    job_desc = request.form.get("job-description")  # 📝 JD from textarea
    resume_files = request.files.getlist("resume_files")  # 📤 Multiple resume uploads

    if not job_desc or not resume_files or resume_files[0].filename == '':
        return render_template('index.html', message="Please provide both job description and resume files.")

    job_embedding = model.encode(job_desc, convert_to_tensor=True)  # 🔍 JD embedding
    jd_skills = extract_skills_from_jd(job_desc)  # 🧠 Extracted JD skills
    top_matches = []

    for file in resume_files:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        text = extract_text(file_path)  # 📄 Resume text
        chunks = chunk_text(text)  # ✂️ Split into chunks
        chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
        similarities = util.cos_sim(job_embedding, chunk_embeddings)[0]
        top_3_avg = round(np.mean(sorted(similarities.tolist(), reverse=True)[:3]) * 100, 2)  # 📈 Top 3 chunk avg

        name, email, mobile = extract_contact_details(text)
        skill_score = compute_skill_score(text, jd_skills, top_3_avg)

        # 📎 Generate PDF file name
        pdf_filename = filename.rsplit('.', 1)[0] + '.pdf'
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)

        # 🔄 Convert if not already PDF
        if not filename.endswith('.pdf'):
            convert_to_pdf(text, pdf_path)
        else:
            pdf_filename = filename

        top_matches.append({
            "name": name,
            "email": email,
            "mobile": mobile,
            "similarity": top_3_avg,
            "skill_score": skill_score,
            "pdf_filename": pdf_filename
        })

    # 🏆 Sort by similarity
    top_matches = sorted(top_matches, key=lambda x: x['similarity'], reverse=True)

    # 📊 Create chart
    chart_filename = create_skill_score_chart(top_matches)

    # 📋 Render results
    return render_template('index.html', top_matches=top_matches, chart_path=chart_filename)

# ------------------------------------------
# 🔁 Start the Flask development server
# ------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
