from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer, util
import os
import PyPDF2
import docx2txt
import re
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = SentenceTransformer('all-mpnet-base-v2')


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


def extract_contact_details(text):
    text = text.replace('\n', ' ').replace('\r', ' ').strip()
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    email = email_match.group() if email_match else "Not found"
    mobile_match = re.search(r"(\+91[\-\s]*)?([6-9][0-9]{4})[\s\-]?([0-9]{5})", text)
    mobile = f"{mobile_match.group(2)} {mobile_match.group(3)}" if mobile_match else "Not found"
    name = "Unknown"
    for line in text.splitlines():
        words = line.strip().split()
        if len(words) >= 2 and all(re.match(r"^[a-zA-Z]+$", w) for w in words[:2]):
            name = ' '.join(w.capitalize() for w in words[:2])
            break
    return name, email, mobile


def chunk_text(text, chunk_size=3):
    lines = text.splitlines()
    return [" ".join(lines[i:i+chunk_size]).strip() for i in range(0, len(lines), chunk_size) if lines[i:i+chunk_size]]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/resume-matcher', methods=['POST'])
def matcher():
    job_desc = request.form.get("job-description")
    resume_files = request.files.getlist("resume_files")

    if not job_desc or not resume_files or resume_files[0].filename == '':
        return render_template('index.html', message="Please provide both job description and resume files.")

    job_embedding = model.encode(job_desc, convert_to_tensor=True)
    top_matches = []

    for file in resume_files:
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(save_path)
        text = extract_text(save_path)
        chunks = chunk_text(text)

        chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
        similarities = util.cos_sim(job_embedding, chunk_embeddings)[0]
        top_3_avg = round(np.mean(sorted(similarities.tolist(), reverse=True)[:3]) * 100, 2)

        name, email, mobile = extract_contact_details(text)
        top_matches.append({
            "name": name,
            "email": email,
            "mobile": mobile,
            "similarity": top_3_avg
        })

    top_matches = sorted(top_matches, key=lambda x: x['similarity'], reverse=True)
    return render_template('index.html', top_matches=top_matches)


if __name__ == '__main__':
    app.run(debug=True)
