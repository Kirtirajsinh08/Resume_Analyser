# 🤖 Smart Resume Matcher
🚀 A powerful **AI-driven Resume Matcher** that reads multiple resumes and intelligently compares them against a given **Job Description** using **Natural Language Processing (NLP)** and **Semantic Similarity**.

📌 Table of Contents
- [✨ Features](#-features)
- [🧠 How it Works](#-how-it-works)
- [🧰 Tech Stack](#-tech-stack)
- [🚀 Setup Instructions](#-setup-instructions)
- [📂 Project Structure](#-project-structure)
- [📝 Future Improvements](#-future-improvements)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)


✨ Features
✅ Upload multiple resumes in `.pdf`, `.docx`, or `.txt` format  
✅ Automatically extract **contact info** like name, email, and phone number  
✅ Dynamically extract **skills from Job Description** using spaCy NLP  
✅ Compute:
  - 🔹 **Semantic Similarity** score using Sentence Transformers
  - 🔹 **Skill Match Score** (out of 10) based on resume contents  
✅ Convert resumes to **PDF** if not already in PDF  
✅ Visualize results in a clean, ranked **table view** with:
  - 📊 **Progress bars** for similarity
  - 📈 **Bar chart** for skill scores
  - 📎 **Download buttons** for each resume  
✅ Beautiful **dark-themed UI** with Bootstrap & custom CSS


🧠 How it Works
1. Paste a job description 📝
2. Upload resumes 📤
3. Each resume is:
   - Parsed and chunked
   - Compared with JD using semantic embeddings
   - Analyzed for skill overlap
   - Scored and ranked
4. Results are visualized with charts and download links


🧰 Tech Stack
| Technology                  | Description                           |
|-----------------------------|---------------------------------------|
| 🐍 Python                  | Core language                          |
| 🧪 Flask                   | Lightweight web framework              |
| 🧠 SentenceTransformers    | For semantic similarity (BERT-based)   |
| 📚 spaCy                   | NLP for skill extraction               |
| 📄 PyPDF2                  | PDF parsing                            |
| 📝 docx2txt                | Word document reading                  |
| 📈 Matplotlib              | Data visualization (bar charts)        |
| 📦 Bootstrap               | UI styling                             |
| 🎨 Custom CSS              | Dark mode styling                      |


🚀 Setup Instructions

1️⃣ Clone the Repository
git clone https://github.com/your-username/resume-matcher.git
cd resume-matcher

2️⃣ Create Virtual Environment
python -m venv .venv
source .venv/bin/activate     # On Windows: .venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt
📝 Make sure en_core_web_sm spaCy model is installed:
python -m spacy download en_core_web_sm

4️⃣ Run the App
python main.py

Then open your browser and visit:
http://127.0.0.1:5000

📂 Project Structure
resume-matcher/
│

├── uploads/              # 📤 Uploaded resumes (auto-created)

├── static/

│   ├── styles.css        # 🎨 Custom dark theme

│   └── skill_score_chart.png  # 📊 Chart output

├── templates/
│   └── index.html        # 🖼️ Main UI

├── main.py               # 🧠 Core logic

├── requirements.txt      # 📦 Python packages

└── README.md             # 📘 This file



📝 Future Improvements
 1 Add PDF preview before download
 2 Allow exporting all results as CSV
 3 Improve skill-matching using synonym detection (e.g. “ML” ≈ “Machine Learning”)
 4 Support for multi-page JD or resumes
 5 User authentication for saving results

🤝 Contributing
Contributions, issues, and feature requests are welcome!
Feel free to fork the project and submit a PR 🔧

📄 License
This project is licensed under the MIT License.
Feel free to use it for learning or real-world applications.

🙏 Acknowledgements
Sentence Transformers by UKP Lab

spaCy for NLP

ReportLab for PDF generation

⚡ Built with passion by Kirtirajsinh Parmar 💙
If you'd like, I can generate the `requirements.txt` for you too, or help with setting up a GitHub repo. Let me know!
Mobile : +91 7043343119 
