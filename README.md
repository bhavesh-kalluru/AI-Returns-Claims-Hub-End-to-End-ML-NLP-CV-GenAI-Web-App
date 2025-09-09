# AI Returns & Claims Hub — End-to-End ML/NLP/CV + GenAI Web App

## Overview
**AI Returns & Claims Hub** is a full-stack AI project simulating a real-world e-commerce returns and claims system.  
It combines **Machine Learning**, **Deep Learning**, **NLP**, **Computer Vision**, and **Generative AI** into a single pipeline with a professional **Streamlit** web UI and **Power BI** analytics integration.  

This project demonstrates how to take AI ideas from research → data engineering → modeling → deployment → visualization, making it portfolio-ready for showcasing **AI/GenAI Engineering skills**.

---

## Features
- **Database (SQLAlchemy + SQLite)**  
  Structured schema for customers, products, and claims, with idempotent migrations.
- **NLP Pipeline**  
  Issue classification, sentiment analysis, and key phrase extraction.
- **Computer Vision Module**  
  Photo quality checks (blur, brightness, contrast) and heuristic damage scoring.
- **ANN Predictor**  
  Neural network estimating refund probability based on sentiment, photo evidence, and damage features.
- **Generative AI (OpenAI API)**  
  Drafts concise claim summaries and empathetic customer replies.
- **Web Application (Streamlit)**  
  Intuitive UI with one-click buttons to create/seed tables, run NLP/ANN/GenAI, attach photos, and visualize claims.
- **Analytics Export (Power BI)**  
  SQL view + CSV exports for dashboards on refund probability, claim categories, and sentiment trends.

---

## Quick Start
```bash
git clone <your-repo-url>.git
cd ai-returns-hub

##create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# add your OpenAI API key in .env
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-xxxx

# launch the app
python3 -m streamlit run app/web/claims_app.py

Tech Stack
Languages/Frameworks: Python, Streamlit, SQLAlchemy
ML/DL/NLP: scikit-learn, VADER, TF-IDF, ANN (MLPRegressor)
Computer Vision: OpenCV, Pillow
GenAI: OpenAI API (GPT models)
Analytics: Power BI (via CSV exports + SQL views)

Why This Project?
This project demonstrates my ability to:
Frame a real-world business problem,
Engineer data pipelines,
Apply NLP, ANN, and CV models,
Integrate Generative AI for user-facing outputs,
Build a working web app with database persistence,
Deliver analytics-ready datasets.

Author
Bhavesh Kalluru
Actively seeking full-time AI/GenAI Engineer roles in the USA.
This repository showcases my capability to design, build, and deploy complete AI solutions.
