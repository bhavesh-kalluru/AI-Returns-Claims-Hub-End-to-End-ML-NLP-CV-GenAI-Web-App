# AI-Returns-Claims-Hub-End-to-End-ML-NLP-CV-GenAI-Web-App
ChatGPT said:  AI Returns &amp; Claims Hub is an end-to-end AI web app for e-commerce claims. It combines NLP, computer vision, ANN predictions, and Generative AI to classify issues, analyze photos, predict refunds, and draft replies. Includes SQL integration, Streamlit UI, and Power BI exports for professional analytics.


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
