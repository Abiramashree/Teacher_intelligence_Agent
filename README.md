# Teacher Intelligence Agent

## Overview

The Teacher Intelligence Agent is a Retrieval-Augmented Generation (RAG) system built to analyze tutoring transcripts and automatically generate actionable learning insights about students — their understanding, misconceptions, emotional tone, and next learning steps.

It combines:

🧠 Semantic retrieval using FAISS vector search

💬 Groq LLM (Llama 3) for contextual summarization

🧩 LangChain Tools for reasoning and automation

📄 Automated PDF report generation for educators

This project demonstrates how AI can empower tutors, educators, and learning platforms to better understand learners and personalize instruction — all powered by a custom-built, transparent RAG pipeline.

## Features

RAG + FAISS Vector Store — Retrieve relevant transcript chunks from large tutoring logs

EmbeddingPipeline — SentenceTransformer-based text chunking and embeddings

Groq LLM Summarization — Generates faithful, concise learning summaries

Insights Agent — Multi-tool agent using LangChain Tools and OpenAI

Automatic PDF Reports — Creates structured “Student Insights” reports for educators

Fully Modular — Plug-and-play architecture for any domain (education, customer support, etc.)

## Tech Stack

Python (LangChain, FAISS, SentenceTransformers)

LLMs: Groq (llama-3.1-8b-instant), OpenAI (gpt-4o-mini)

Libraries: LangChain, LangChain-Groq, ReportLab, dotenv, Pydantic

Tools: RAG Search + PDF Generator (custom LangChain Tools)

Storage: FAISS Vector Store

## Key Components
### Data Injestion and Embedding Pipeline

Splits documents into contextual chunks and embeds them using SentenceTransformer.

### FaissVectorStore

Stores and retrieves embeddings efficiently for semantic search.

### RAGSearch

Retrieves top chunks and summarizes with a Groq LLM.

### TutoringInsightsAgent

A tool-using LangChain agent that: Analyzes transcripts for understanding, misconceptions, and sentiment. Suggests targeted learning actions. Optionally exports insights as a PDF report


### Output (sample PDF fields):

Topics covered

Student understanding: medium

Misconceptions: Confused between heat and temperature

Suggested activities: Interactive simulations, real-life examples

Sentiment: Neutral

Confidence: 0.6

Bloom’s Level: Understand


## Use Cases

EdTech Platforms: Personalized learning analytics

Tutors & Teachers: Track student progress automatically

Learning Analytics Research: Evaluate engagement & comprehension trends

Student Support Teams: Identify confusion early

## Getting Started
git clone

cd tutoring-insights-agent

pip install -r requirements.txt


## Set your environment variables:

OPENAI_API_KEY=your_openai_key

GROQ_API_KEY=your_groq_key
