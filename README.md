# AI Recruitment Assistant System

## 🤖 Powered by Groq Llama-3.3-70B, LangChain, and Advanced RAG (Hybrid Search)

This system follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology to ensure a standardized and professional workflow.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [License](#license)

---

## 🎯 Overview

The AI Recruitment Assistant is an intelligent system designed to automate and streamline the recruitment process. It leverages state-of-the-art language models and retrieval-augmented generation (RAG) techniques to:

- Screen CVs against Job Descriptions objectively
- Generate relevant interview questions
- Evaluate candidate responses with bias detection

---

## ✨ Features

### Core Capabilities

- **📄 CV Screening**: Automated semantic matching between resumes and job descriptions using Hybrid Search (BM25 + Dense embeddings)
- **🎯 Match Scoring**: Objective and transparent scoring of candidate-job fit
- **❓ Interview Question Generation**: Context-aware question generation based on candidate qualifications
- **🔍 Answer Evaluation**: Systematic evaluation of candidate responses with bias checks
- **💬 Interactive Interface**: Gradio-based UI for easy interaction

### Technical Highlights

- **Hybrid Search**: Combines BM25 (lexical) and Dense (semantic) search for optimal retrieval
- **Advanced Embeddings**: Uses sentence-transformers for high-quality text embeddings
- **LLM-Powered**: Leverages Groq's Llama-3.3-70B for fast and accurate inference
- **Reranking**: Implements cross-encoder reranking for improved relevance

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Recruitment System                     │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   CV/PDF     │    │  Job Desc    │    │   Gradio     │   │
│  │   Parser     │    │   Processor  │    │     UI       │   │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘   │
│         │                   │                   │           │
│         └───────────┬───────┴───────────────────┘           │
│                     │                                       │
│            ┌────────▼────────┐                              │
│            │  Hybrid Search  │                              │
│            │  (BM25 + Dense) │                              │
│            └────────┬────────┘                              │
│                     │                                       │
│            ┌────────▼────────┐                              │
│            │   Reranker      │                              │
│            │  (Cross-Encoder)│                              │
│            └────────┬────────┘                              │
│                     │                                       │
│            ┌────────▼────────┐                              │
│            │  Llama-3.3-70B  │                              │
│            │    (via Groq)   │                              │
│            └─────────────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Jupyter Notebook or Google Colab
- Groq API Key

### Step-by-Step Installation

1. **Clone the repository** (if applicable)
   ```bash
   git clone <repository-url>
   cd ai-recruiter-system
   ```

2. **Install dependencies**
   ```bash
   pip install groq langchain pypdf gradio faiss-cpu sentence-transformers langchain-experimental rank_bm25
   ```

3. **Set up environment variables**
   ```bash
   export GROQ_API_KEY="your-groq-api-key"
   ```

---

## 📖 Usage

### Running the Notebook

1. Open `ai_recruiter_system.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells sequentially
3. The Gradio interface will launch automatically

### Basic Workflow

1. **Upload Documents**: Provide CV (PDF) and Job Description (text)
2. **Process & Analyze**: The system extracts and processes the text
3. **Get Results**: Receive match scores, interview questions, and evaluations

### Code Example

```python
# Initialize the system
from your_module import RecruitmentAssistant

assistant = RecruitmentAssistant(api_key="your-groq-key")

# Process CV and JD
result = assistant.evaluate(cv_path="candidate.pdf", jd_text="Job description here...")

# Get match score
print(f"Match Score: {result['score']}%")

# Generate interview questions
questions = assistant.generate_questions(result['context'])
```

---

## 📁 Project Structure

```
ai-recruiter-system/
├── README.md                           # This file
├── ai_recruiter_system.ipynb          # Main notebook
├── requirements.txt                    # Python dependencies
└── data/                               # Sample data (optional)
    ├── sample_cv.pdf
    └── sample_jd.txt
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| groq | Latest | LLM API access |
| langchain | Latest | LLM orchestration |
| pypdf | Latest | PDF parsing |
| gradio | Latest | Web interface |
| faiss-cpu | Latest | Vector similarity search |
| sentence-transformers | Latest | Text embeddings |
| langchain-experimental | Latest | Advanced RAG features |
| rank_bm25 | Latest | BM25 lexical search |

---

## 🔑 CRISP-DM Methodology

This project follows the CRISP-DM framework:

1. **Business Understanding**: Automate recruitment, reduce bias
2. **Data Understanding**: CVs (PDF) and Job Descriptions (text)
3. **Data Preparation**: Text extraction, cleaning, and chunking
4. **Modeling**: Hybrid search + LLM-based evaluation
5. **Evaluation**: Match scoring and bias detection
6. **Deployment**: Interactive Gradio interface

---

## ⚠️ Important Notes

- **API Key Required**: You need a valid Groq API key to use the LLM features
- **Memory Requirements**: Loading embedding models requires ~2GB RAM
- **First Run**: Initial model download may take several minutes

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 📞 Support

For issues and questions, please open an issue in the repository.

---

*Built with ❤️ using LangChain, Groq, and Advanced RAG techniques*
