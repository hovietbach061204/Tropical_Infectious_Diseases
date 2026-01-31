# 🦠 Tropical Infectious Disease Diagnosis System
### A Hybrid RAG & Fine-Tuned LLM Approach for Medical Diagnosis

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Stack](https://img.shields.io/badge/Tech-LangChain%20%7C%20Qdrant%20%7C%20Neo4j-green)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)

## 📖 Abstract
This project presents an advanced diagnositc system for tropical and infectious diseases (e.g., Dengue, Typhoid, Malaria) utilizing a **Hybrid Retrieval-Augmented Generation (RAG)** framework. By integrating vector search (**Qdrant**) with a knowledge graph (**Neo4j**), the system retrieves both semantic context and structured relationships (symptoms, locations, risk factors). The generation is handled by a fine-tuned **Qwen2.5-7B/Llama-3.1** model, optimized for medical reasoning.

## 🏗️ System Architecture
The system follows a modular pipeline designed to maximize context recall and diagnosis accuracy:
1.  **Data Ingestion:** Processing medical case reports using `Docling` for structured extraction.
2.  **Hybrid Retrieval:**
    * **Vector Search (Qdrant):** Dense retrieval using `nomic-embed-text` embeddings.
    * **Knowledge Graph (Neo4j):** Graph traversal to link symptoms with diseases and risk factors.
3.  **Generation:** A custom fine-tuned LLM (Qwen/Llama) generates evidence-based diagnoses.
4.  **Evaluation:** Automated benchmarking using **Ragas** and **DeepEval** (Faithfulness, Context Recall, Answer Correctness).

## 🚀 Key Features
* **Dual-Path Retrieval:** Combines vector similarity with graph-based reasoning to handle complex medical queries.
* **Domain-Specific Fine-Tuning:** Models were fine-tuned on a curated dataset of tropical disease case studies.
* **Hallucination Guardrails:** Implements RAG metrics to ensure answers are grounded in retrieved context.
* **Evaluation Pipeline:** Integrated `eeval_model.py` for continuous performance monitoring.

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.10+
* Neo4j Database (Local or AuraDB)
* Qdrant Instance (Docker or Cloud)

### 1. Clone the Repository
```bash
git clone [https://github.com/hovietbach061204/Tropical_Infectious_Diseases.git](https://github.com/hovietbach061204/Tropical_Infectious_Diseases.git)
cd Tropical_Infectious_Diseases
