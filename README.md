# 🦠 Tropical Infectious Disease Diagnosis System
### A Hybrid RAG & Fine-Tuned LLM Approach for Medical Diagnosis

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Stack](https://img.shields.io/badge/Tech-LangChain%20%7C%20Qdrant%20%7C%20Neo4j-green)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📖 Abstract
This project presents an advanced diagnostic system for tropical and infectious diseases (e.g., Dengue, Typhoid, Malaria) utilizing a **Hybrid Retrieval-Augmented Generation (RAG)** framework. By integrating vector search (**Qdrant**) with a knowledge graph (**Neo4j**), the system retrieves both semantic context and structured relationships (symptoms, locations, risk factors). The generation is handled by a fine-tuned **Qwen2.5-7B/Llama-3.1** model, specifically optimized for medical reasoning and evidence-based diagnosis.

## 🏗️ System Architecture
The system follows a modular pipeline designed to maximize context recall and diagnosis accuracy:
1.  **Data Ingestion:** Processing medical case reports using `Docling` for structured extraction.
2.  **Hybrid Retrieval:**
    * **Vector Search (Qdrant):** Dense retrieval using `nomic-embed-text` embeddings.
    * **Knowledge Graph (Neo4j):** Graph traversal to link symptoms with diseases and risk factors.
3.  **Generation:** A custom fine-tuned LLM (Qwen/Llama) generates diagnoses with "Diagnosis" and "Explanation" blocks.
4.  **Evaluation:** Automated benchmarking using **Ragas** and **DeepEval** (Faithfulness, Context Recall, Answer Correctness).

---

## 🛠️ Installation & Setup

### 1. Prerequisites
* Python 3.10+
* [Git LFS](https://git-lfs.com/) (Required for large model weights)
* Neo4j Database (Local Desktop or AuraDB)
* Qdrant Instance (Docker or Cloud)

### 2. Clone the Repository
```bash
git clone [https://github.com/hovietbach061204/Tropical_Infectious_Diseases.git](https://github.com/hovietbach061204/Tropical_Infectious_Diseases.git)
cd Tropical_Infectious_Diseases
