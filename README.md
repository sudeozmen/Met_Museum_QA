# End-to-End RAG Architecture and Fine-Tuning System Design for The Met Collection

This project aims to build a Retrieval-Augmented Generation (RAG) system using information from The Metropolitan Museum of Art (The Met) collection. The main objective is to mitigate the "hallucination" tendency of Large Language Models (LLMs) when generating out-of-context information, and to analyze how models adapted to a specific domain (fine-tuned) interact with RAG.

## Project Overview
Generating accurate and reliable answers in domain-specific knowledge bases, such as large museum collections, is a challenging task. In this system, the performance of various component combinations (embedding, chunking) was measured, and the limitations of the models were tested.

## Data Sources
- **Primary Source:** The open-access `MetObjects.csv` dataset provided via Kaggle was used, specifically narrowed down to artworks marked as "Is Highlight" by The Met.
- **Secondary Source:** Art history texts extracted from The Met's official website via web scraping (BeautifulSoup) were included in the documents.

## System Architecture
- **Vector Database:** FAISS infrastructure was utilized for information retrieval.
- **Embedding Models:** `intfloat/multilingual-e5-base`, `BAAI/bge-large-en-v1.5`, and `sentence-transformers/all-MiniLM-L6-v2` were tested. (The `e5-base` model was preferred in the final system for multilingual query compatibility).
- **LLM Layer:** To prevent VRAM overflow, the Mistral-7B-Instruct-v0.2 model was integrated into the system in a 4-bit quantized (QLoRA) format using BitsAndBytesConfig.

## Key Findings and Performance
- **Retrieval Success:** The highest context retrieval performance was achieved using the BGE-Large embedding model combined with a Recursive chunking strategy, reaching a 96.52% Hit Rate and a 96.52% Recall score.
- **Best System Combination:** Instead of relying solely on Fine-Tuning or RAG independently, the combined architecture of **RAG + Fine-Tuned LLM** demonstrated the highest success rate.
- This peak architecture maximized the Exact Match score to 0.200 and the F1 score to 0.170, while keeping the hallucination rate at an absolute minimum.

## Installation and Reproducibility
Due to hardware constraints, the experiments were conducted on a Kaggle T4 x2 GPU environment. To ensure scientific reproducibility, all Python scripts related to data preprocessing, vector database construction, synthetic data generation (Data Augmentation), and the 4-bit QLoRA fine-tuning processes have been openly provided.

You can access the executable source codes and complete experimental workflows via the link below:
**[Kaggle Notebook - Met QA Final](https://www.kaggle.com/code/sudeozmen/met-qa-final)**

## Full Project Report
For a deep dive into the methodology, embedding combinations, hallucination analysis, and evaluation metrics, please refer to the detailed academic report (in Turkish):

**[View the Full Report PDF](./255172006_sude_ozmen_Met_Final_Report.pdf)**
