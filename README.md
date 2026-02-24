# Corrective-Retrieval-Augmented-Generation-CRAG-System-using-LangGraph

An advanced Agentic RAG system that implements a **Corrective Retrieval-Augmented Generation (CRAG)** pipeline. This project utilizes **LangGraph** for workflow orchestration and local **Hugging Face** models to ensure high-accuracy responses without external API costs.

## 🎯 Project Overview
Standard RAG systems can sometimes provide inaccurate answers if the retrieved context is noisy. This CRAG system introduces a **Grader Node** to evaluate the relevance of documents before they reach the generation stage. If a document is deemed irrelevant, it is filtered out, ensuring the final answer is grounded only in accurate data.

## 🚀 Key Features
* **Graph-Based Logic**: Managed via **LangGraph**, enabling a modular "Retrieve -> Grade -> Generate" workflow.
* **Local Execution**: Uses `google/flan-t5-base` via Hugging Face, providing a secure and private AI environment.
* **Semantic Search**: Powered by **ChromaDB** and `all-MiniLM-L6-v2` embeddings for precise information retrieval.
* **Dynamic Knowledge Base**: Scrapes and indexes multiple technical sources for real-time query handling:
    * LLM Powered Autonomous Agents
    * Prompt Engineering Guide
    * Adversarial Attacks on LLMs

## 🛠️ Tech Stack
* **Frameworks**: LangChain, LangGraph
* **LLM**: Google Flan-T5
* **Vector Database**: ChromaDB
* **Embeddings**: Hugging Face Sentence-Transformers
* **Language**: Python 3.12

## ⚙️ Installation & Setup

**Step 1: Clone the Repository**
```bash
git clone [https://github.com/sujalrana-ai/Corrective-Retrieval-Augmented-Generation-CRAG-System-using-LangGraph.git](https://github.com/sujalrana-ai/Corrective-Retrieval-Augmented-Generation-CRAG-System-using-LangGraph.git)
cd Corrective-Retrieval-Augmented-Generation-CRAG-System-using-LangGraph
```
Step 2: Install Dependencies
```
pip install -r requirements.txt
```
Step 3: Run the System
```
python crag_pipeline.py
```
🧪 Example Output

Input Question: "What are the components of an LLM agent?"

Final Answer: "memory, planning and reflection mechanisms"
