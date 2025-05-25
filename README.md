# LangChain Study Project

This project is a collection of Python scripts demonstrating the use of the LangChain library and its main techniques for language model applications. The code examples cover various features such as prompt templates, sequential chains, memory, retrieval with embeddings, PDF document processing, and integration with Azure OpenAI.

## Main Topics Covered
- **Prompt Engineering:** How to create and use prompt templates for different tasks.
- **Chains:** Building sequential and conversational chains to process and transform user input step by step.
- **Memory:** Using buffer memory to maintain conversation context.
- **Retrieval Augmented Generation (RAG):** Loading documents (including PDFs), splitting text, generating embeddings, and retrieving relevant information using vector databases (FAISS).
- **Azure OpenAI Integration:** Connecting LangChain with Azure OpenAI deployments for both chat and embeddings.
- **Output Parsing:** Parsing and structuring LLM outputs using Pydantic models and JSON parsers.

## Example Files
- `langchain_prompt_template.py`: Prompt template usage.
- `langchain_simple_sequential_chain.py`: Simple sequential chain example.
- `langchain_buffer.py`: Conversation buffer memory example.
- `langchain_lcel.py`: LCEL (LangChain Expression Language) composition.
- `langchain_jsonparser.py`: Parsing LLM output to JSON with Pydantic.
- `langchain_retrieval.py`: Text retrieval with embeddings and FAISS.
- `langchain-retrieval_pdf.py`: Retrieval from PDF files.
- `langchain_complaint_analyser.py`: Complaint analysis pipeline.

## Requirements
- Python 3.10+
- See `requirements.txt` for all dependencies.

## Usage
1. Clone this repository.
2. Set up your `.env` file with your Azure OpenAI or OpenAI credentials.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run any script for the topic you want to study:
   ```
   python langchain_prompt_template.py
   ```

---
This project is for study and learning purposes. Feel free to explore and modify the scripts!
