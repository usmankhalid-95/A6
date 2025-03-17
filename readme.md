# RAG-AI Chatbot: Intelligent Retrieval-Augmented Generation for Q&A  

## Overview  
This project implements a **Retrieval-Augmented Generation (RAG)** chatbot that enhances AI-driven responses with document-based knowledge retrieval. The chatbot is designed to answer user queries based on preloaded documents, ensuring responses are **accurate, contextual, and well-supported by retrieved information**.  

## Features  
- **Advanced Retrieval Mechanism**: Uses **semantic search** and **vector embeddings** for efficient document retrieval.  
- **Intelligent Response Generation**: Leverages **state-of-the-art transformer models** to generate responses grounded in retrieved documents.  
- **Context-Aware Q&A**: Ensures responses align with user queries while avoiding hallucinations.  
- **Source Referencing**: Displays retrieved documents to enhance transparency and trust.  

## Technology Stack  
- **Retrieval Model**: `hkunlp/instructor-base` (Hugging Face)  
- **Vector Store**: `FAISS` for fast and scalable similarity search  
- **Generator Model**: Transformer-based model using **Hugging Face Pipelines**  
- **Framework**: LangChain for seamless integration of retrieval and generation components  

## Challenges & Solutions  
| **Issue** | **Description** | **Solution** |  
|-----------|---------------|------------|  
| **Unrelated Responses** | Model may return off-topic results. | Improve embedding model training and filtering techniques. |  
| **Hallucination** | Model generates incorrect or non-existent facts. | Use hybrid retrieval and fact-checking mechanisms. |  
| **Context Loss** | Model struggles to maintain conversation flow. | Implement memory-based context tracking. |  
| **Scalability** | Performance drops with large datasets. | Optimize indexing and use distributed vector databases. |  

## Application Demonstration  
A web-based chat interface allows users to interact with the chatbot. Users can ask **questions**, and the model will retrieve **relevant documents** before generating well-informed responses.  

For instance, if a user asks:  
❓ *"What experience do you have in AI?"*  
💬 The chatbot will extract relevant details from provided documents and respond accordingly, along with references to the retrieved sources.  

### Website Screenshot  
Here’s how the chatbot interface looks:  

![Website Screenshot](website.png)  

## Output Example  
The chatbot generates responses based on retrieved data. Here’s a sample JSON output:  

```json
{
  "query": "What is your experience in AI?",
  "retrieved_documents": [
    "Document 2: usman_portfolio.pdf"
  ],
  "response": "I have worked on multiple AI projects focusing on NLP and deep learning. My experience includes developing AI-powered chatbots and RAG-based applications."
}
```

### 📄 [View Full JSON Output](output.json)

Clone the repository:  
   ```bash
   git clone https://github.com/usmankhalid-95/A6.git
   ```  
## Future Enhancements  
- Implement **real-time document ingestion** for dynamic updates.  
- Optimize retrieval with **hybrid search (dense + keyword-based retrieval).**  
- Introduce **multi-modal capabilities** for image-based document retrieval.  


### **Note on System Limitations**  
⚠️ **Important:** While running the chatbot, I encountered system limitations due to memory constraints on my **M1 MacBook**. Specifically, when attempting to answer the question:  

**"What specific research interests or academic goals do you hope to achieve during your time as a master’s student?"**  

My system **crashed after approximately 244 minutes** of processing due to an **out-of-memory (OOM) error**. The error message displayed was:  

```
RuntimeError: MPS backend out of memory (MPS allocated: 9.04 GB, other allocations: 3.92 MB, max allowed: 9.07 GB). Tried to allocate 68.66 MB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).
```

This happened consistently after several attempts, and the issue was likely caused by **high computational demands** exceeding the available memory on my machine.

#### **Potential Solutions (For Future Improvement)**  
- **Optimize model parameters** to reduce memory consumption.  
- **Use a cloud-based GPU** for better resource allocation.  
- **Experiment with model quantization** to decrease memory load.  
- **Adjust the MPS (Metal Performance Shaders) memory allocation** by tweaking `PYTORCH_MPS_HIGH_WATERMARK_RATIO` for better memory management.  