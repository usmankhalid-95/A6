import streamlit as st
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
from langchain import PromptTemplate
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain import HuggingFacePipeline
import logging

# Set up Streamlit config
st.set_page_config(page_title="Usman AI Chat", page_icon="🤖", layout="wide")

# Device setup
device = torch.device("mps")

# Constants for file and model paths
VECTOR_DB_PATH = 'vector-store/nlp_db'
EMBEDDING_MODEL = 'hkunlp/instructor-base'
LANGUAGE_MODEL_ID = 'models/fastchat-t5-3b-v1.0'

# Define template for the prompt
chat_prompt = """
    Hello, I am RAGBot, a conversational AI designed to assist with questions using document-based information.
    I strive to ensure answers are accurate and well-cited.
    Context: {context}
    Question: {question}
    Answer:
""".strip()

# Initialize PromptTemplate
prompt_template = PromptTemplate.from_template(template=chat_prompt)

# Load embeddings using HuggingFace
embedding_model = HuggingFaceInstructEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": device}
)

# Load FAISS vector store
try:
    faiss_db = FAISS.load_local(
        folder_path=VECTOR_DB_PATH,
        embeddings=embedding_model,
        index_name='nlp'
    )
    retriever = faiss_db.as_retriever()
except Exception as e:
    logging.error(f"Error loading FAISS vector store: {e}")
    faiss_db, retriever = None, None

# Set up tokenizer and model for generation
tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_ID)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Configure quantization
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
device = torch.device("cpu")

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(
    LANGUAGE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

# Initialize pipeline
generation_pipeline = pipeline(
    task="text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    model_kwargs={"temperature": 0.3, "repetition_penalty": 1.5}
)

# HuggingFace pipeline for interaction
huggingface_pipe = HuggingFacePipeline(pipeline=generation_pipeline)

# Setup LangChain question generation
question_chain = LLMChain(
    llm=huggingface_pipe,
    prompt=CONDENSE_QUESTION_PROMPT,
    verbose=True
)

# QA Chain setup
qa_chain = load_qa_chain(
    llm=huggingface_pipe,
    chain_type='stuff',
    prompt=prompt_template,
    verbose=True
)

# Memory setup
conversation_memory = ConversationBufferWindowMemory(
    k=3,
    memory_key="chat_history",
    return_messages=True,
    output_key='answer'
)

# Conversational Retrieval Chain setup
if retriever:
    conv_chain = ConversationalRetrievalChain(
        retriever=retriever,
        question_generator=question_chain,
        combine_docs_chain=qa_chain,
        return_source_documents=True,
        memory=conversation_memory,
        verbose=True,
        get_chat_history=lambda history: history
    )

# Streamlit UI for interaction
st.title("🤖 Usman AI Chat")

st.write("Ask me anything, and I will answer based on available documents.")

# Input field for user query
user_query = st.text_input("What can I help you with today?", "")

# List of polite words
polite_words = [
    'hi', 'hello', 'hey', 'good morning', 'good evening', 'thanks', 'please', 'sorry'
]

# Button to submit query
if st.button("Ask"):
    if any(word in user_query.lower() for word in polite_words) and '?' not in user_query:
        response = "Hello! How can I assist you?"
        references = []
    else:
        try:
            result = conv_chain({"question": user_query})

            # Clean up the output answer
            response = result['answer'].strip()
            response = ' '.join([word for word in response.split() if word not in ['<pad>', '<eos>', '<ragbot>']])

            references = []

            # Add references if available
            for doc in result['source_documents']:
                meta = doc.metadata
                filename = meta['source'].split('/')[-1]
                page = meta['page'] + 1
                total_pages = meta['total_pages']
                references.append({"text": f"{filename} - page {page}/{total_pages}",
                                   "link": f"{filename}#page={page}"})

        except Exception as e:
            logging.error(f"Error generating response: {e}")
            response = "I couldn't process your request. Please try again later."
            references = []

    # Show response
    st.subheader("AI Response:")
    st.write(response)

    # Display references if available
    if references:
        st.subheader("Sources:")
        for ref in references:
            st.markdown(f"[{ref['text']}]({ref['link']})")

# Display conversation history if available
if "chat_history" in st.session_state and st.session_state.chat_history:
    st.subheader("Conversation History")
    for chat in st.session_state.chat_history:
        st.text(f"You: {chat['question']}")
        st.text(f"Bot: {chat['answer']}")
