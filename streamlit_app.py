import os
import streamlit as st
import tempfile
import pandas as pd
from pathlib import Path
from typing import List, Optional

from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamlitCallbackHandler

# Configure page
st.set_page_config(
    page_title="Nuclear Documentation Assistant",
    page_icon="☢️",
    layout="wide"
)

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.embeddings = OpenAIEmbeddings(
            openai_api_base=os.getenv('OPENAI_API_BASE', "https://helixmind.online/v1"),
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )

    def process_documents(self, uploaded_files: List[tempfile.NamedTemporaryFile]) -> Optional[Chroma]:
        """Process uploaded documents and create a vector store."""
        documents = []
        
        for file in uploaded_files:
            temp_file_path = Path(file.name)
            
            if temp_file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(temp_file_path))
                documents.extend(loader.load())
            elif temp_file_path.suffix.lower() == '.txt':
                loader = TextLoader(str(temp_file_path))
                documents.extend(loader.load())
                
        if not documents:
            return None
            
        split_docs = self.text_splitter.split_documents(documents)
        
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        return vectorstore

def initialize_chat_engine(vectorstore: Chroma) -> ConversationalRetrievalChain:
    """Initialize the chat engine with the vector store."""
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0,
        streaming=True,
        openai_api_base=os.getenv('OPENAI_API_BASE', "https://helixmind.online/v1"),
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        verbose=True
    )
    
    return conversation_chain

# Sidebar configuration
with st.sidebar:
    st.title("Nuclear Documentation Assistant")
    st.markdown("### Settings")
    
    uploaded_files = st.file_uploader(
        "Upload IAEA/NUREG Documents (PDF/TXT)",
        accept_multiple_files=True,
        type=["pdf", "txt"]
    )
    
    if uploaded_files:
        temp_files = []
        for file in uploaded_files:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(file.getvalue())
            temp_files.append(temp_file)
            
        # Process documents
        doc_processor = DocumentProcessor()
        with st.spinner("Processing documents..."):
            vectorstore = doc_processor.process_documents(temp_files)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.conversation = initialize_chat_engine(vectorstore)
                st.success("Documents processed successfully!")
            
        # Cleanup temporary files
        for temp_file in temp_files:
            temp_file.close()
            os.unlink(temp_file.name)

# Main chat interface
st.title("Nuclear Documentation Query System")

if not st.session_state.conversation:
    st.info("Please upload IAEA or NUREG documents to begin.")
else:
    # Chat interface
    if prompt := st.chat_input("Ask about the nuclear documentation"):
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            response = st.session_state.conversation(
                {"question": prompt},
                callbacks=[st_callback]
            )
            
            # Display sources if available
            if response.get("source_documents"):
                with st.expander("View Sources"):
                    for i, doc in enumerate(response["source_documents"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.markdown(doc.page_content)
                        st.markdown("---")
            
            st.session_state.chat_history.append(
                {"role": "user", "content": prompt}
            )
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response["answer"]}
            )

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])