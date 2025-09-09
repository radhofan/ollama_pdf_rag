"""
Streamlit application for PDF-based Retrieval-Augmented Generation (RAG) using Ollama + LangChain.

This application allows users to upload a PDF, process it,
and then ask questions about the content using a selected language model.
"""

import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import warnings
import base64
import requests
from typing import List, Tuple, Dict, Any, Optional

# Suppress torch warning
warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# Set protobuf environment variable to avoid error messages
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Define persistent directory for ChromaDB
PERSIST_DIRECTORY = os.path.join("data", "vectors")

# Streamlit page configuration
st.set_page_config(
    page_title="Ollama PDF RAG Streamlit UI",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

def get_ngrok_url():
    """Get ngrok URL from environment or fallback to default"""
    # Try to get from environment variable first
    ngrok_url = os.environ.get("NGROK_URL")
    if ngrok_url:
        return ngrok_url.rstrip('/')
    
    # Try Streamlit secrets
    try:
        ngrok_url = st.secrets.get("NGROK_URL")
        if ngrok_url:
            return ngrok_url.rstrip('/')
    except:
        pass
    
    # Fallback to hardcoded (update this with your actual ngrok URL)
    return "https://15a386eab580.ngrok-free.app"

def test_ollama_connection(base_url: str) -> Tuple[bool, str]:
    """Test if Ollama is accessible via the provided URL"""
    try:
        # Add ngrok bypass headers for free plan
        headers = {
            'ngrok-skip-browser-warning': 'true',
            'User-Agent': 'StreamlitApp/1.0',
            'Accept': 'application/json'
        }
        
        # Test with direct HTTP request first
        response = requests.get(f"{base_url}/api/tags", headers=headers, timeout=30)
        
        if response.status_code == 200:
            models_data = response.json()
            logger.info(f"Successfully connected to Ollama. Models: {models_data}")
            return True, "Connected successfully"
        else:
            error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
            logger.error(f"HTTP request failed: {error_msg}")
            return False, error_msg
            
    except requests.exceptions.Timeout:
        error_msg = "Connection timeout - ngrok tunnel may be slow or down"
        logger.error(error_msg)
        return False, error_msg
    except requests.exceptions.ConnectionError:
        error_msg = "Connection refused - check if ngrok tunnel is running"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Connection error: {str(e)}"
        logger.error(f"Failed to connect to Ollama at {base_url}: {e}")
        return False, error_msg

def get_available_models(base_url: str) -> Tuple[List[str], str]:
    """Get available models from Ollama"""
    try:
        headers = {
            'ngrok-skip-browser-warning': 'true',
            'User-Agent': 'StreamlitApp/1.0',
            'Accept': 'application/json'
        }
        
        response = requests.get(f"{base_url}/api/tags", headers=headers, timeout=30)
        
        if response.status_code == 200:
            models_data = response.json()
            if 'models' in models_data and models_data['models']:
                model_names = [model.get('name', model.get('model', '')) for model in models_data['models']]
                return model_names, "Success"
            else:
                return [], "No models found"
        else:
            return [], f"HTTP {response.status_code}: {response.text[:200]}"
            
    except Exception as e:
        return [], f"Error getting models: {str(e)}"

# Get base URL
base_url = get_ngrok_url()

def create_vector_db(file_upload) -> Chroma:
    """
    Create a vector database from an uploaded PDF file.
    """
    logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()

    try:
        path = os.path.join(temp_dir, file_upload.name)
        with open(path, "wb") as f:
            f.write(file_upload.getvalue())
            logger.info(f"File saved to temporary path: {path}")
            
        loader = UnstructuredPDFLoader(path)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        logger.info("Document split into chunks")

        # Create embeddings - LangChain handles the HTTP requests internally
        try:
            embeddings = OllamaEmbeddings(
                model="nomic-embed-text", 
                base_url=base_url
            )
            # Test embeddings by embedding a small text
            test_embedding = embeddings.embed_query("test")
            logger.info("Embeddings model working correctly")
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            raise Exception(f"Cannot connect to Ollama embeddings model. Error: {e}")

        # Ensure persist directory exists
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY,
            collection_name=f"pdf_{hash(file_upload.name)}"
        )
        logger.info("Vector DB created with persistent storage")
        
    except Exception as e:
        logger.error(f"Error creating vector DB: {e}")
        raise
    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Temporary directory {temp_dir} removed")
    
    return vector_db

def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    """
    Process a user question using the vector database and selected language model.
    """
    logger.info(f"Processing question: {question} using model: {selected_model}")
    
    try:
        # Initialize LLM - LangChain handles HTTP requests internally
        llm = ChatOllama(
            model=selected_model, 
            base_url=base_url
        )
        
        # Query prompt template
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate 2
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}""",
        )

        # Set up retriever
        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(), 
            llm,
            prompt=QUERY_PROMPT
        )

        # RAG prompt template
        template = """Answer the question based ONLY on the following context:
        {context}
        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        # Create chain
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        response = chain.invoke(question)
        logger.info("Question processed and response generated")
        return response
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return f"Error processing your question: {str(e)}. Please check your Ollama connection and try again."

@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    """Extract all pages from a PDF file as images."""
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages

def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    """Delete the vector database and clear related session state."""
    logger.info("Deleting vector DB")
    if vector_db is not None:
        try:
            vector_db.delete_collection()
            
            # Clear session state
            st.session_state.pop("pdf_pages", None)
            st.session_state.pop("file_upload", None)
            st.session_state.pop("vector_db", None)
            
            st.success("Collection and temporary files deleted successfully.")
            logger.info("Vector DB and related session state cleared")
            st.rerun()
        except Exception as e:
            st.error(f"Error deleting collection: {str(e)}")
            logger.error(f"Error deleting collection: {e}")
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")

def main() -> None:
    """Main function to run the Streamlit application."""

    def image_to_base64(path: str) -> str:
        """Convert image to base64, with error handling"""
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return base64.b64encode(f.read()).decode()
        except Exception as e:
            logger.warning(f"Could not load logo: {e}")
        return ""

    # Display header (with or without logo)
    logo_path = "data/assets/Logo.png"
    logo_base64 = image_to_base64(logo_path)
    
    if logo_base64:
        st.markdown(
            f"""
            <h3 style="display: flex; align-items: center; gap: 8px; margin: 0;">
                <img src="data:image/png;base64,{logo_base64}" width="32">
                Computing Lab Chatbot
            </h3>
            <hr style="border: 1px solid #ddd; margin: 0.5em 0;">
            """,
            unsafe_allow_html=True,
        )
    else:
        st.title("🤖 Computing Lab Chatbot")
        st.markdown("---")

    # Display current ngrok URL
    st.info(f"🔗 Connecting to Ollama at: {base_url}")

    # Test connection
    is_connected, connection_msg = test_ollama_connection(base_url)
    
    if is_connected:
        st.success(f"✅ {connection_msg}")
    else:
        st.error(f"❌ Connection failed: {connection_msg}")
        st.markdown("""
        ### Troubleshooting Steps:
        1. **Check ngrok tunnel**: Make sure ngrok is running with `ngrok http 11434 --request-header-add='ngrok-skip-browser-warning:true'`
        2. **Verify Ollama**: Test locally with `curl http://localhost:11434/api/tags`
        3. **Update URL**: Set the correct ngrok URL in environment variables or update the hardcoded URL in the code
        4. **Check firewall**: Ensure port 11434 is accessible
        """)
        return

    # Get available models
    available_models, models_msg = get_available_models(base_url)
    
    if not available_models:
        st.error(f"No models available: {models_msg}")
        st.info("Make sure you have models installed. Run: `ollama pull llama2` or `ollama pull mistral`")
        return

    # Create layout
    col1, col2 = st.columns([1.5, 2])

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None
    if "use_sample" not in st.session_state:
        st.session_state["use_sample"] = False

    # Model selection
    selected_model = col2.selectbox(
        f"Pick a model (Found {len(available_models)} models) ↓", 
        available_models,
        key="model_select"
    )

    # Add checkbox for sample PDF
    use_sample = col1.toggle(
        "Computing Lab PDF", 
        key="sample_checkbox"
    )
    
    # Clear vector DB if switching between sample and upload
    if use_sample != st.session_state.get("use_sample"):
        if st.session_state["vector_db"] is not None:
            try:
                st.session_state["vector_db"].delete_collection()
            except:
                pass
            st.session_state["vector_db"] = None
            st.session_state["pdf_pages"] = None
        st.session_state["use_sample"] = use_sample

    if use_sample:
        # Use the sample PDF
        sample_path = "data/pdfs/sample/computing-lab.pdf"
        if os.path.exists(sample_path):
            if st.session_state["vector_db"] is None:
                with st.spinner("Processing sample PDF..."):
                    try:
                        loader = UnstructuredPDFLoader(file_path=sample_path)
                        data = loader.load()
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
                        chunks = text_splitter.split_documents(data)
                        
                        embeddings = OllamaEmbeddings(
                            model="nomic-embed-text", 
                            base_url=base_url
                        )
                        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
                        
                        st.session_state["vector_db"] = Chroma.from_documents(
                            documents=chunks,
                            embedding=embeddings,
                            persist_directory=PERSIST_DIRECTORY,
                            collection_name="sample_pdf"
                        )
                        # Display PDF
                        with pdfplumber.open(sample_path) as pdf:
                            st.session_state["pdf_pages"] = [page.to_image().original for page in pdf.pages]
                    except Exception as e:
                        st.error(f"Error processing sample PDF: {e}")
        else:
            st.error("Sample PDF file not found.")
    else:
        # Regular file upload
        file_upload = col1.file_uploader(
            "Upload a PDF file ↓", 
            type="pdf", 
            accept_multiple_files=False,
            key="pdf_uploader"
        )

        if file_upload:
            if st.session_state["vector_db"] is None:
                with st.spinner("Processing uploaded PDF..."):
                    try:
                        st.session_state["vector_db"] = create_vector_db(file_upload)
                        st.session_state["file_upload"] = file_upload
                        # Extract and store PDF pages
                        with pdfplumber.open(file_upload) as pdf:
                            st.session_state["pdf_pages"] = [page.to_image().original for page in pdf.pages]
                    except Exception as e:
                        st.error(f"Error processing PDF: {e}")

    # Display PDF if pages are available
    if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
        # PDF display controls
        zoom_level = col1.slider(
            "Zoom Level", 
            min_value=100, 
            max_value=1000, 
            value=700, 
            step=50,
            key="zoom_slider"
        )

        # Display PDF pages
        with col1:
            with st.container(height=410, border=True):
                for page_image in st.session_state["pdf_pages"]:
                    st.image(page_image, width=zoom_level)

    # Delete collection button
    delete_collection = col1.button(
        "⚠️ Delete collection", 
        type="secondary",
        key="delete_button"
    )

    if delete_collection:
        delete_vector_db(st.session_state["vector_db"])

    # Chat interface
    with col2:
        message_container = st.container(height=500, border=True)

        # Display chat history
        for i, message in enumerate(st.session_state["messages"]):
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        # Chat input and processing
        if prompt := st.chat_input("Enter a prompt here...", key="chat_input"):
            try:
                # Add user message to chat
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with message_container.chat_message("user", avatar="😎"):
                    st.markdown(prompt)

                # Process and display assistant response
                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["vector_db"] is not None:
                            response = process_question(
                                prompt, st.session_state["vector_db"], selected_model
                            )
                            st.markdown(response)
                            # Add assistant response to chat history
                            st.session_state["messages"].append(
                                {"role": "assistant", "content": response}
                            )
                        else:
                            warning_msg = "Please upload a PDF file first."
                            st.warning(warning_msg)

            except Exception as e:
                st.error(f"Error: {e}", icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["vector_db"] is None:
                st.warning("Upload a PDF file or use the sample PDF to begin chat...")

if __name__ == "__main__":
    main()