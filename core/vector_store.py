# Import operating system utilities
import os 

# Chroma vector database
from langchain_chroma import Chroma 

# HuggingFace embedding model
from langchain_community.embeddings import HuggingFaceEmbeddings

# Text splitter for chunking large text
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Document object used by LangChain
from langchain_core.documents import Document


# Folder where vector database will be stored
CHROMA_DIR = "vector_db"

# Name of the collection inside Chroma DB
COLLECTION_NAME = "meeting_transcript"

# Embedding model name from HuggingFace
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------
# Load embedding model
# ---------------------------------------------------
def get_embedding():
    """
    Loads HuggingFace embedding model.

    This model converts text into vectors (numerical embeddings)
    so semantic similarity search can be performed.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,

        # Use CPU for embeddings
        model_kwargs={"device": "cpu"}
    )


# ---------------------------------------------------
# Build vector store from transcript
# ---------------------------------------------------
def build_vector_store(transcript: str) -> Chroma:

    print("Building vector store")

    # Create text splitter
    # chunk_size = max characters per chunk
    # chunk_overlap = overlap between chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    # Split transcript into smaller chunks
    chunks = splitter.split_text(transcript)

    # Convert chunks into LangChain Documents
    docs = [
        Document(
            page_content=chunk,

            # Store metadata for tracking chunk index
            metadata={"chunk_index": i}
        )
        for i, chunk in enumerate(chunks)
    ]

    # Load embedding model
    embeddings = get_embedding()

    # Create Chroma vector database
    vector_store = Chroma.from_documents(

        # Documents to store
        documents=docs,

        # Embedding model
        embedding=embeddings,

        # Collection name
        collection_name=COLLECTION_NAME,

        # Directory where DB is stored
        persist_directory=CHROMA_DIR
    )

    return vector_store


# ---------------------------------------------------
# Load existing vector database
# ---------------------------------------------------
def load_vector_store() -> Chroma:

    print("Loading vector store")

    # Load embedding model
    embeddings = get_embedding()

    # Connect to existing Chroma DB
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,

        # Embedding function used for queries
        embedding_function=embeddings,

        # Database directory
        persist_directory=CHROMA_DIR
    )

    return vector_store


# ---------------------------------------------------
# Create retriever
# ---------------------------------------------------
def get_retriever(vector_store: Chroma, k: int = 4):

    """
    Converts vector store into retriever.

    k = number of similar chunks to retrieve
    """

    return vector_store.as_retriever(

        # Similarity search
        search_type="similarity",

        # Number of chunks to return
        search_kwargs={"k": k}
    )