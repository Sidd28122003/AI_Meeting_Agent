# ---------------------------------------------------
# Import Required Libraries
# ---------------------------------------------------

# Used to access environment variables
import os

# Mistral AI LLM integration with LangChain
from langchain_mistralai import ChatMistralAI

# Used for creating structured prompts
from langchain_core.prompts import ChatPromptTemplate

# Converts model response into plain string
from langchain_core.output_parsers import StrOutputParser

# LCEL utilities
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda
)

# Import custom vector store functions
from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever
)


# ---------------------------------------------------
# Load Mistral LLM
# ---------------------------------------------------

def get_llm():
    """
    Creates and returns Mistral AI language model.

    temperature = 0.3
    Lower temperature gives more focused and stable answers.
    """

    return ChatMistralAI(

        # Mistral model name
        model="mistral-small-latest",

        # API key stored in .env file
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),

        # Controls creativity
        temperature=0.3,
    )


# ---------------------------------------------------
# Format Retrieved Documents
# ---------------------------------------------------

def format_docs(docs):
    """
    Combines retrieved document chunks
    into one single string.

    Input:
        List[Document]

    Output:
        Single formatted string
    """

    return "\n\n".join(
        doc.page_content for doc in docs
    )


# ---------------------------------------------------
# Build New RAG Chain
# ---------------------------------------------------

def build_rag_chain(transcript: str):

    """
    Creates a complete RAG pipeline.

    Flow:
    Transcript
        ↓
    Chunking
        ↓
    Embeddings
        ↓
    Vector Store
        ↓
    Retriever
        ↓
    Prompt
        ↓
    LLM
        ↓
    Final Answer
    """

    # ---------------------------------------------------
    # Create vector database from transcript
    # ---------------------------------------------------

    vector_store = build_vector_store(transcript)

    # ---------------------------------------------------
    # Create retriever
    # k=4 means retrieve top 4 relevant chunks
    # ---------------------------------------------------

    retriever = get_retriever(
        vector_store,
        k=4
    )

    # ---------------------------------------------------
    # Load Mistral model
    # ---------------------------------------------------

    llm = get_llm()

    # ---------------------------------------------------
    # Create Prompt Template
    # ---------------------------------------------------

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",

                """
                You are an expert meeting assistant.

                Answer the user's question
                based ONLY on the meeting transcript
                context provided below.

                If the answer is not found
                in the context, say:

                "I could not find this information
                in the meeting transcript."

                Always be concise and precise.

                If quoting someone,
                mention it clearly.

                Context from meeting transcript:
                {context}
                """,
            ),

            # User question placeholder
            ("human", "{question}"),
        ]
    )

    # ---------------------------------------------------
    # Full LCEL RAG Chain
    # ---------------------------------------------------

    rag_chain = (

        # ---------------------------------------------------
        # Input Mapping
        # ---------------------------------------------------
        {
            # Retrieve relevant chunks
            # then format them into string
            "context": (
                retriever
                | RunnableLambda(format_docs)
            ),

            # Pass original user question
            "question": RunnablePassthrough()
        }

        # Insert values into prompt
        | prompt

        # Send prompt to LLM
        | llm

        # Convert AI response into string
        | StrOutputParser()
    )

    return rag_chain


# ---------------------------------------------------
# Load Existing RAG Chain
# ---------------------------------------------------

def load_rag_chain():
    """
    Loads existing vector database
    instead of rebuilding it.

    Useful for production systems.
    """

    # ---------------------------------------------------
    # Load existing vector database
    # ---------------------------------------------------

    vector_store = load_vector_store()

    # ---------------------------------------------------
    # Create retriever from loaded DB
    # ---------------------------------------------------

    # NOTE:
    # Your original code had a small mistake:
    #
    # retriver = get_retriever()
    #
    # Correct:
    # retriever = get_retriever(vector_store)
    #

    retriever = get_retriever(
        vector_store,
        k=4
    )

    # ---------------------------------------------------
    # Load Mistral model
    # ---------------------------------------------------

    llm = get_llm()

    # ---------------------------------------------------
    # Prompt Template
    # ---------------------------------------------------

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an expert meeting assistant.

            Answer the user's question
            based ONLY on the meeting transcript
            context provided below.

            If the answer is not found
            in the context, say:

            "I could not find this information
            in the meeting transcript."

            Always be concise and precise.

            If quoting someone,
            mention it clearly.

            Context from meeting transcript:
            {context}
            """,
        ),

        # User question placeholder
        ("human", "{question}"),
    ])

    # ---------------------------------------------------
    # Create LCEL RAG Pipeline
    # ---------------------------------------------------

    rag_chain = (
        {

            # Retrieve + format context
            "context": (
                retriever
                | RunnableLambda(format_docs)
            ),

            # Pass question directly
            "question": RunnablePassthrough(),
        }

        # Insert into prompt
        | prompt

        # Send to LLM
        | llm

        # Convert output into string
        | StrOutputParser()
    )

    return rag_chain


# ---------------------------------------------------
# Ask Question Function
# ---------------------------------------------------

def ask_question(rag_chain, question: str) -> str:
    """
    Sends user question to RAG pipeline
    and returns AI-generated answer.

    Parameters:
        rag_chain -> Complete RAG pipeline
        question -> User question

    Returns:
        AI-generated answer string
    """

    # Print question in terminal
    print(f"Question: {question}")

    # Run complete RAG pipeline
    answer = rag_chain.invoke(question)

    # Print answer in terminal
    print(f"Answer: {answer}")

    return answer