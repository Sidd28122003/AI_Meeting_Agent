# Import ChatMistralAI model from LangChain integration
from langchain_mistralai import ChatMistralAI

# Used to create structured prompts for the LLM
from langchain_core.prompts import ChatPromptTemplate

# Converts model output into plain string text
from langchain_core.output_parsers import StrOutputParser

# Splits large text into smaller chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Utilities for chaining data flow in LangChain
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import os


# ---------------------------------------------------
# Function: get_llm()
# Purpose:
# Creates and returns the Mistral language model
# ---------------------------------------------------
def get_llm():

    return ChatMistralAI(

        # Mistral model name
        model_name="mistral-small-latest",

        # API key stored in environment variables
        api_key=os.getenv("MISTRAL_API_KEY"),

        # Controls randomness
        # Lower = more focused responses
        temperature=0.3
    )


# ---------------------------------------------------
# Function: split_transcript()
# Purpose:
# Splits long transcript into smaller chunks
# because LLMs have token limits
# ---------------------------------------------------
def split_transcript(transcript: str) -> list:

    # Create text splitter
    splitter = RecursiveCharacterTextSplitter(

        # Maximum characters in one chunk
        chunk_size=3000,

        # Overlap keeps context continuity
        chunk_overlap=200
    )

    # Returns list of text chunks
    return splitter.split_text(transcript)


# ---------------------------------------------------
# Function: summarize()
# Purpose:
# Generates final meeting summary
# Workflow:
# 1. Split transcript
# 2. Summarize each chunk
# 3. Combine summaries
# 4. Generate final summary
# ---------------------------------------------------
def summarize(transcript: str) -> str:

    # Load LLM
    llm = get_llm()

    # ---------------- MAP STEP ----------------
    # Prompt for summarizing EACH chunk
    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Summarize this portion of a meeting transcript concisely."
            ),

            ("human", "{text}"),
        ]
    )

    # Create map chain
    # Prompt -> LLM -> String Output
    map_chain = map_prompt | llm | StrOutputParser()

    # Split transcript into chunks
    chunks = split_transcript(transcript)

    # Generate summary for every chunk
    chunk_summaries = [
        map_chain.invoke({"text": chunk})
        for chunk in chunks
    ]

    # Combine all chunk summaries into one text
    combined = "\n\n".join(chunk_summaries)

    # ---------------- REDUCE STEP ----------------
    # Final prompt to combine summaries
    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert meeting summarizer. "
                "Combine these partial summaries into one "
                "final professional meeting summary in bullet points.",
            ),

            ("human", "{text}"),
        ]
    )

    # Create final reduce chain
    combined_chain = (

        # Pass input forward unchanged
        RunnablePassthrough()

        # Convert raw string into dictionary
        | RunnableLambda(lambda x: {"text": x})

        # Apply prompt template
        | combined_prompt

        # Send to LLM
        | llm

        # Convert response to string
        | StrOutputParser()
    )

    # Return final combined summary
    return combined_chain.invoke(combined)


# ---------------------------------------------------
# Function: generate_title()
# Purpose:
# Generates short professional meeting title
# ---------------------------------------------------
def generate_title(transcript: str) -> str:

    # Load LLM
    llm = get_llm()

    # Build title generation chain
    title_chain = (

        # Pass transcript as-is
        RunnablePassthrough()

        # Convert string into dictionary
        | RunnableLambda(lambda x: {"text": x})

        # Prompt template
        | ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Based on the meeting transcript, generate "
                    "a short professional meeting title "
                    "(max 8 words). "
                    "Only return the title, nothing else.",
                ),

                ("human", "{text}"),
            ]
        )

        # Send prompt to model
        | llm

        # Convert output into plain text
        | StrOutputParser()
    )

    # Use only first 2000 characters for title generation
    return title_chain.invoke(transcript[:2000])


    