import os
from dotenv import load_dotenv

load_dotenv()

embedding_model = "baai/bge-m3" 

# Gemini Flash-Lite is an ultra-fast, highly-compressed ~8-Billion parameter multimodal model engineered by Google for extreme cost-efficiency and sub-second latency in high-volume, repetitive AI tasks.

# Dots3-Note Preview is an open-weight mixture-of-experts model from Dots Studio, with 16B active parameters out of 280B total. It is the lightest model in the Dots 3 family and is suited for reasoning, coding, multimodal understanding, long-context processing, and multi-step agent workflows

# The bge-m3 embedding model encodes sentences, paragraphs, and long documents into a 1024-dimensional dense vector space, delivering high-quality semantic embeddings optimized for multilingual retrieval, semantic search, and large-context applications.

def get_agent_model():

    from langchain_openrouter import ChatOpenRouter
    return ChatOpenRouter(
        model="dots-studio/dots-3-note-preview:free",
        temperature=0.3,
    )

    # from langchain_google_genai import ChatGoogleGenerativeAI
    # return ChatGoogleGenerativeAI(
    #     model="gemini-flash-lite-latest",
    #     temperature=0.3,
    # )


def get_judge_model(): 

    from deepeval.models import OpenRouterModel
    return OpenRouterModel(
        model="dots-studio/dots-3-note-preview:free",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        temperature=0    
    )

    # from deepeval.models import GeminiModel
    # return GeminiModel(
    #     model="gemini-flash-lite-latest",
    #     api_key=os.environ.get("GOOGLE_API_KEY"),
    #     temperature=0,
    # )
