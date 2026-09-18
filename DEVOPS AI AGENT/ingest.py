import os
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

from pathlib import Path

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set in your .env file.")


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "research_docs"


# ---------------------------------------------------------
# Load text files
# ---------------------------------------------------------


def load_documents():
    documents = []

    if not DATA_DIR.exists():
        raise RuntimeError(f"Data directory does not exist: {DATA_DIR.resolve()}")

    files = list(DATA_DIR.glob("*.txt"))

    print(f"Found {len(files)} text file(s).")

    for path in files:
        print(f"Loading: {path}")

        text = path.read_text(encoding="utf-8")

        if not text.strip():
            print(f"Skipping empty file: {path}")
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(path),
                    "filename": path.name,
                },
            )
        )

    return documents


# ---------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------


def main():

    print("=" * 70)
    print("RESEARCH AGENT - DOCUMENT INGESTION")
    print("=" * 70)

    # -----------------------------------------------------
    # 1. Load documents
    # -----------------------------------------------------

    documents = load_documents()

    print(f"\nLoaded documents: {len(documents)}")

    if not documents:
        raise RuntimeError(
            "No documents found.\nPut .txt files inside the data/ directory."
        )

    # -----------------------------------------------------
    # 2. Split documents
    # -----------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(documents)

    print(f"Created chunks: {len(chunks)}")

    if not chunks:
        raise RuntimeError("Document splitting produced zero chunks.")

    # Print a few chunks for debugging
    print("\nSample chunks:")

    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i + 1} ---")

        print(chunk.page_content[:300])

        print(f"Source: {chunk.metadata.get('source')}")

    # -----------------------------------------------------
    # 3. BGE-M3 embeddings through OpenRouter
    # -----------------------------------------------------

    print("\nInitializing BGE-M3...")

    class OpenRouterEmbeddings(OpenAIEmbeddings):
        def embed_documents(self, texts, chunk_size=None, **kwargs):
            embeddings = []
            for text in texts:
                response = self.client.create(model=self.model, input=text)
                embeddings.append(response.data[0].embedding)
            return embeddings

        def embed_query(self, text, **kwargs):
            response = self.client.create(model=self.model, input=text)
            return response.data[0].embedding

    embeddings = OpenRouterEmbeddings(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,  # type: ignore
        model=config.embedding_model,
    )

    # -----------------------------------------------------
    # 4. Test embedding API
    # -----------------------------------------------------

    print("\nTesting embedding API...")

    test_embedding = embeddings.embed_query("What is retrieval augmented generation?")

    print(f"Embedding dimensions: {len(test_embedding)}")

    if len(test_embedding) != 1024:
        raise RuntimeError(
            "Unexpected BGE-M3 embedding dimension. "
            f"Expected 1024, got {len(test_embedding)}."
        )

    print("BGE-M3 embedding test passed.")

    # -----------------------------------------------------
    # 5. Create Chroma vector store
    # -----------------------------------------------------

    print("\nCreating Chroma vector store...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
        collection_name=COLLECTION_NAME,
    )

    # -----------------------------------------------------
    # 6. Verify retrieval
    # -----------------------------------------------------

    print("\nTesting retrieval...")

    results = vectorstore.similarity_search(
        "What is retrieval augmented generation?",
        k=2,
    )

    print(f"Retrieved documents: {len(results)}")

    for i, document in enumerate(
        results,
        start=1,
    ):
        print(f"\n--- Retrieved document {i} ---")

        print(document.page_content[:500])

        print(f"Source: {document.metadata.get('source')}")

    print("\n" + "=" * 70)
    print("INGESTION SUCCESSFUL")
    print("=" * 70)

    print(f"Documents : {len(documents)}")

    print(f"Chunks    : {len(chunks)}")

    print(f"Vector DB : {DB_DIR}")

    print(f"Collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
