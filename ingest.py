"""
ingest.py — Run this ONCE to download the EU AI Act PDF and build the
ChromaDB vector store.

Usage:
    python ingest.py

What it does:
    1. Downloads the official EU AI Act PDF from EUR-Lex
    2. Splits it into overlapping chunks
    3. Embeds each chunk with OpenAI text-embedding-3-small
    4. Persists everything to ./chroma_db
"""

import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EU_AI_ACT_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/" "?uri=OJ:L_202401689"
PDF_PATH = Path("./data/eu_ai_act.pdf")
CHROMA_DIR = "./chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def download_pdf(url: str, dest: Path) -> None:
    """Download the EU AI Act PDF if not already present."""
    if dest.exists():
        print(f"✅  PDF already exists at {dest} — skipping download.")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print("⬇️   Downloading EU AI Act from EUR-Lex...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    size_mb = dest.stat().st_size / 1_000_000
    print(f"✅  Saved {size_mb:.1f} MB to {dest}")


def build_vectorstore(pdf_path: Path, chroma_dir: str) -> None:
    """Load, split, embed, and persist documents to ChromaDB."""

    print("📄  Loading and splitting PDF...")
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(pages)
    print(f"✅  Created {len(chunks)} chunks from {len(pages)} pages.")

    print("🔢  Embedding chunks and building ChromaDB...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
        collection_name="eu_ai_act",
    )
    vectorstore.persist()
    print(f"✅  Vector store saved to {chroma_dir}  ({len(chunks)} vectors)")


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("❌  OPENAI_API_KEY not set. Copy .env.example → .env and add your key.")
        sys.exit(1)

    download_pdf(EU_AI_ACT_URL, PDF_PATH)
    build_vectorstore(PDF_PATH, CHROMA_DIR)
    print("\n🎉  Ingestion complete! Run the app with:  streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()
