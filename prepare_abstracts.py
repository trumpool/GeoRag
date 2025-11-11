#!/usr/bin/env python3
"""
Prepare abstracts from the abstracts directory for RAG system.
Converts abstract files into corpus.jsonl format.
"""

import json
import os
import re
from pathlib import Path


def split_into_sentences(text: str) -> list:
    """
    Split text into sentences using regex.

    Args:
        text: Input text to split

    Returns:
        List of sentences
    """
    # Split on sentence endings (.!?) followed by whitespace or end of string
    # This regex keeps the punctuation with the sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Filter out empty sentences and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def load_abstracts_from_directory(abstracts_dir: str = "abstracts") -> list:
    """
    Load all abstract files from the abstracts directory and split into sentences.

    Args:
        abstracts_dir: Path to directory containing abstract files

    Returns:
        List of dictionaries with id, text, and source fields (one per sentence)
    """
    abstracts_path = Path(abstracts_dir)

    if not abstracts_path.exists():
        raise FileNotFoundError(f"Directory '{abstracts_dir}' not found")

    documents = []

    # Get all files in the abstracts directory
    abstract_files = sorted([f for f in abstracts_path.iterdir() if f.is_file()])

    print(f"Found {len(abstract_files)} abstract files")

    for file_path in abstract_files:
        # Read the abstract text
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        # Use filename as ID (likely PubMed ID or similar)
        doc_id = file_path.name

        # Split into sentences
        sentences = split_into_sentences(text)

        # Create a document entry for each sentence
        for i, sentence in enumerate(sentences, 1):
            doc = {
                "id": f"{doc_id}_sent_{i}",
                "text": sentence,
                "source": f"Abstract_{doc_id}"
            }
            documents.append(doc)

        print(f"  ✓ Loaded: {doc_id} ({len(sentences)} sentences, {len(text)} characters)")

    return documents


def save_corpus(documents: list, output_path: str = "corpus.jsonl"):
    """
    Save documents to JSONL format for RAG system.

    Args:
        documents: List of document dictionaries
        output_path: Output file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(json.dumps(doc) + '\n')

    print(f"\n✅ Saved {len(documents)} sentence chunks to {output_path}")


def create_sample_queries(output_path: str = "queries.jsonl"):
    """
    Create sample queries about Alzheimer's disease research.
    """
    sample_queries = [
        {
            "id": "q1",
            "query": "What genetic loci have been identified for Alzheimer's disease?"
        },
        {
            "id": "q2",
            "query": "How does hypertension affect Alzheimer's disease risk?"
        },
        {
            "id": "q3",
            "query": "What genes are associated with AD and what are their biological functions?"
        },
        {
            "id": "q4",
            "query": "What is the role of APOE and other genes in Alzheimer's pathology?"
        },
        {
            "id": "q5",
            "query": "What are the findings from genome-wide association studies of Alzheimer's?"
        }
    ]

    with open(output_path, 'w', encoding='utf-8') as f:
        for query in sample_queries:
            f.write(json.dumps(query) + '\n')

    print(f"✅ Created {len(sample_queries)} sample queries in {output_path}")


def main():
    """Main function to prepare abstracts for RAG."""
    print("="*70)
    print("📚 PREPARING ABSTRACTS FOR RAG KNOWLEDGE BASE")
    print("="*70)

    # Load abstracts
    print("\n1️⃣  Loading abstracts from directory...")
    documents = load_abstracts_from_directory("abstracts")

    # Save as corpus
    print("\n2️⃣  Saving corpus...")
    save_corpus(documents, "corpus.jsonl")

    # Create sample queries
    print("\n3️⃣  Creating sample queries...")
    create_sample_queries("queries.jsonl")

    # Create outputs directory if needed
    os.makedirs("outputs", exist_ok=True)
    print("✅ Created outputs directory")

    print("\n" + "="*70)
    print("🎉 PREPARATION COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run: python rag_langchain_improved.py")
    print("  2. The system will build the knowledge base and answer queries")
    print("="*70)


if __name__ == "__main__":
    main()
