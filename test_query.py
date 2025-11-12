#!/usr/bin/env python3
"""Test single query to debug prompt and context."""

from rag_langchain_improved import GeoRAG

# Initialize system with Qwen for better instruction following
rag = GeoRAG(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="google/flan-t5-base",
    cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Load corpus
documents = rag.load_corpus("corpus.jsonl")
rag.build_vectorstore(documents, persist_dir="vectorstore_faiss_improved")

# Initialize models
rag.initialize_llm()
rag.initialize_cross_encoder()

# Create prompt template
rag.create_qa_chain()

# Test single query
test_query = "What genetic loci have been identified for Alzheimer's disease?"
result = rag.query(test_query, initial_k=20, rerank_k=5)

print("\n" + "="*70)
print("FINAL ANSWER:")
print("="*70)
print(result['answer'])
print("="*70)
