#!/usr/bin/env python3
"""
Improved LangChain-based RAG System with Cross-Encoder Re-ranking
Uses bi-encoder for initial retrieval, cross-encoder for re-ranking, and LLM for generation.
Two-stage retrieval pipeline:
1. Bi-encoder (sentence-transformers) retrieves top-K candidates (e.g., K=20)
2. Cross-encoder (ms-marco-MiniLM) re-ranks to top-N documents (e.g., N=5)
3. LLM generates answer from re-ranked documents
"""

import json
import os
from typing import List, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
from sentence_transformers import CrossEncoder


class GeoRAG:
    """LangChain RAG with cross-encoder re-ranking for improved retrieval accuracy."""
    
    def __init__(self,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 llm_model: str = "google/flan-t5-base",
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize with Qwen2.5-1.5B-Instruct for better instruction following."""
        print(f"="*70)
        print("🧬 IMPROVED LANGCHAIN RAG SYSTEM WITH CROSS-ENCODER")
        print(f"="*70)
        print(f"\nEmbedding model: {embedding_model}")
        print(f"Cross-encoder model: {cross_encoder_model}")
        print(f"Language model: {llm_model} (~3GB)")

        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.cross_encoder_model_name = cross_encoder_model
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.cross_encoder = None
        self.qa_chain = None
        self.prompt_template = None
        
    def load_corpus(self, corpus_path: str) -> List[Document]:
        """Load corpus and convert to LangChain Documents."""
        print(f"\n📚 Loading corpus from {corpus_path}")
        
        documents = []
        with open(corpus_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                doc = Document(
                    page_content=data["text"],
                    metadata={
                        "id": data["id"],
                        "source": data.get("source", "Unknown")
                    }
                )
                documents.append(doc)
        
        print(f"✅ Loaded {len(documents)} documents")
        return documents
    
    def build_vectorstore(self, documents: List[Document], persist_dir: str = None):
        """Build FAISS vectorstore from documents."""
        print("\n🔍 Building embeddings and vectorstore...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        if persist_dir:
            self.vectorstore.save_local(persist_dir)
            print(f"💾 Vectorstore saved to {persist_dir}")
        
        print(f"✅ Vectorstore built with {len(documents)} documents")
        
    def initialize_llm(self):
        """Initialize language model (supports both Qwen and FLAN-T5)."""
        print(f"\n🤖 Initializing language model: {self.llm_model_name}")

        # Determine model type
        is_qwen = "qwen" in self.llm_model_name.lower()
        is_flan = "flan" in self.llm_model_name.lower()

        if is_qwen:
            print("⏳ Downloading Qwen model on first run (~3GB)...")
            tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                device_map="cpu",
                trust_remote_code=True
            )

            # Create pipeline for causal LM
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1
            )

        elif is_flan:
            print("⏳ Downloading FLAN-T5 model on first run (~900MB)...")
            tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.llm_model_name)

            # Create pipeline for seq2seq
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2
            )

        else:
            raise ValueError(f"Unsupported model: {self.llm_model_name}. Use Qwen or FLAN-T5 models.")

        self.llm = HuggingFacePipeline(pipeline=pipe)
        print("✅ Language model initialized successfully")

    def initialize_cross_encoder(self):
        """Initialize cross-encoder for re-ranking."""
        print(f"\n🔄 Initializing cross-encoder: {self.cross_encoder_model_name}")
        print("⏳ Downloading model on first run (~80MB)...")

        self.cross_encoder = CrossEncoder(self.cross_encoder_model_name)
        print("✅ Cross-encoder initialized successfully")

    def rerank_documents(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Re-rank documents using cross-encoder."""
        if not documents:
            return []

        # Prepare query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]

        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)

        # Create list of (document, score) tuples
        doc_score_pairs = list(zip(documents, scores))

        # Sort by score (descending) and take top-k
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, score in doc_score_pairs[:top_k]]

        print(f"  🔄 Re-ranked {len(documents)} → {len(reranked_docs)} documents")

        return reranked_docs

    def create_qa_chain(self):
        """Create prompt template for answer generation."""
        print(f"\n⚙️  Creating prompt template...")

        # Improved prompt template with example
        prompt_template = """You are a scientific assistant specializing in genomics and Alzheimer's disease research.

Based on the context provided below, answer the question in a comprehensive yet concise manner (maximum 8 sentences).

IMPORTANT INSTRUCTIONS:Cite sources using [Source_ID] format after each key statement. Ensure citations correspond to the provided context.
EXAMPLE:
Question: What role does APOE play in Alzheimer's disease?
Context: 
[Source_A] APOE ε4 is the strongest genetic risk factor for late-onset Alzheimer's disease. 
[Source_B] The ε4 allele increases amyloid-beta accumulation in the brain.

Answer: 
APOE ε4 is the strongest genetic risk factor for Alzheimer's disease [Source_A]. The ε4 allele increases disease risk in the brain [Source_B].

Summarize the context and provide a well-cited answer, don't just copy-paste.
Now answer the following:

Context:
{context}

Question: {question}

Scientific Answer (with citations):"""

        self.prompt_template = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        print("✅ Prompt template created successfully")
        
    def query(self, question: str, initial_k: int = 20, rerank_k: int = 5) -> Dict:
        """Query the RAG system with cross-encoder re-ranking.

        Args:
            question: The query question
            initial_k: Number of documents to retrieve initially
            rerank_k: Number of documents to keep after re-ranking
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized")
        if self.llm is None:
            raise ValueError("LLM not initialized")
        if self.prompt_template is None:
            raise ValueError("Prompt template not initialized")

        print(f"\n🔍 Processing: {question}")

        # Step 1: Initial retrieval using bi-encoder
        print(f"  📥 Retrieving top-{initial_k} documents...")
        initial_docs = self.vectorstore.similarity_search(question, k=initial_k)

        # Step 2: Re-rank using cross-encoder
        if self.cross_encoder is not None:
            source_docs = self.rerank_documents(question, initial_docs, top_k=rerank_k)
        else:
            source_docs = initial_docs[:rerank_k]

        # Step 3: Build context from re-ranked documents
        context = "\n\n".join([
            f"[{doc.metadata['id']}] {doc.page_content}"
            for doc in source_docs
        ])

        # Step 4: Generate answer using LLM
        prompt = self.prompt_template.format(context=context, question=question)

        # Debug: Print the actual prompt being sent
        print(f"\n{'='*70}")
        print("🔍 DEBUG: Full Prompt Sent to LLM")
        print(f"{'='*70}")
        print(prompt)
        print(f"{'='*70}\n")

        answer = self.llm.invoke(prompt)

        # Extract citations
        citations = {}
        for doc in source_docs:
            doc_id = doc.metadata["id"]
            citations[doc_id] = {
                "text": doc.page_content,
                "source": doc.metadata["source"]
            }

        return {
            "query": question,
            "answer": answer.strip(),
            "citations": citations,
            "num_sources": len(source_docs)
        }
    
    def format_output(self, result: Dict) -> str:
        """Format result with nice formatting."""
        output = []
        output.append("\n" + "="*70)
        output.append(f"❓ QUERY: {result['query']}")
        output.append("="*70)
        output.append(f"\n💡 ANSWER:\n{result['answer']}")
        output.append(f"\n📊 Sources Used: {result['num_sources']}")
        output.append("\n📚 CITATIONS:")
        for doc_id, citation in result['citations'].items():
            output.append(f"\n  [{doc_id}] {citation['source']}")
            output.append(f"  \"{citation['text'][:120]}...\"")
        output.append("\n" + "="*70)
        return "\n".join(output)


def main():
    """Main function."""
    # Initialize system - using Qwen for better instruction following
    llm_model = "google/flan-t5-base"  # Better at following citation instructions

    rag = GeoRAG(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model=llm_model,
        cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # Load and index corpus
    documents = rag.load_corpus("corpus.jsonl")
    rag.build_vectorstore(documents, persist_dir="vectorstore_faiss_improved")

    # Initialize LLM
    rag.initialize_llm()

    # Initialize cross-encoder for re-ranking
    rag.initialize_cross_encoder()

    # Create prompt template
    rag.create_qa_chain()
    
    # Process queries
    print("\n" + "="*70)
    print("📝 PROCESSING QUERIES")
    print("="*70)

    with open("queries.jsonl", 'r') as f:
        queries = [json.loads(line) for line in f]

    results = []
    for query_obj in queries:
        query = query_obj["query"]
        # Retrieve 20 docs initially, re-rank to top 5
        result = rag.query(query, initial_k=8, rerank_k=3)
        results.append(result)
        print(rag.format_output(result))
    
    # Save results
    output_path = "outputs/langchain_improved_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_path}")
    
    formatted_path = "outputs/langchain_improved_results.txt"
    with open(formatted_path, 'w') as f:
        for result in results:
            f.write(rag.format_output(result) + "\n\n")
    print(f"✅ Formatted results saved to {formatted_path}")
    
    print("\n" + "="*70)
    print("🎉 IMPROVED LANGCHAIN RAG WITH CROSS-ENCODER COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
