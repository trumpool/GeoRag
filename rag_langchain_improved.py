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
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
from sentence_transformers import CrossEncoder
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


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

    def rerank_documents(self, query: str, documents: List[Document], top_k: int = 5,
                        score_threshold: float = None) -> List[Document]:
        """Re-rank documents using cross-encoder with optional score threshold.

        Args:
            query: Query text
            documents: Documents to re-rank
            top_k: Number of top documents to return
            score_threshold: Minimum cross-encoder score (typically -5 to 5, higher is better)

        Returns:
            Re-ranked and filtered documents
        """
        if not documents:
            return []

        # Prepare query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]

        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)

        # Create list of (document, score) tuples
        doc_score_pairs = list(zip(documents, scores))

        # Sort by score (descending)
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Apply score threshold if specified
        if score_threshold is not None:
            filtered_pairs = [(doc, score) for doc, score in doc_score_pairs if score >= score_threshold]
            print(f"  📊 Score threshold {score_threshold}: {len(filtered_pairs)}/{len(doc_score_pairs)} docs passed")
            doc_score_pairs = filtered_pairs

        # Take top-k
        reranked_docs = [doc for doc, score in doc_score_pairs[:top_k]]

        print(f"  🔄 Re-ranked {len(documents)} → {len(reranked_docs)} documents")

        return reranked_docs

    def create_qa_chain(self):
        """Create prompt template for answer generation."""
        print(f"\n⚙️  Creating prompt template...")

        # Improved prompt template with stronger instructions and multiple examples
        prompt_template = """You are a scientific assistant specializing in genomics and Alzheimer's disease research.

CRITICAL RULES - YOU MUST FOLLOW THESE:
1. ALWAYS cite sources using [Source_ID] format after EVERY statement
2. Use the EXACT source IDs from the context (e.g., [30617256_sent_1])
3. Every sentence MUST end with at least one citation like [Source_ID]
4. If you mention information, cite it immediately

EXAMPLES OF CORRECT CITATION FORMAT:

Question: How does tau protein affect neurons?
Context:
[Tau_1] Hyperphosphorylated tau forms neurofibrillary tangles in neurons.
[Tau_2] Tau tangles disrupt microtubule stability and axonal transport.

Correct Answer:
Hyperphosphorylated tau forms neurofibrillary tangles in neurons [Tau_1]. These tangles disrupt microtubule stability and axonal transport [Tau_2].

NOW ANSWER THE FOLLOWING QUESTION:

Context:
{context}

Question: {question}

Answer with citations (use exact Source_IDs from context above):"""

        self.prompt_template = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        print("✅ Prompt template created successfully")

    def has_valid_citations(self, answer: str, source_doc_ids: List[str]) -> bool:
        """Check if answer contains valid citations in [doc_id] format.

        Args:
            answer: The generated answer text
            source_doc_ids: List of valid document IDs from retrieved sources

        Returns:
            True if answer contains at least one valid citation, False otherwise
        """
        # Pattern to find citations like [30617256_sent_1]
        citation_pattern = r'\[([^\]]+)\]'
        found_citations = re.findall(citation_pattern, answer)

        # Check if we found any citations that match our source document IDs
        valid_citations = [cite for cite in found_citations if cite in source_doc_ids]

        return len(valid_citations) > 0

    def query(self, question: str, initial_k: int = 20, rerank_k: int = 5,
             similarity_threshold: float = None, rerank_threshold: float = None,
             max_retries: int = 3) -> Dict:
        """Query the RAG system with cross-encoder re-ranking and citation validation.

        Args:
            question: The query question
            initial_k: Number of documents to retrieve initially
            rerank_k: Number of documents to keep after re-ranking
            similarity_threshold: Minimum cosine similarity for initial retrieval (0-1, typically 0.3-0.7)
            rerank_threshold: Minimum cross-encoder score for re-ranking (typically -5 to 5)
            max_retries: Maximum number of regeneration attempts if citations are missing
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized")
        if self.llm is None:
            raise ValueError("LLM not initialized")
        if self.prompt_template is None:
            raise ValueError("Prompt template not initialized")

        print(f"\n🔍 Processing: {question}")

        # Step 1: Initial retrieval using bi-encoder with similarity scores
        print(f"  📥 Retrieving top-{initial_k} documents...")
        docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=initial_k)

        # Apply similarity threshold if specified
        if similarity_threshold is not None:
            initial_docs = [doc for doc, score in docs_with_scores if score <= similarity_threshold]
            filtered_count = len(initial_docs)
            print(f"  📊 Similarity threshold {similarity_threshold}: {filtered_count}/{initial_k} docs passed")

            # If too few docs pass threshold, warn user
            if filtered_count == 0:
                print(f"  ⚠️  No documents passed threshold! Using top document anyway.")
                initial_docs = [docs_with_scores[0][0]]
            elif filtered_count < rerank_k:
                print(f"  ⚠️  Only {filtered_count} docs passed threshold (< rerank_k={rerank_k})")
        else:
            initial_docs = [doc for doc, score in docs_with_scores]

        # Step 2: Re-rank using cross-encoder with optional threshold
        if self.cross_encoder is not None:
            source_docs = self.rerank_documents(
                question, initial_docs,
                top_k=rerank_k,
                score_threshold=rerank_threshold
            )

            # Safety check: if threshold is too strict and no docs pass, use top doc
            if len(source_docs) == 0:
                print(f"  ⚠️  Rerank threshold too strict! Using top document anyway.")
                source_docs = initial_docs[:1]
        else:
            source_docs = initial_docs[:rerank_k]

        # Step 3: Build context from re-ranked documents
        context = "\n\n".join([
            f"[{doc.metadata['id']}] {doc.page_content}"
            for doc in source_docs
        ])

        # Get list of valid document IDs for citation checking
        valid_doc_ids = [doc.metadata['id'] for doc in source_docs]

        # Step 4: Generate answer with retry logic for missing citations
        answer = None
        for attempt in range(1, max_retries + 1):
            print(f"  🤖 Generating answer (attempt {attempt}/{max_retries})...")

            # Generate answer using LLM
            prompt = self.prompt_template.format(context=context, question=question)
            answer = self.llm.invoke(prompt).strip()

            # Check if answer contains valid citations
            if self.has_valid_citations(answer, valid_doc_ids):
                print(f"  ✅ Valid citations found in answer")
                break
            else:
                print(f"  ⚠️  No valid citations found in answer")
                if attempt < max_retries:
                    print(f"  🔄 Retrying generation...")
                else:
                    print(f"  ❌ Max retries reached, keeping last answer")

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
            "answer": answer,
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

    def visualize_embeddings(self,
                            query: Optional[str] = None,
                            retrieved_doc_ids: Optional[List[str]] = None,
                            method: str = "umap",
                            n_samples: int = 500,
                            save_path: str = "outputs/embedding_visualization.png"):
        """Visualize document embeddings in 2D space using UMAP or t-SNE.

        Args:
            query: Optional query text to highlight on the plot
            retrieved_doc_ids: Optional list of retrieved document IDs to highlight
            method: Dimensionality reduction method - "umap" or "tsne"
            n_samples: Number of random documents to visualize (for performance)
            save_path: Path to save the visualization
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized")

        print(f"\n🎨 Generating {method.upper()} embedding visualization...")

        # Get all document vectors from FAISS
        index = self.vectorstore.index
        total_docs = index.ntotal
        print(f"  📊 Total documents in vectorstore: {total_docs}")

        # Sample documents if total is too large
        if total_docs > n_samples:
            print(f"  🎲 Sampling {n_samples} documents for visualization...")
            sample_indices = np.random.choice(total_docs, n_samples, replace=False)
        else:
            sample_indices = np.arange(total_docs)
            n_samples = total_docs

        # Extract embeddings
        embeddings = np.array([index.reconstruct(int(i)) for i in sample_indices])
        print(f"  📐 Embedding shape: {embeddings.shape}")

        # Get document IDs for sampled documents
        doc_ids = []
        docstore = self.vectorstore.docstore
        index_to_id = self.vectorstore.index_to_docstore_id
        for idx in sample_indices:
            doc_id_key = index_to_id.get(int(idx))
            if doc_id_key:
                doc = docstore.search(doc_id_key)
                if doc:
                    doc_ids.append(doc.metadata.get('id', 'unknown'))
                else:
                    doc_ids.append('unknown')
            else:
                doc_ids.append('unknown')

        # Add query embedding if provided
        query_embedding = None
        if query:
            query_embedding = np.array(self.embeddings.embed_query(query)).reshape(1, -1)
            embeddings = np.vstack([embeddings, query_embedding])
            doc_ids.append("QUERY")
            print(f"  ➕ Added query to visualization")

        # Dimensionality reduction
        print(f"  🔄 Reducing dimensions with {method.upper()}...")
        if method.lower() == "umap" and UMAP_AVAILABLE:
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        elif method.lower() == "tsne" or not UMAP_AVAILABLE:
            if method.lower() == "umap" and not UMAP_AVAILABLE:
                print("  ⚠️  UMAP not available, falling back to t-SNE")
                print("  💡 Install UMAP with: pip install umap-learn")
            reducer = TSNE(n_components=2, perplexity=30, random_state=42, metric='cosine')
            embeddings_2d = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Unsupported method: {method}. Use 'umap' or 'tsne'")

        # Create visualization
        print(f"  🎨 Creating plot...")
        plt.figure(figsize=(12, 10))

        # Prepare colors
        colors = []
        sizes = []
        labels_to_show = []

        for i, doc_id in enumerate(doc_ids):
            if doc_id == "QUERY":
                colors.append('red')
                sizes.append(300)
                labels_to_show.append((embeddings_2d[i], "Query"))
            elif retrieved_doc_ids and doc_id in retrieved_doc_ids:
                colors.append('orange')
                sizes.append(150)
                labels_to_show.append((embeddings_2d[i], doc_id))
            else:
                colors.append('lightblue')
                sizes.append(50)

        # Plot all points
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                   c=colors, s=sizes, alpha=0.6, edgecolors='black', linewidth=0.5)

        # Add labels for query and retrieved documents
        for pos, label in labels_to_show:
            plt.annotate(label, xy=pos, xytext=(5, 5),
                        textcoords='offset points', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        # Formatting
        method_name = method.upper() if method.lower() != "umap" or UMAP_AVAILABLE else "t-SNE"
        plt.title(f'Document Embeddings Visualization ({method_name})\n' +
                 f'Total: {n_samples} documents', fontsize=14, fontweight='bold')
        plt.xlabel(f'{method_name} Dimension 1', fontsize=12)
        plt.ylabel(f'{method_name} Dimension 2', fontsize=12)

        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                      markersize=8, label='Corpus Documents'),
        ]
        if query:
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                          markersize=12, label='Query')
            )
        if retrieved_doc_ids:
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                          markersize=10, label='Retrieved Documents')
            )

        plt.legend(handles=legend_elements, loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save figure
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Visualization saved to {save_path}")

        plt.close()

        return save_path


def main():
    """Main function."""
    # Initialize system - using FLAN-T5
    llm_model = "google/flan-t5-base"

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

    # ============================================================================
    # RETRIEVAL SETTINGS - Adjust thresholds to filter irrelevant documents
    # ============================================================================
    INITIAL_K = 10              # Number of docs to retrieve initially
    RERANK_K = 4                # Number of docs to keep after re-ranking

    # Similarity threshold for initial retrieval (cosine similarity)
    # - Range: 0-1 (lower is more similar, FAISS uses L2 distance)
    # - None = no filtering
    # - Typical values: 0.3-0.7 (lower is more strict)
    SIMILARITY_THRESHOLD = 0.75  # No filtering - keep all retrieved docs

    # Re-rank threshold for cross-encoder score
    # - Range: -5 to 5 (higher is better)
    # - None = no filtering
    # - Typical values: -1.0 to 2.0 (higher is more strict)
    RERANK_THRESHOLD = None      # Set to 0.0 to filter low-scored docs

    print(f"\n⚙️  Retrieval Settings:")
    print(f"    Initial K: {INITIAL_K}")
    print(f"    Rerank K: {RERANK_K}")
    print(f"    Similarity Threshold: {SIMILARITY_THRESHOLD}")
    print(f"    Rerank Threshold: {RERANK_THRESHOLD}")
    # ============================================================================

    # Process queries
    print("\n" + "="*70)
    print("📝 PROCESSING QUERIES")
    print("="*70)

    with open("queries.jsonl", 'r') as f:
        queries = [json.loads(line) for line in f]

    results = []
    for query_obj in queries:
        query = query_obj["query"]
        # Query with optional thresholds
        result = rag.query(
            query,
            initial_k=INITIAL_K,
            rerank_k=RERANK_K,
            similarity_threshold=SIMILARITY_THRESHOLD,
            rerank_threshold=RERANK_THRESHOLD
        )
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

    # Generate visualizations
    print("\n" + "="*70)
    print("📊 GENERATING EMBEDDING VISUALIZATIONS")
    print("="*70)

    # 1. Overall corpus visualization (using UMAP if available, otherwise t-SNE)
    try:
        rag.visualize_embeddings(
            method="umap",
            n_samples=500,
            save_path="outputs/embedding_viz_overall.png"
        )
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")

    # 2. Visualization for each query and its retrieved documents
    if results:
        for idx, result in enumerate(results, start=1):
            try:
                rag.visualize_embeddings(
                    query=result["query"],
                    retrieved_doc_ids=list(result["citations"].keys()),
                    method="umap",
                    n_samples=500,
                    save_path=f"outputs/embedding_viz_query{idx}.png"
                )
            except Exception as e:
                print(f"⚠️  Query {idx} visualization failed: {e}")

    print("\n" + "="*70)
    print("🎉 IMPROVED LANGCHAIN RAG WITH CROSS-ENCODER COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
