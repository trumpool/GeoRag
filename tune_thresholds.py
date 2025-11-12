#!/usr/bin/env python3
"""
Threshold Tuning Helper
Helps you find the optimal similarity and rerank thresholds for your RAG system.
"""

import json
import numpy as np
from rag_langchain_improved import GeoRAG


def analyze_similarity_scores(rag: GeoRAG, queries: list, k: int = 20):
    """
    Analyze similarity score distribution to help choose threshold.

    Args:
        rag: Initialized RAG system
        queries: List of query dictionaries
        k: Number of documents to retrieve
    """
    print("\n" + "="*70)
    print("📊 ANALYZING SIMILARITY SCORES")
    print("="*70)

    all_scores = []

    for query_obj in queries:
        query = query_obj["query"]
        # Get documents with similarity scores
        docs_with_scores = rag.vectorstore.similarity_search_with_score(query, k=k)

        scores = [score for doc, score in docs_with_scores]
        all_scores.extend(scores)

        print(f"\nQuery: {query[:60]}...")
        print(f"  Score range: {min(scores):.4f} - {max(scores):.4f}")
        print(f"  Top 5 scores: {[f'{s:.4f}' for s in scores[:5]]}")

    # Overall statistics
    print("\n" + "="*70)
    print("📈 OVERALL SIMILARITY SCORE STATISTICS")
    print("="*70)
    print(f"  Mean:   {np.mean(all_scores):.4f}")
    print(f"  Median: {np.median(all_scores):.4f}")
    print(f"  Std:    {np.std(all_scores):.4f}")
    print(f"  Min:    {np.min(all_scores):.4f}")
    print(f"  Max:    {np.max(all_scores):.4f}")

    # Percentiles
    percentiles = [25, 50, 75, 90, 95]
    print(f"\n  Percentiles:")
    for p in percentiles:
        val = np.percentile(all_scores, p)
        print(f"    {p}th: {val:.4f}")

    print("\n💡 RECOMMENDATIONS:")
    print(f"  Conservative (keep most docs):  threshold < {np.percentile(all_scores, 75):.4f}")
    print(f"  Moderate (balanced):            threshold < {np.percentile(all_scores, 50):.4f}")
    print(f"  Aggressive (only best matches): threshold < {np.percentile(all_scores, 25):.4f}")

    return all_scores


def analyze_rerank_scores(rag: GeoRAG, queries: list, k: int = 20):
    """
    Analyze cross-encoder rerank score distribution.

    Args:
        rag: Initialized RAG system with cross-encoder
        queries: List of query dictionaries
        k: Number of documents to retrieve for re-ranking
    """
    print("\n" + "="*70)
    print("📊 ANALYZING CROSS-ENCODER RERANK SCORES")
    print("="*70)

    if rag.cross_encoder is None:
        print("❌ Cross-encoder not initialized!")
        return None

    all_scores = []

    for query_obj in queries:
        query = query_obj["query"]
        # Get initial documents
        initial_docs = rag.vectorstore.similarity_search(query, k=k)

        # Get cross-encoder scores
        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = rag.cross_encoder.predict(pairs)

        all_scores.extend(scores.tolist())

        print(f"\nQuery: {query[:60]}...")
        print(f"  Score range: {min(scores):.4f} - {max(scores):.4f}")
        print(f"  Top 5 scores: {[f'{s:.4f}' for s in sorted(scores, reverse=True)[:5]]}")

    # Overall statistics
    print("\n" + "="*70)
    print("📈 OVERALL RERANK SCORE STATISTICS")
    print("="*70)
    print(f"  Mean:   {np.mean(all_scores):.4f}")
    print(f"  Median: {np.median(all_scores):.4f}")
    print(f"  Std:    {np.std(all_scores):.4f}")
    print(f"  Min:    {np.min(all_scores):.4f}")
    print(f"  Max:    {np.max(all_scores):.4f}")

    # Percentiles
    percentiles = [25, 50, 75, 90, 95]
    print(f"\n  Percentiles:")
    for p in percentiles:
        val = np.percentile(all_scores, p)
        print(f"    {p}th: {val:.4f}")

    print("\n💡 RECOMMENDATIONS:")
    print(f"  Conservative (keep most docs):  threshold > {np.percentile(all_scores, 25):.4f}")
    print(f"  Moderate (balanced):            threshold > {np.percentile(all_scores, 50):.4f}")
    print(f"  Aggressive (only best matches): threshold > {np.percentile(all_scores, 75):.4f}")

    return all_scores


def test_thresholds(rag: GeoRAG, queries: list,
                   similarity_thresholds: list = None,
                   rerank_thresholds: list = None):
    """
    Test different threshold combinations and show how many docs are retrieved.

    Args:
        rag: Initialized RAG system
        queries: List of query dictionaries
        similarity_thresholds: List of similarity thresholds to test
        rerank_thresholds: List of rerank thresholds to test
    """
    if similarity_thresholds is None:
        similarity_thresholds = [None, 0.5, 0.6, 0.7]

    if rerank_thresholds is None:
        rerank_thresholds = [None, -1.0, 0.0, 1.0]

    print("\n" + "="*70)
    print("🧪 TESTING THRESHOLD COMBINATIONS")
    print("="*70)

    test_query = queries[0]["query"]
    print(f"\nTest Query: {test_query}\n")

    results = []

    for sim_thresh in similarity_thresholds:
        for rerank_thresh in rerank_thresholds:
            # Get documents with scores
            docs_with_scores = rag.vectorstore.similarity_search_with_score(test_query, k=10)

            # Apply similarity threshold
            if sim_thresh is not None:
                initial_docs = [doc for doc, score in docs_with_scores if score <= sim_thresh]
            else:
                initial_docs = [doc for doc, score in docs_with_scores]

            # Re-rank
            if rag.cross_encoder is not None and len(initial_docs) > 0:
                pairs = [[test_query, doc.page_content] for doc in initial_docs]
                scores = rag.cross_encoder.predict(pairs)
                doc_score_pairs = list(zip(initial_docs, scores))
                doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

                # Apply rerank threshold
                if rerank_thresh is not None:
                    final_docs = [doc for doc, score in doc_score_pairs if score >= rerank_thresh]
                else:
                    final_docs = [doc for doc, score in doc_score_pairs[:4]]
            else:
                final_docs = initial_docs[:4]

            result = {
                "sim_threshold": sim_thresh,
                "rerank_threshold": rerank_thresh,
                "initial_docs": len(initial_docs),
                "final_docs": len(final_docs)
            }
            results.append(result)

            sim_str = f"{sim_thresh:.2f}" if sim_thresh else "None"
            rerank_str = f"{rerank_thresh:.2f}" if rerank_thresh else "None"
            print(f"  Sim={sim_str:>6}, Rerank={rerank_str:>6} → "
                  f"{len(initial_docs):>2} initial → {len(final_docs):>2} final docs")

    print("\n💡 Choose thresholds that give you 3-5 final documents")

    return results


def main():
    """Run threshold analysis."""
    print("="*70)
    print("🔧 THRESHOLD TUNING HELPER")
    print("="*70)

    # Initialize RAG system
    print("\n1️⃣  Initializing RAG system...")
    rag = GeoRAG(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model="google/flan-t5-base",
        cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # Load corpus
    documents = rag.load_corpus("corpus.jsonl")
    rag.build_vectorstore(documents, persist_dir="vectorstore_faiss_improved")

    # Initialize cross-encoder
    rag.initialize_cross_encoder()

    # Load queries
    with open("queries.jsonl", 'r') as f:
        queries = [json.loads(line) for line in f]

    print(f"\n📊 Analyzing {len(queries)} queries...")

    # Analyze similarity scores
    similarity_scores = analyze_similarity_scores(rag, queries, k=10)

    # Analyze rerank scores
    rerank_scores = analyze_rerank_scores(rag, queries, k=10)

    # Test different thresholds
    test_results = test_thresholds(rag, queries)

    # Save analysis
    output = {
        "similarity_scores": {
            "mean": float(np.mean(similarity_scores)),
            "median": float(np.median(similarity_scores)),
            "std": float(np.std(similarity_scores)),
            "percentiles": {
                "25": float(np.percentile(similarity_scores, 25)),
                "50": float(np.percentile(similarity_scores, 50)),
                "75": float(np.percentile(similarity_scores, 75))
            }
        },
        "rerank_scores": {
            "mean": float(np.mean(rerank_scores)),
            "median": float(np.median(rerank_scores)),
            "std": float(np.std(rerank_scores)),
            "percentiles": {
                "25": float(np.percentile(rerank_scores, 25)),
                "50": float(np.percentile(rerank_scores, 50)),
                "75": float(np.percentile(rerank_scores, 75))
            }
        },
        "threshold_tests": test_results
    }

    with open("outputs/threshold_analysis.json", 'w') as f:
        json.dump(output, f, indent=2)

    print("\n✅ Analysis saved to outputs/threshold_analysis.json")

    print("\n" + "="*70)
    print("🎉 THRESHOLD ANALYSIS COMPLETE")
    print("="*70)
    print("\n💡 Next steps:")
    print("  1. Review the recommendations above")
    print("  2. Edit SIMILARITY_THRESHOLD and RERANK_THRESHOLD in rag_langchain_improved.py")
    print("  3. Run python rag_langchain_improved.py to test your settings")
    print("="*70)


if __name__ == "__main__":
    main()
