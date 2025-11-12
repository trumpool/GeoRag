#!/usr/bin/env python3
"""
Comprehensive RAG System Evaluation
Evaluates retrieval quality, generation quality, and citation accuracy.
"""

import json
import re
from typing import List, Dict, Tuple
from rag_langchain_improved import GeoRAG


class RAGEvaluator:
    """Evaluate RAG system performance."""

    def __init__(self, rag: GeoRAG):
        self.rag = rag

    # =========================================================================
    # 1. RETRIEVAL EVALUATION
    # =========================================================================

    def evaluate_retrieval(self, query: str, relevant_doc_ids: List[str],
                          initial_k: int = 20, rerank_k: int = 5) -> Dict:
        """
        Evaluate retrieval quality using standard IR metrics.

        Args:
            query: Query text
            relevant_doc_ids: Ground truth relevant document IDs
            initial_k: Number of documents for initial retrieval
            rerank_k: Number of documents after re-ranking

        Returns:
            Dictionary with retrieval metrics
        """
        # Step 1: Initial retrieval
        initial_docs = self.rag.vectorstore.similarity_search(query, k=initial_k)
        initial_doc_ids = [doc.metadata['id'] for doc in initial_docs]

        # Step 2: Re-ranking
        if self.rag.cross_encoder:
            reranked_docs = self.rag.rerank_documents(query, initial_docs, top_k=rerank_k)
            reranked_doc_ids = [doc.metadata['id'] for doc in reranked_docs]
        else:
            reranked_doc_ids = initial_doc_ids[:rerank_k]

        # Calculate metrics for both stages
        initial_metrics = self._calculate_retrieval_metrics(
            retrieved=initial_doc_ids,
            relevant=relevant_doc_ids,
            k=initial_k
        )

        reranked_metrics = self._calculate_retrieval_metrics(
            retrieved=reranked_doc_ids,
            relevant=relevant_doc_ids,
            k=rerank_k
        )

        return {
            "initial_retrieval": initial_metrics,
            "after_reranking": reranked_metrics,
            "retrieved_docs": {
                "initial": initial_doc_ids,
                "reranked": reranked_doc_ids
            }
        }

    def _calculate_retrieval_metrics(self, retrieved: List[str],
                                    relevant: List[str], k: int) -> Dict:
        """Calculate Precision@K, Recall@K, F1@K, MRR."""
        retrieved_set = set(retrieved[:k])
        relevant_set = set(relevant)

        # True Positives
        tp = len(retrieved_set & relevant_set)

        # Precision@K
        precision = tp / k if k > 0 else 0

        # Recall@K
        recall = tp / len(relevant_set) if len(relevant_set) > 0 else 0

        # F1@K
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Mean Reciprocal Rank (MRR)
        mrr = 0
        for i, doc_id in enumerate(retrieved[:k], 1):
            if doc_id in relevant_set:
                mrr = 1 / i
                break

        return {
            "precision@k": round(precision, 4),
            "recall@k": round(recall, 4),
            "f1@k": round(f1, 4),
            "mrr": round(mrr, 4),
            "num_retrieved": k,
            "num_relevant_retrieved": tp,
            "num_ground_truth": len(relevant_set)
        }

    # =========================================================================
    # 2. CITATION EVALUATION
    # =========================================================================

    def evaluate_citations(self, answer: str, retrieved_doc_ids: List[str]) -> Dict:
        """
        Evaluate citation quality in the generated answer.

        Args:
            answer: Generated answer with citations
            retrieved_doc_ids: List of retrieved document IDs

        Returns:
            Citation quality metrics
        """
        # Extract citations from answer
        citation_pattern = r'\[([^\]]+)\]'
        found_citations = re.findall(citation_pattern, answer)

        # Valid citations (match retrieved docs)
        valid_citations = [c for c in found_citations if c in retrieved_doc_ids]

        # Invalid citations (hallucinated)
        invalid_citations = [c for c in found_citations if c not in retrieved_doc_ids]

        # Citation density (citations per sentence)
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        citation_density = len(found_citations) / len(sentences) if sentences else 0

        # Coverage (how many retrieved docs are cited)
        cited_docs = set(valid_citations)
        coverage = len(cited_docs) / len(retrieved_doc_ids) if retrieved_doc_ids else 0

        return {
            "total_citations": len(found_citations),
            "valid_citations": len(valid_citations),
            "invalid_citations": len(invalid_citations),
            "unique_cited_docs": len(cited_docs),
            "citation_accuracy": len(valid_citations) / len(found_citations) if found_citations else 0,
            "citation_density": round(citation_density, 2),
            "source_coverage": round(coverage, 2),
            "has_citations": len(found_citations) > 0,
            "invalid_citation_list": invalid_citations
        }

    # =========================================================================
    # 3. ANSWER QUALITY EVALUATION (requires ground truth)
    # =========================================================================

    def evaluate_answer_quality(self, generated_answer: str,
                               reference_answer: str) -> Dict:
        """
        Evaluate answer quality using simple metrics.
        Note: For proper evaluation, use metrics like ROUGE, BLEU, or BERTScore.

        Args:
            generated_answer: Generated answer
            reference_answer: Ground truth answer

        Returns:
            Answer quality metrics
        """
        # Token overlap (simple precision/recall)
        gen_tokens = set(generated_answer.lower().split())
        ref_tokens = set(reference_answer.lower().split())

        overlap = len(gen_tokens & ref_tokens)

        precision = overlap / len(gen_tokens) if gen_tokens else 0
        recall = overlap / len(ref_tokens) if ref_tokens else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Length comparison
        length_ratio = len(generated_answer) / len(reference_answer) if reference_answer else 0

        return {
            "token_precision": round(precision, 4),
            "token_recall": round(recall, 4),
            "token_f1": round(f1, 4),
            "length_ratio": round(length_ratio, 2),
            "generated_length": len(generated_answer),
            "reference_length": len(reference_answer)
        }

    # =========================================================================
    # 4. COMPREHENSIVE EVALUATION
    # =========================================================================

    def evaluate_query(self, query: str, relevant_doc_ids: List[str],
                      reference_answer: str = None,
                      initial_k: int = 20, rerank_k: int = 5,
                      similarity_threshold: float = None,
                      rerank_threshold: float = None) -> Dict:
        """
        Comprehensive evaluation of a single query.

        Args:
            query: Query text
            relevant_doc_ids: Ground truth relevant document IDs
            reference_answer: Optional ground truth answer
            initial_k: Number of docs for initial retrieval
            rerank_k: Number of docs after re-ranking
            similarity_threshold: Similarity threshold for filtering
            rerank_threshold: Rerank threshold for filtering

        Returns:
            Complete evaluation results
        """
        print(f"\n{'='*70}")
        print(f"Evaluating: {query}")
        print(f"{'='*70}")

        # 1. Retrieval evaluation
        print("  📊 Evaluating retrieval...")
        retrieval_results = self.evaluate_retrieval(
            query, relevant_doc_ids, initial_k, rerank_k
        )

        # 2. Generate answer
        print("  🤖 Generating answer...")
        result = self.rag.query(
            query,
            initial_k=initial_k,
            rerank_k=rerank_k,
            similarity_threshold=similarity_threshold,
            rerank_threshold=rerank_threshold
        )

        # 3. Citation evaluation
        print("  📝 Evaluating citations...")
        citation_results = self.evaluate_citations(
            result['answer'],
            list(result['citations'].keys())
        )

        # 4. Answer quality evaluation (if reference provided)
        answer_quality = None
        if reference_answer:
            print("  ✅ Evaluating answer quality...")
            answer_quality = self.evaluate_answer_quality(
                result['answer'],
                reference_answer
            )

        return {
            "query": query,
            "retrieval_metrics": retrieval_results,
            "citation_metrics": citation_results,
            "answer_quality": answer_quality,
            "generated_answer": result['answer'],
            "num_sources_used": result['num_sources']
        }


def print_evaluation_report(eval_result: Dict):
    """Print a formatted evaluation report."""
    print("\n" + "="*70)
    print("📊 EVALUATION REPORT")
    print("="*70)

    print(f"\n❓ Query: {eval_result['query']}")

    # Retrieval metrics
    print("\n🔍 RETRIEVAL METRICS:")
    print("\n  Initial Retrieval:")
    initial = eval_result['retrieval_metrics']['initial_retrieval']
    print(f"    Precision@K: {initial['precision@k']:.3f}")
    print(f"    Recall@K:    {initial['recall@k']:.3f}")
    print(f"    F1@K:        {initial['f1@k']:.3f}")
    print(f"    MRR:         {initial['mrr']:.3f}")

    print("\n  After Re-ranking:")
    reranked = eval_result['retrieval_metrics']['after_reranking']
    print(f"    Precision@K: {reranked['precision@k']:.3f}")
    print(f"    Recall@K:    {reranked['recall@k']:.3f}")
    print(f"    F1@K:        {reranked['f1@k']:.3f}")
    print(f"    MRR:         {reranked['mrr']:.3f}")

    # Citation metrics
    print("\n📚 CITATION METRICS:")
    citations = eval_result['citation_metrics']
    print(f"    Total Citations:     {citations['total_citations']}")
    print(f"    Valid Citations:     {citations['valid_citations']}")
    print(f"    Invalid Citations:   {citations['invalid_citations']}")
    print(f"    Citation Accuracy:   {citations['citation_accuracy']:.3f}")
    print(f"    Citation Density:    {citations['citation_density']:.2f} per sentence")
    print(f"    Source Coverage:     {citations['source_coverage']:.3f}")

    # Answer quality
    if eval_result['answer_quality']:
        print("\n✨ ANSWER QUALITY:")
        quality = eval_result['answer_quality']
        print(f"    Token F1:        {quality['token_f1']:.3f}")
        print(f"    Length Ratio:    {quality['length_ratio']:.2f}")

    # Generated answer
    print(f"\n💡 GENERATED ANSWER:")
    print(f"    {eval_result['generated_answer']}")

    print("\n" + "="*70)


def main():
    """Example evaluation workflow."""
    print("="*70)
    print("🔬 RAG SYSTEM EVALUATION")
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

    # Initialize models
    rag.initialize_llm()
    rag.initialize_cross_encoder()
    rag.create_qa_chain()

    # Create evaluator
    evaluator = RAGEvaluator(rag)

    # Example evaluation with ground truth
    print("\n2️⃣  Running evaluation...")

    # Example 1: Query with known relevant documents
    query = "What is the relationship between cardiovascular risk factors and Alzheimer's disease genetics?"
    relevant_docs = [
        "30413934_sent_1",
        "30413934_sent_2",
        "30413934_sent_3",
        "30413934_sent_9",
        "30413934_sent_10",
        "30413934_sent_11"
    ]

    eval_result = evaluator.evaluate_query(
        query=query,
        relevant_doc_ids=relevant_docs,
        initial_k=10,
        rerank_k=4
    )

    print_evaluation_report(eval_result)

    # Save evaluation results
    output_path = "outputs/evaluation_results.json"
    with open(output_path, 'w') as f:
        json.dump(eval_result, f, indent=2)
    print(f"\n✅ Evaluation results saved to {output_path}")

    # Batch evaluation on all queries
    print("\n3️⃣  Running batch evaluation on all queries...")
    batch_evaluation_example(evaluator)


def batch_evaluation_example(evaluator: RAGEvaluator):
    """Example of batch evaluation across multiple queries."""

    # Define test queries with ground truth
    test_cases = [
        {
            "query": "What is the relationship between cardiovascular risk factors and Alzheimer's disease genetics?",
            "relevant_docs": ["30413934_sent_1", "30413934_sent_2", "30413934_sent_3",
                            "30413934_sent_9", "30413934_sent_10", "30413934_sent_11"]
        },
        {
            "query": "How do MBLAC1, MINK1, and DDB2 genes relate to Alzheimer's disease?",
            "relevant_docs": ["30413934_sent_6", "30413934_sent_7", "30413934_sent_8"]
        },
        {
            "query": "What are the main findings from the IGAP consortium regarding AD-associated SNPs?",
            "relevant_docs": ["30448613_sent_1", "30448613_sent_2", "30448613_sent_3", "30448613_sent_8"]
        }
    ]

    all_results = []
    metrics_summary = {
        "precision": [],
        "recall": [],
        "f1": [],
        "citation_accuracy": [],
        "has_citations": []
    }

    for test_case in test_cases:
        result = evaluator.evaluate_query(
            query=test_case["query"],
            relevant_doc_ids=test_case["relevant_docs"],
            initial_k=10,
            rerank_k=4
        )
        all_results.append(result)

        # Collect metrics
        reranked = result['retrieval_metrics']['after_reranking']
        metrics_summary["precision"].append(reranked['precision@k'])
        metrics_summary["recall"].append(reranked['recall@k'])
        metrics_summary["f1"].append(reranked['f1@k'])
        metrics_summary["citation_accuracy"].append(result['citation_metrics']['citation_accuracy'])
        metrics_summary["has_citations"].append(result['citation_metrics']['has_citations'])

    # Calculate averages
    print("\n" + "="*70)
    print("📈 BATCH EVALUATION SUMMARY")
    print("="*70)
    print(f"  Total Queries: {len(test_cases)}")
    print(f"\n  Average Metrics (after re-ranking):")
    print(f"    Avg Precision@K:       {sum(metrics_summary['precision'])/len(metrics_summary['precision']):.3f}")
    print(f"    Avg Recall@K:          {sum(metrics_summary['recall'])/len(metrics_summary['recall']):.3f}")
    print(f"    Avg F1@K:              {sum(metrics_summary['f1'])/len(metrics_summary['f1']):.3f}")
    print(f"    Avg Citation Accuracy: {sum(metrics_summary['citation_accuracy'])/len(metrics_summary['citation_accuracy']):.3f}")
    print(f"    Queries with Citations: {sum(metrics_summary['has_citations'])}/{len(test_cases)}")
    print("="*70)

    # Save batch results
    with open("outputs/batch_evaluation.json", 'w') as f:
        json.dump({
            "summary": metrics_summary,
            "all_results": all_results
        }, f, indent=2)
    print("\n✅ Batch evaluation saved to outputs/batch_evaluation.json")


if __name__ == "__main__":
    main()
