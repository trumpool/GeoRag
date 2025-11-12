# RAG System Evaluation Guide

## Overview

This guide explains how to evaluate your RAG (Retrieval-Augmented Generation) system using multiple metrics and methodologies.

## Evaluation Dimensions

### 1. Retrieval Quality (Information Retrieval Metrics)

**Metrics:**
- **Precision@K**: Proportion of retrieved documents that are relevant
  ```
  Precision@K = (Relevant Retrieved Docs) / K
  ```

- **Recall@K**: Proportion of relevant documents that are retrieved
  ```
  Recall@K = (Relevant Retrieved Docs) / (Total Relevant Docs)
  ```

- **F1@K**: Harmonic mean of Precision and Recall
  ```
  F1@K = 2 * (Precision * Recall) / (Precision + Recall)
  ```

- **MRR (Mean Reciprocal Rank)**: Position of first relevant document
  ```
  MRR = 1 / (Rank of First Relevant Doc)
  ```

**What to Look For:**
- ✅ High Precision@K (>0.7): Most retrieved documents are relevant
- ✅ High Recall@K (>0.6): Most relevant documents are retrieved
- ✅ Improvement after re-ranking: Cross-encoder boosts quality

---

### 2. Citation Quality

**Metrics:**
- **Citation Accuracy**: Valid citations / Total citations
- **Citation Density**: Citations per sentence
- **Source Coverage**: Retrieved docs that are cited / Total retrieved docs
- **Hallucination Rate**: Invalid citations / Total citations

**What to Look For:**
- ✅ Citation Accuracy > 0.9 (no hallucinated citations)
- ✅ Citation Density ≈ 0.8-1.5 (proper citation frequency)
- ✅ Source Coverage > 0.5 (most sources are used)
- ❌ Invalid citations = 0 (no hallucinations)

---

### 3. Answer Quality (requires ground truth)

**Metrics:**
- **Token Overlap F1**: Overlap between generated and reference answer
- **ROUGE Score**: Measures n-gram overlap (requires rouge-score package)
- **BERTScore**: Semantic similarity using BERT embeddings (requires bert-score package)

**What to Look For:**
- ✅ Token F1 > 0.5 (reasonable content overlap)
- ✅ Answers are factually correct (manual check)
- ✅ Answers are coherent and well-structured

---

## How to Run Evaluation

### Quick Start

```bash
# Run the evaluation script
python evaluate_rag.py
```

This will:
1. Initialize the RAG system
2. Evaluate sample queries
3. Generate evaluation reports
4. Save results to `outputs/evaluation_results.json`

---

### Custom Evaluation

```python
from evaluate_rag import RAGEvaluator
from rag_langchain_improved import GeoRAG

# Initialize RAG
rag = GeoRAG()
documents = rag.load_corpus("corpus.jsonl")
rag.build_vectorstore(documents)
rag.initialize_llm()
rag.initialize_cross_encoder()
rag.create_qa_chain()

# Create evaluator
evaluator = RAGEvaluator(rag)

# Evaluate a single query
query = "Your question here"
relevant_doc_ids = ["doc1_sent_1", "doc2_sent_3"]  # Ground truth

result = evaluator.evaluate_query(
    query=query,
    relevant_doc_ids=relevant_doc_ids,
    initial_k=10,
    rerank_k=4
)

# Print report
from evaluate_rag import print_evaluation_report
print_evaluation_report(result)
```

---

## Creating Ground Truth Data

### Option 1: Manual Annotation

Create a file `ground_truth.jsonl`:

```json
{"query": "What genes are associated with AD?", "relevant_docs": ["30617256_sent_5", "30617256_sent_6"]}
{"query": "How does age affect AD genetics?", "relevant_docs": ["30979435_sent_3", "30979435_sent_5"]}
```

### Option 2: Expert Review

1. Run queries through the system
2. Have domain experts review retrieved documents
3. Mark which documents are truly relevant
4. Use these as ground truth for evaluation

---

## Evaluation Best Practices

### 1. Test Set Design
- ✅ Use diverse queries (factual, analytical, comparison)
- ✅ Include easy and hard queries
- ✅ Cover different topics in your corpus
- ✅ Minimum 20-30 test queries for reliability

### 2. Metrics to Prioritize

**For Scientific/Medical RAG:**
- Citation Accuracy (most important - no hallucinations!)
- Precision@K (retrieved docs must be relevant)
- Answer Factuality (manual check against sources)

**For General RAG:**
- F1@K (balance precision and recall)
- Citation Coverage (use diverse sources)
- Answer Quality (semantic similarity to reference)

### 3. Iterative Improvement

```
Evaluate → Identify Weaknesses → Adjust → Re-evaluate

Common Adjustments:
- Chunk size (currently sentence-level)
- Initial K and rerank K values
- Prompt template
- Retrieval model (embedding model)
- Re-ranking model (cross-encoder)
- LLM model
```

---

## Benchmark Targets

Based on research literature, good RAG systems should achieve:

| Metric | Target | Excellent |
|--------|--------|-----------|
| Precision@5 | >0.6 | >0.8 |
| Recall@5 | >0.5 | >0.7 |
| Citation Accuracy | >0.85 | >0.95 |
| MRR | >0.5 | >0.8 |

---

## Advanced Evaluation (Optional)

### Install Additional Libraries

```bash
pip install rouge-score bert-score
```

### Use ROUGE and BERTScore

```python
from rouge_score import rouge_scorer
from bert_score import score

# ROUGE evaluation
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference_answer, generated_answer)

# BERTScore evaluation
P, R, F1 = score([generated_answer], [reference_answer], lang='en')
```

---

## Evaluation Outputs

### Console Output
```
📊 EVALUATION REPORT
======================================================================

❓ Query: What is the relationship between cardiovascular...

🔍 RETRIEVAL METRICS:
  Initial Retrieval:
    Precision@K: 0.700
    Recall@K:    0.583
    F1@K:        0.636
    MRR:         1.000

  After Re-ranking:
    Precision@K: 0.750
    Recall@K:    0.500
    F1@K:        0.600
    MRR:         1.000

📚 CITATION METRICS:
    Total Citations:     5
    Valid Citations:     5
    Invalid Citations:   0
    Citation Accuracy:   1.000
    Citation Density:    0.83 per sentence
    Source Coverage:     0.750
```

### JSON Output (outputs/evaluation_results.json)
```json
{
  "query": "...",
  "retrieval_metrics": {...},
  "citation_metrics": {...},
  "generated_answer": "..."
}
```

---

## Troubleshooting

### Low Precision
- Corpus may be too diverse
- Increase specificity in queries
- Adjust embedding model

### Low Recall
- Increase initial K value
- Check if relevant docs exist in corpus
- Improve query formulation

### Poor Citation Accuracy
- Strengthen prompt instructions
- Increase max_retries
- Consider different LLM model

### Low Source Coverage
- Some retrieved docs may not be used
- LLM may be cherry-picking sources
- Normal if some docs are redundant

---

## Next Steps

1. ✅ Run baseline evaluation with `evaluate_rag.py`
2. ✅ Create ground truth annotations for your queries
3. ✅ Run batch evaluation on 20+ queries
4. ✅ Analyze results and identify weak points
5. ✅ Iterate on system design
6. ✅ Re-evaluate to measure improvement

---

## References

- [BEIR Benchmark](https://github.com/beir-cellar/beir) - Standard IR evaluation
- [RAGAS](https://github.com/explodinggradients/ragas) - RAG evaluation framework
- [TruLens](https://github.com/truera/trulens) - LLM application evaluation
