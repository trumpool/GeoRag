# Threshold Filtering Guide

## Overview

Threshold filtering helps you **exclude irrelevant documents** from the retrieval process, ensuring only high-quality, relevant information is passed to the LLM for answer generation.

---

## Why Use Thresholds?

**Without thresholds:**
- ❌ Irrelevant documents may be retrieved
- ❌ LLM gets confused by off-topic context
- ❌ Lower answer quality
- ❌ Potential hallucinations

**With thresholds:**
- ✅ Only relevant documents pass through
- ✅ Cleaner context for LLM
- ✅ Better answer quality
- ✅ More accurate citations

---

## Two-Stage Filtering

### Stage 1: Similarity Threshold (Bi-Encoder)

**What it does:**
- Filters documents after initial retrieval using cosine similarity
- Based on embedding distance (FAISS L2 distance)

**Parameter:** `similarity_threshold`
- **Range:** 0-1 (lower values = more similar)
- **Default:** `None` (no filtering)
- **Typical values:** 0.3-0.7

**Example:**
```python
result = rag.query(
    "What genes are associated with AD?",
    initial_k=10,
    similarity_threshold=0.5  # Only keep docs with score ≤ 0.5
)
```

### Stage 2: Rerank Threshold (Cross-Encoder)

**What it does:**
- Filters documents after re-ranking using cross-encoder scores
- More accurate than cosine similarity

**Parameter:** `rerank_threshold`
- **Range:** -5 to 5 (higher = better match)
- **Default:** `None` (no filtering)
- **Typical values:** -1.0 to 2.0

**Example:**
```python
result = rag.query(
    "What genes are associated with AD?",
    initial_k=10,
    rerank_k=4,
    rerank_threshold=0.0  # Only keep docs with score ≥ 0.0
)
```

---

## How to Find Optimal Thresholds

### Step 1: Run Threshold Analysis

```bash
python tune_thresholds.py
```

This will:
1. Analyze similarity score distributions across your queries
2. Analyze cross-encoder score distributions
3. Test different threshold combinations
4. Provide recommendations

### Step 2: Review Output

**Similarity Scores:**
```
📊 ANALYZING SIMILARITY SCORES
======================================================================

Overall Statistics:
  Mean:   0.4532
  Median: 0.4421
  75th Percentile: 0.5234

💡 RECOMMENDATIONS:
  Conservative (keep most docs):  threshold < 0.52
  Moderate (balanced):            threshold < 0.44
  Aggressive (only best matches): threshold < 0.35
```

**Rerank Scores:**
```
📊 ANALYZING CROSS-ENCODER RERANK SCORES
======================================================================

Overall Statistics:
  Mean:   1.2345
  Median: 1.5678
  25th Percentile: 0.8912

💡 RECOMMENDATIONS:
  Conservative (keep most docs):  threshold > 0.89
  Moderate (balanced):            threshold > 1.57
  Aggressive (only best matches): threshold > 2.13
```

### Step 3: Test Different Thresholds

The script will show you how many documents pass each threshold:

```
🧪 TESTING THRESHOLD COMBINATIONS
======================================================================

  Sim=  None, Rerank=  None → 10 initial → 4 final docs
  Sim=  0.50, Rerank=  None →  8 initial → 4 final docs
  Sim=  0.50, Rerank=  0.00 →  8 initial → 3 final docs
  Sim=  0.40, Rerank=  1.00 →  6 initial → 2 final docs
```

💡 **Choose thresholds that give you 3-5 final documents**

---

## Configuring Thresholds

### In Main Script

Edit `rag_langchain_improved.py`:

```python
# Find this section around line 541
INITIAL_K = 10
RERANK_K = 4

# Set your thresholds
SIMILARITY_THRESHOLD = 0.5    # Filter after initial retrieval
RERANK_THRESHOLD = 0.0        # Filter after re-ranking
```

### In Custom Code

```python
from rag_langchain_improved import GeoRAG

rag = GeoRAG()
# ... initialize ...

result = rag.query(
    query="Your question",
    initial_k=10,
    rerank_k=4,
    similarity_threshold=0.5,   # Optional
    rerank_threshold=0.0        # Optional
)
```

---

## Threshold Selection Guide

### For High Precision (Scientific/Medical RAG)

**Goal:** Only retrieve highly relevant documents

```python
SIMILARITY_THRESHOLD = 0.4    # Strict initial filter
RERANK_THRESHOLD = 1.0        # Only keep high-scoring docs
```

**Expected:** 2-4 documents per query, all highly relevant

### For Balanced Retrieval

**Goal:** Balance precision and recall

```python
SIMILARITY_THRESHOLD = 0.5    # Moderate filter
RERANK_THRESHOLD = 0.0        # Keep reasonably scored docs
```

**Expected:** 3-5 documents per query, mostly relevant

### For High Recall

**Goal:** Don't miss any potentially relevant information

```python
SIMILARITY_THRESHOLD = 0.7    # Loose filter
RERANK_THRESHOLD = -1.0       # Very permissive
```

**Expected:** 4-6 documents per query, may include some marginally relevant docs

### No Filtering (Default)

**Goal:** Let the LLM decide what's relevant

```python
SIMILARITY_THRESHOLD = None   # No filtering
RERANK_THRESHOLD = None       # No filtering
```

**Expected:** All retrieved documents are used

---

## Safety Features

The system includes automatic fallbacks:

1. **No documents pass threshold**
   ```
   ⚠️  No documents passed threshold! Using top document anyway.
   ```
   → At least 1 document is always returned

2. **Too few documents**
   ```
   ⚠️  Only 2 docs passed threshold (< rerank_k=4)
   ```
   → Warning shown, but continues with available docs

3. **Rerank threshold too strict**
   ```
   ⚠️  Rerank threshold too strict! Using top document anyway.
   ```
   → Prevents zero-document edge cases

---

## Monitoring Threshold Effects

### During Execution

Watch for threshold messages in output:

```
📥 Retrieving top-10 documents...
📊 Similarity threshold 0.5: 7/10 docs passed
🔄 Re-ranked 7 → 4 documents
📊 Score threshold 0.0: 3/4 docs passed
```

### In Evaluation

Compare metrics with/without thresholds:

```python
# Without thresholds
result1 = evaluator.evaluate_query(query, relevant_docs)

# With thresholds
result2 = evaluator.evaluate_query(
    query, relevant_docs,
    similarity_threshold=0.5,
    rerank_threshold=0.0
)

# Compare citation accuracy, precision, etc.
```

---

## Best Practices

### 1. Start Conservative
```python
# Begin with loose thresholds
SIMILARITY_THRESHOLD = 0.6
RERANK_THRESHOLD = -0.5
```

### 2. Monitor Performance
- Check how many documents are filtered out
- Evaluate answer quality
- Look for missing relevant information

### 3. Tighten Gradually
```python
# If too many irrelevant docs, tighten
SIMILARITY_THRESHOLD = 0.5  # was 0.6
RERANK_THRESHOLD = 0.0      # was -0.5
```

### 4. Domain-Specific Tuning
- Medical/Scientific: Strict thresholds (high precision)
- General Q&A: Moderate thresholds (balanced)
- Exploratory: Loose thresholds (high recall)

### 5. Test on Sample Queries
Run `tune_thresholds.py` whenever you:
- Add new documents to corpus
- Change embedding model
- Change cross-encoder model
- Work with different domain

---

## Troubleshooting

### Too Many Documents Filtered Out

**Problem:** Most queries return only 1-2 documents

**Solution:**
```python
# Relax thresholds
SIMILARITY_THRESHOLD = 0.7    # was 0.4
RERANK_THRESHOLD = -1.0       # was 1.0
```

### Still Getting Irrelevant Documents

**Problem:** Answers contain off-topic information

**Solution:**
```python
# Tighten thresholds
SIMILARITY_THRESHOLD = 0.4    # was 0.6
RERANK_THRESHOLD = 1.0        # was 0.0
```

### Answers Missing Information

**Problem:** Relevant documents are being filtered out

**Solution:**
1. Check `tune_thresholds.py` output
2. Lower thresholds to be more inclusive
3. Verify relevant documents exist in corpus
4. Consider improving query formulation

---

## Example Workflow

```bash
# 1. Analyze score distributions
python tune_thresholds.py

# 2. Review recommendations in terminal output
# Output saved to: outputs/threshold_analysis.json

# 3. Edit rag_langchain_improved.py with chosen thresholds
# Set SIMILARITY_THRESHOLD and RERANK_THRESHOLD

# 4. Test with your queries
python rag_langchain_improved.py

# 5. Evaluate results
python evaluate_rag.py

# 6. Adjust and repeat if needed
```

---

## Advanced: Dynamic Thresholds

For production systems, consider adjusting thresholds per query:

```python
# Easy queries: strict thresholds
if is_factual_query(query):
    sim_thresh, rerank_thresh = 0.4, 1.0

# Complex queries: relaxed thresholds
else:
    sim_thresh, rerank_thresh = 0.6, 0.0

result = rag.query(
    query,
    similarity_threshold=sim_thresh,
    rerank_threshold=rerank_thresh
)
```

---

## Summary

| Threshold | Range | Purpose | Recommended |
|-----------|-------|---------|-------------|
| **Similarity** | 0-1 (lower=better) | Filter after bi-encoder | 0.4-0.6 |
| **Rerank** | -5 to 5 (higher=better) | Filter after cross-encoder | -0.5 to 1.0 |

**Key Takeaway:** Use `tune_thresholds.py` to find optimal values for your specific corpus and queries!
