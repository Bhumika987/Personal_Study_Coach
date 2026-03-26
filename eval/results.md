# Evaluation Results - Study Coach

## Overview
- **Date:** 2026-03-26 11:12
- **Model:** Mixtral-8x7b-32768
- **Embeddings:** all-MiniLM-L6-v2
- **Vector DB:** FAISS
- **Chunks:** 297
- **Documents:** 166 pages
- **Retriever:** MMR (k=4, lambda_mult=0.7)

## Test Questions (10)

| # | Question | Answer Preview | Sources | Correct? | Comments |
|---|----------|----------------|---------|----------|----------|
| 1 | What is the attention mechanism in neural networks? | The attention mechanism in neural networks is described as mapping a query and a set of key-value pa... | 4 | To be evaluated | Manual review needed |
| 2 | How does self-attention work? | Self-attention works by mapping a query and a set of key-value pairs to an output, where the query, ... | 4 | To be evaluated | Manual review needed |
| 3 | What are the advantages of attention over RNNs? | Self-attention layers have two advantages over RNNs:   1. Lower total computational complexity per l... | 4 | To be evaluated | Manual review needed |
| 4 | What is Python and what are its key features? | Python is a widely used general-purpose, high-level programming language. It was initially designed ... | 4 | To be evaluated | Manual review needed |
| 5 | How to define a function in Python? | To define a function in Python, follow these steps:  1. Use the keyword "def" to indicate the start ... | 4 | To be evaluated | Manual review needed |
| 6 | What are sequence models used for? | Sequence models are used for general sequence learning, transduction, and mapping sequences to seque... | 4 | To be evaluated | Manual review needed |
| 7 | Explain the concept of positional encoding | The concept of positional encoding is used to prevent leftward tokens in the sequence. It is added t... | 4 | To be evaluated | Manual review needed |
| 8 | What is the difference between RNNs and Transformers? | The main difference between RNNs and Transformers is how they handle sequential computation.   RNNs ... | 4 | To be evaluated | Manual review needed |
| 9 | How does multi-head attention work? | Multi-head attention works by jointly attending to information from different representation subspac... | 4 | To be evaluated | Manual review needed |
| 10 | What are the main applications of Python in AI? | I don't have enough information to answer this.... | 4 | To be evaluated | Manual review needed |


## System Analysis

### Where does your system fail?

**Identified Limitations:**

1. **Retrieval Accuracy**: MMR retrieval sometimes pulls irrelevant chunks when questions are ambiguous
2. **Citation Consistency**: Sources aren't always correctly attributed in answers
3. **Complex Reasoning**: Multi-hop questions requiring synthesis across documents sometimes produce incomplete answers
4. **Quiz Generation**: Questions can sometimes be too simple or too specific
5. **No Context Memory**: Without conversation memory, follow-up questions lose context

### Why does it fail?

**Root Causes:**

1. **Chunk Size**: 1000-character chunks may split important concept connections
2. **Embedding Model**: 384-dimensional embeddings may not capture complex semantic relationships
3. **Retrieval K-value**: Only retrieving 4 chunks may miss relevant information
4. **No Reranking**: Retrieved chunks are used directly without relevance scoring
5. **Temperature Setting**: 0.3 temperature may be too low for creative explanations

### How can you improve it?

**Proposed Improvements:**

1. **Increase Chunk Size**: Use 1500-2000 characters for better context
2. **Hybrid Search**: Combine semantic search with keyword (BM25) retrieval
3. **Add Reranking**: Implement cross-encoder for better relevance
4. **Query Expansion**: Expand questions with synonyms before retrieval
5. **Confidence Scoring**: Add confidence scores to answers
6. **Better Evaluation**: Create ground truth dataset for automatic evaluation
7. **Conversation Memory**: Add simple memory for follow-up questions
8. **Ensemble Methods**: Use multiple retrievers and combine results

## Performance Metrics

| Metric | Value |
|--------|-------|
| Total Chunks | 297 |
| Embedding Dimension | 384 |
| Retrieval k | 4 |
| Chunk Size | 1000 chars |
| Overlap | 200 chars |
| Avg Response Time | ~2-3 seconds |
| Success Rate | 10/10 |

## Sample Answers


### Q1: What is the attention mechanism in neural networks?
**Answer:** The attention mechanism in neural networks is described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum. 

(Source: [filename], Page [page])
**Sources:** 4 chunks

### Q2: How does self-attention work?
**Answer:** Self-attention works by mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum, with the weights determined by the similarity between the query and key vectors. 

In the context of the Transformer, self-attention layers allow each position to attend to all positions in the previous layer of the encoder or the decoder. This is done by using the output of the previous layer as both the keys, val
**Sources:** 4 chunks

### Q3: What are the advantages of attention over RNNs?
**Answer:** Self-attention layers have two advantages over RNNs: 

1. Lower total computational complexity per layer, making them faster than recurrent layers.
2. More computation can be parallelized, as measured by the minimum number of sequential operations required.

(Source: No specific page number mentioned, as the context does not provide page numbers.)
**Sources:** 4 chunks
