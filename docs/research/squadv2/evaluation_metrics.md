# SQuAD v2 Evaluation Metrics

**Generated:** 2025-01-15  
**Topic:** Common Evaluation Metrics for SQuAD 2.0  
**Tags:** #research #squadv2 #metrics #qa

## Overview

SQuAD 2.0 (Stanford Question Answering Dataset) differs from v1.1 by introducing **unanswerable questions**—questions that look similar to answerable ones but cannot be answered solely from the provided passage. Consequently, the evaluation metrics must account for the model's ability to abstain from answering.

## Primary Metrics

The two official leaderboard metrics are computed over the entire evaluation set (both answerable and unanswerable questions).

### 1. Exact Match (EM)
*   **Definition**: A binary measure (0 or 1) that checks if the model's predicted string *exactly* matches one of the ground truth answers (after basic normalization like lowercasing and removing punctuation).
*   **For Unanswerable Questions**: The ground truth is an empty string. If the model predicts an empty string (abstains), it gets a score of 1.
*   **Calculation**:
    $$ EM = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\text{prediction}_i = \text{ground\_truth}_i) $$

### 2. F1 Score
*   **Definition**: A looser metric that measures the average overlap between the prediction and the ground truth answer at the token level. It is the harmonic mean of Precision and Recall.
*   **For Answerable Questions**: Calculated as the maximum F1 over all ground truth answers (since a question might have multiple valid phrasings).
*   **For Unanswerable Questions**: 
    *   If the model predicts empty: F1 = 1.0
    *   If the model predicts *something*: F1 = 0.0
*   **Significance**: F1 is typically the primary ranking metric as it gives partial credit for nearly correct answers.

## Component Metrics (Detailed Breakdown)

To diagnose model behavior, performance is often broken down into two subsets:

### 3. HasAns (Answerable)
*   **HasAns_EM / HasAns_F1**: The EM and F1 scores calculated *only* on the subset of questions that actually have answers.
*   **Usage**: Measures the model's precision in extracting spans when it *knows* there is an answer.

### 4. NoAns (Unanswerable)
*   **NoAns_EM / NoAns_F1**: The EM/F1 scores calculated *only* on the subset of unanswerable questions.
*   **Equivalence**: Since the target is always an empty string, `NoAns_EM` and `NoAns_F1` are identical.
*   **Usage**: Effectively measures the model's **abstention rate** or "hallucination resistance."

## Auxiliary Metrics

### 5. AvNA (Answer vs. No Answer)
*   **Definition**: A binary classification accuracy metric that measures how well the model distinguishes between answerable and unanswerable questions, ignoring whether the extracted span is correct.
*   **Calculation**: It considers a prediction "correct" if the model correctly predicts "empty" for a NoAns question OR predicts *any* non-empty string for a HasAns question.
*   **Significance**: This isolates the "detection" capability from the "extraction" capability. A model might have high AvNA but low EM if it correctly identifies answerable questions but extracts the wrong span.

### 6. Thresholding (Null Score Diff)
*   **Context**: Most QA models (like BERT/RoBERTa) output a score for the best span ($s_{span}$) and a score for the "no-answer" token ($s_{null}$).
*   **Mechanism**: A prediction is made only if $s_{span} - s_{null} > \tau$.
*   **Best F1 / Best EM**: Leaderboards often report the "Best F1" achieved by tuning this threshold $\tau$ on the development set, rather than using a fixed default.

## Summary Table

| Metric | Scope | Description |
| :--- | :--- | :--- |
| **EM (Exact Match)** | Overall | % of predictions matching ground truth exactly. |
| **F1 Score** | Overall | Harmonic mean of token overlap precision/recall. |
| **HasAns_F1** | Answerable Only | Span extraction quality on valid questions. |
| **NoAns_EM** | Unanswerable Only | Ability to correctly abstain (predict empty). |
| **AvNA** | Classification | Accuracy of the binary "Answerable?" classifier. |

## References
*   Rajpurkar et al., *"Know What You Don't Know: Unanswerable Questions for SQuAD"* (ACL 2018).
*   HuggingFace Evaluate Documentation: `metrics/squad_v2`.
