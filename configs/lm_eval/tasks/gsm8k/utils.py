"""
Utility functions for GSM8K evaluation.
Implements custom metrics that combine filtered and unfiltered evaluations.
"""

import re
import sacrebleu
from rouge_score import rouge_scorer


# Initialize ROUGE scorer once for efficiency
ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"])


def extract_answer_strict(text):
    """
    Extract answer using strict GSM8K format: #### <number>

    Args:
        text: Model output text

    Returns:
        Extracted number as string, or empty string if not found
    """
    match = re.search(r"####\s*(-?[0-9\.\,]+)", text)
    if match:
        # Remove commas and dollar signs for normalization
        return match.group(1).replace(",", "").replace("$", "").strip()
    return ""


def extract_answer_flexible(text):
    """
    Extract answer using flexible extraction: last number in text

    Args:
        text: Model output text

    Returns:
        Extracted number as string, or empty string if not found
    """
    # Find all numbers in the text
    numbers = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)
    if numbers:
        # Get the last number found
        last_num = numbers[-1]
        # numbers is a list of tuples due to the regex groups
        num_str = last_num[0] if last_num[0] else last_num[1]
        # Remove commas and dollar signs for normalization
        return num_str.replace(",", "").replace("$", "").strip()
    return ""


def normalize_answer(answer_str):
    """
    Normalize answer string for comparison.
    Removes commas, dollar signs, and trailing periods.

    Args:
        answer_str: Answer string to normalize

    Returns:
        Normalized string
    """
    if not answer_str:
        return ""

    # Remove specific characters as defined in the metric config
    normalized = answer_str
    normalized = normalized.replace(",", "")
    normalized = normalized.replace("$", "")
    # Remove trailing period
    if normalized.endswith("."):
        normalized = normalized[:-1]

    return normalized.strip().lower()


def exact_match(prediction, reference):
    """
    Compute exact match between prediction and reference.

    Args:
        prediction: Predicted answer string
        reference: Ground truth answer string

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    pred_norm = normalize_answer(prediction)
    ref_norm = normalize_answer(reference)

    if not pred_norm or not ref_norm:
        return 0.0

    return 1.0 if pred_norm == ref_norm else 0.0


def compute_bleu(prediction, reference):
    """
    Compute BLEU score between prediction and reference.

    Args:
        prediction: Predicted text
        reference: Ground truth text

    Returns:
        BLEU score (0-100)
    """
    score = sacrebleu.corpus_bleu(
        [prediction],
        [[reference]],
        smooth_method="exp",
        smooth_value=0.0,
        force=False,
        lowercase=False,
        tokenize="intl",
        use_effective_order=False,
    ).score
    return score


def compute_rouge(prediction, reference):
    """
    Compute ROUGE scores between prediction and reference.

    Args:
        prediction: Predicted text
        reference: Ground truth text

    Returns:
        Dictionary with rouge1, rouge2, rougeL scores (0-100)
    """

    # Prepare texts (add newlines between sentences for rougeLsum)
    def _prepare_summary(summary):
        summary = summary.replace(" . ", ".\n")
        return summary

    pred = _prepare_summary(prediction)
    ref = _prepare_summary(reference)

    scores = ROUGE_SCORER.score(ref, pred)

    return {
        "rouge1": scores["rouge1"].fmeasure * 100,
        "rouge2": scores["rouge2"].fmeasure * 100,
        "rougeL": scores["rougeLsum"].fmeasure * 100,
    }


def process_results(doc, results):
    """
    Process results for GSM8K evaluation.

    Computes:
    1. exact_match with strict-match filter (#### format)
    2. exact_match with flexible-extract filter (last number)
    3. BLEU on unfiltered output
    4. ROUGE-1 on unfiltered output
    5. ROUGE-2 on unfiltered output
    6. ROUGE-L on unfiltered output

    Args:
        doc: Document containing question and answer
        results: List containing model output (typically one element for generate_until)

    Returns:
        Dictionary of metric names to values
    """
    # Get model output
    model_output = results[0] if results else ""

    # Get ground truth answer
    # GSM8K format: "Step by step solution\n#### 42"
    ground_truth_full = doc.get("answer", "")

    # Extract ground truth number (after ####)
    gt_match = re.search(r"####\s*(-?[0-9\.\,]+)", ground_truth_full)
    if gt_match:
        ground_truth_number = (
            gt_match.group(1).replace(",", "").replace("$", "").strip()
        )
    else:
        # Fallback: try to extract last number from ground truth
        ground_truth_number = extract_answer_flexible(ground_truth_full)

    # 1. Extract answer using strict-match filter
    pred_strict = extract_answer_strict(model_output)

    # 2. Extract answer using flexible-extract filter
    pred_flexible = extract_answer_flexible(model_output)

    # Compute exact match metrics
    em_strict = exact_match(pred_strict, ground_truth_number)
    em_flexible = exact_match(pred_flexible, ground_truth_number)

    # Compute BLEU and ROUGE on unfiltered outputs
    bleu_score = compute_bleu(model_output, ground_truth_full)
    rouge_scores = compute_rouge(model_output, ground_truth_full)

    return {
        "exact_match,strict-match": em_strict,
        "exact_match,flexible-extract": em_flexible,
        "bleu": bleu_score,
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
    }
