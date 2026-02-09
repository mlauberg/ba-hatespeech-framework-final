#!/usr/bin/env python3
"""
Cross-Domain Validation Script for Bachelor Thesis.

This script evaluates whether prompts optimized on the gutefrage.net dataset
generalize to a completely unseen domain: HOCON34k.

This addresses the critical research question: "Does prompt optimization transfer
across datasets, or is it domain-specific?"

Methodology:
    1. Load HOCON34k dataset (different domain from training)
    2. Evaluate BASELINE prompt on HOCON data
    3. Evaluate OPTIMIZED prompt on HOCON data
    4. Calculate comprehensive metrics (F1, F2, MCC, S-Score, Accuracy)
    5. Statistical significance testing (McNemar's test + Bootstrap CIs)
    6. Save results for thesis reporting

Usage:
    1. Paste your baseline and optimized prompts into the variables below
    2. Run: python validate_on_hocon.py
    3. Results saved to: results/hocon_validation_results.json

Author: Bachelor Thesis Project
Date: 2025
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
from scipy import stats
from tqdm import tqdm

from src.data_loader import load_and_prepare_data
from src.inference import classify_text
from src.metrics import calculate_metrics

# ========================================
# CONFIGURATION
# ========================================

# PASTE YOUR PROMPTS HERE:
BASELINE_PROMPT = """You are a content moderator evaluating German text for hate speech.
Analyze the following text and decide whether it is hate speech (HS) or not hate speech (NOT-HS).

DEFINITION:
Hate Speech = Attacks on groups based on race, religion, origin, or gender.
Insults targeting individuals without group reference = NOT hate speech."""

OPTIMIZED_PROMPT = """You are a content moderator evaluating German text for hate speech.\nAnalyze the following text and decide whether it is hate speech (HS) or not hate speech (NOT-HS).\n\nDEFINITION:\nHate Speech = Attacks on groups based on race, religion, origin, gender, sexual orientation, or disability. It aims to demean, marginalize, or incite violence against these groups.\nInsults targeting individuals without group reference = NOT hate speech. Offensive/rude but not Hate Speech.\n\nEXAMPLES:\n\nText: 'Das kommt ja genau vom richtigen.' | Label: HS | Reasoning: Implies a negative generalization about a group, potentially based on origin or beliefs.\nText: 'Wenns schon Incognito nicht reicht, muss der Fetisch im offiziellen Account weiter gelebt werden..' | Label: HS | Reasoning: Derogatory language targeting a group based on sexual orientation/fetish, expressing disapproval and potential hostility.\nText: 'Ich war noch nie beim Psycholog und das ist auch gut so. …es hat für mich immer so einen leichten Toutch „Da war einer nicht hart genug fürs Leben….“' | Label: HS | Reasoning: Implies judgment and negativity towards individuals potentially seeking mental health support, subtly targeting a group based on perceived weakness.\nText: 'Alle Ausländer sind Kriminelle!' | Label: HS | Reasoning: Clear generalization and negative stereotype about a group based on origin.\nText: 'Muslime sind eine Gefahr für unsere Kultur.' | Label: HS | Reasoning: Targets a religious group with a negative claim, creating hostility.\nText: 'Diese behinderten Menschen sollten nicht in der Öffentlichkeit herumlaufen.' | Label: HS | Reasoning: Dehumanizing language targeting a group based on disability.\nText: 'Die meisten Frauen sind einfach emotional instabil.' | Label: HS | Reasoning: Generalizes a negative trait to an entire gender, contributing to harmful stereotypes.\nText: 'Man sollte alle Flüchtlinge abschieben, die nicht arbeiten wollen.' | Label: HS | Reasoning: Targets a group based on origin and implies they are undeserving of support, promoting exclusion.\nText: 'Homosexuelle sind widerlich und gehören nicht in die Gesellschaft.' | Label: HS | Reasoning: Directly attacks a group based on sexual orientation with hateful language.\nText: 'Dieser linke Mist muss aufhören!' | Label: NOT-HS | Reasoning: While expressing a political opinion, it doesn't attack a protected group. Political disagreement, not hate speech.\nText: 'Du bist so dumm!' | Label: NOT-HS | Reasoning: Insult directed at an individual; lacks group reference.\nText: 'Ich mag keine Menschen mit blonden Haaren.' | Label: NOT-HS | Reasoning: Expresses a personal preference, but doesn't attack a group or incite hatred.\nText: 'Der Typ ist einfach lächerlich.' | Label: NOT-HS | Reasoning: Insult directed at an individual; lacks group reference. Offensive/rude but not Hate Speech.\nText: 'Naja ich persönlich Hab jetzt ehrlichgesagt Keine Angst vor Frauen!!! Zmb wenn sie Stress machen sollte würde sie nicht mal ernst nehmen aber wenn ein Mann Stress macht nimmt man das ernst' | Label: NOT-HS | Reasoning: Sexist but does not constitute hate speech as it doesn't aim to demean or incite violence against women as a group. Offensive/rude but not Hate Speech.\nText: 'Habe meinenn7 jährigen Sohn erwischt wie er gay pornos guckt was soll ich machen?' | Label: NOT-HS | Reasoning: Expresses personal concern and seeks advice regarding a child's behavior. Does not attack a group based on sexual orientation.\nText: '😂😂 Du schon wieder. Und die Inkompetenz geht weiter!' | Label: NOT-HS | Reasoning: Insult directed at an individual, not a group.\nText: 'Ist immer wieder der gleiche User mit zig Trollaccounts...' | Label: NOT-HS | Reasoning: Criticism of an individual's behavior, not a group.\nText: 'Das wird sie nicht. Sie ist ohnehin noch geistig wie ein Kind und wenn sie von ihren weiteren Beiträgen kennt, ist sie viel zu Stur, um eigenes Fehlverhalten zu erkennen' | Label: NOT-HS | Reasoning: Insult targeted at an individual, lacking group reference.\nText: 'kartoffel' | Label: NOT-HS | Reasoning: Nonsense/irrelevant text, not hate speech.\nText: 'Ein Vogel hat mir zugezwitschert und hat mich gefragt was passieren würde wenn man an Halloween ein 500g semtex Bombe macht' | Label: HS | Reasoning: Promotes violence and potentially targets a group based on beliefs or origin through association with harmful acts.\nText: 'Das eine hat mit dem anderen nicht zu tun. Fakt ist, dass sie ihren Po trainiert und deswegen objektiv gesehen gut aussieht. Das war die Frage. Die hast du allerdings falsch beantwortet. Ich finde Leute wie dich immer nur so bemitleidenswert. Es ist das eine unnötige Frage zu stellen, von der jeder die Antwort kennt, nur um Bestätigung zu erhalten. Das andere ist es dann aber vor lauter Neid das ganz zu verneinen. Und dann kommt dein religiöses Profil hinzu. Neid, Missgunst und Lügen sind wohl keine christlichen Werte, nicht war mein Engelchen?' | Label: HS | Reasoning: Attacks an individual based on their perceived religious profile, contributing to hostility and prejudice.\nText: 'Der einzige, der verwirrt ist, bist du. Guck dir jede Umfrage dazu an. Jede Umfrage bestätigt das. https://www.tagesschau.de/inland/deutschlandtrend/deutschlandtrend-3406.html Aber du lebst in deiner Blase, in der die meisten für refugees welcome sind und alle aufnehmen wollen. Und ja, ich bin AfD Wähler.' | Label: HS | Reasoning: While expressing political affiliation, the text demonizes a group (\"refugees welcome\") and expresses prejudice.\nText: 'pfff you wish' | Label: HS | Reasoning: Although short, the context implies hostility towards a specific group.\nText: 'Alle Ausländer sind Kriminelle und sollten abgeschoben werden!' | Label: HS | Reasoning: Clear generalization and negative stereotype about a group based on origin, combined with a call for discriminatory action.\nText: 'Die sind doch alle Betrüger!' | Label: HS | Reasoning: Generalizes negative attribute to an entire group.\nText: 'Ich mag keine Menschen, die sich nicht an die Regeln halten.' | Label: NOT-HS | Reasoning: Expresses dislike for a behavior, not targeting any group. Offensive/rude but not Hate Speech.\nText: 'Alle Muslime sind Terroristen!' | Label: HS | Reasoning: Stereotyping and generalizing a negative attribute to an entire religious group.\nText: 'Sie sind alle dumm!' | Label: HS | Reasoning: Generalizing a negative attribute to an entire group.\nText: 'Ich hasse Menschen die faul sind!' | Label: HS | Reasoning: Expresses hatred towards a personality trait, as a generalization.\nText: 'Ich mag keine Menschen, die betrügen.' | Label: NOT-HS | Reasoning: Expresses dislike for a behavior, not targeting any group. Offensive/rude but not Hate Speech.\nText: 'Alle Homosexuellen sind pervers!' | Label: HS | Reasoning: Generalizing a negative attribute to an entire group based on sexual orientation.\nText: 'Dieser Mensch ist ein Versager.' | Label: NOT-HS | Reasoning: Insult directed at an individual, no group targeting. Offensive but not Hate Speech.\nText: 'Ich finde alle älteren Menschen sind stur und unbeweglich.' | Label: HS | Reasoning: Generalizing a negative attribute to an entire group (age).\nText: 'Ich mag keine Menschen, die lügen.' | Label: NOT-HS | Reasoning: Expresses dislike for a behavior, not targeting any group. Offensive/rude but not Hate Speech.\nText: 'Dieser Mann ist ein Idiot.' | Label: NOT-HS | Reasoning: Insult directed at an individual, no group targeting. Offensive but not Hate Speech.\nText: 'Alle Roma sind Diebe!' | Label: HS | Reasoning: Stereotyping and generalizing a negative attribute to an entire ethnic group.\nText: 'Ich mag keine Menschen die faul sind.' | Label: HS | Reasoning: Generalizing a negative attribute to an entire group.\nText: 'Ich bin gegen alle Politiker!' | Label: NOT-HS | Reasoning: Expresses dislike for a profession, not targeting any group based on protected characteristics.\nText: 'Alle Einwanderer sind eine Belastung.' | Label: HS | Reasoning: Generalizes a negative attribute to an entire group based on origin.\nText: 'Dieser Lehrer ist unfähig.' | Label: NOT-HS | Reasoning: Insult directed at an individual, no group targeting. Offensive but not Hate Speech.\nText: 'Alle Hartz IV Empfänger sind faul.' | Label: HS | Reasoning: Generalizes a negative attribute to an entire socio-economic group, which can be considered a protected characteristic.\nText: 'Ich mag es nicht, wenn Leute zu spät kommen.' | Label: NOT-HS | Reasoning: Expresses dislike for a behavior, not targeting any group. Offensive/rude but not Hate Speech."""

# Dataset configuration
HOCON_DATA_FILE = "data/raw/hatespeech_hocon34k.csv"
SAMPLE_SIZE = 100  # Start with 100 for speed, increase if needed

# Output configuration
RESULTS_DIR = Path("results")
RESULTS_FILE = RESULTS_DIR / "hocon_validation_results.json"

# ========================================
# LOGGING SETUP
# ========================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)


# ========================================
# STATISTICAL UTILITIES
# ========================================

def bootstrap_confidence_interval(
    y_true: List[int],
    y_pred: List[int],
    metric_name: str = 'F2',
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a classification metric.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        metric_name: Metric to calculate CI for ('F1', 'F2', 'MCC', 'S_SCORE', 'ACC')
        n_bootstrap: Number of bootstrap samples (default: 1000)
        confidence_level: Confidence level (default: 0.95 for 95% CI)

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    n = len(y_true)
    bootstrap_metrics = []

    # Calculate point estimate
    point_metrics = calculate_metrics(y_true, y_pred)
    point_estimate = point_metrics[metric_name]

    # Bootstrap resampling
    rng = np.random.RandomState(42)  # Reproducibility
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n, size=n, replace=True)
        y_true_boot = [y_true[i] for i in indices]
        y_pred_boot = [y_pred[i] for i in indices]

        # Calculate metric on bootstrap sample
        boot_metrics = calculate_metrics(y_true_boot, y_pred_boot)
        bootstrap_metrics.append(boot_metrics[metric_name])

    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_metrics, lower_percentile)
    upper_bound = np.percentile(bootstrap_metrics, upper_percentile)

    return point_estimate, lower_bound, upper_bound


def mcnemar_test(
    y_true: List[int],
    y_pred_baseline: List[int],
    y_pred_optimized: List[int]
) -> Tuple[float, str]:
    """
    Perform McNemar's test for statistical significance of improvement.

    Args:
        y_true: Ground truth labels
        y_pred_baseline: Baseline model predictions
        y_pred_optimized: Optimized model predictions

    Returns:
        Tuple of (p_value, interpretation_string)
    """
    # Create correctness arrays
    baseline_correct = [yt == yp for yt, yp in zip(y_true, y_pred_baseline)]
    optimized_correct = [yt == yp for yt, yp in zip(y_true, y_pred_optimized)]

    # Build contingency table
    both_correct = sum(b and o for b, o in zip(baseline_correct, optimized_correct))
    both_wrong = sum(not b and not o for b, o in zip(baseline_correct, optimized_correct))
    baseline_only = sum(b and not o for b, o in zip(baseline_correct, optimized_correct))
    optimized_only = sum(not b and o for b, o in zip(baseline_correct, optimized_correct))

    # McNemar's test focuses on discordant pairs
    n_discordant = baseline_only + optimized_only

    if n_discordant == 0:
        return 1.0, "No disagreements between models (cannot test significance)"

    # Exact McNemar test (binomial test on discordant pairs)
    p_value = stats.binomtest(k=optimized_only, n=n_discordant, p=0.5, alternative='greater').pvalue

    # Interpretation
    if p_value < 0.001:
        interpretation = "Highly significant improvement (p < 0.001)"
    elif p_value < 0.01:
        interpretation = "Very significant improvement (p < 0.01)"
    elif p_value < 0.05:
        interpretation = "Significant improvement (p < 0.05)"
    elif p_value < 0.10:
        interpretation = "Marginally significant improvement (p < 0.10)"
    else:
        interpretation = "No significant improvement (p ≥ 0.10)"

    logging.info(f"McNemar's test contingency table:")
    logging.info(f"  Both correct: {both_correct}")
    logging.info(f"  Baseline only correct: {baseline_only}")
    logging.info(f"  Optimized only correct: {optimized_only}")
    logging.info(f"  Both wrong: {both_wrong}")

    return p_value, interpretation


# ========================================
# VALIDATION FUNCTIONS
# ========================================

def run_inference(df, prompt: str, description: str) -> Tuple[List[int], List[int]]:
    """
    Run inference on dataset with given prompt.

    Args:
        df: DataFrame with HOCON data
        prompt: System prompt to use
        description: Description for progress bar

    Returns:
        Tuple of (y_true, y_pred)
    """
    y_true: List[int] = []
    y_pred: List[int] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=description):
        text: str = row['text']
        true_label: int = row['label']

        result: Dict[str, Any] = classify_text(text, prompt)
        pred_label: int = result['label']

        y_true.append(true_label)
        y_pred.append(pred_label)

    return y_true, y_pred


def print_comparison_table(
    baseline_metrics: Dict[str, float],
    optimized_metrics: Dict[str, float],
    baseline_cis: Dict[str, Tuple[float, float]],
    optimized_cis: Dict[str, Tuple[float, float]]
) -> None:
    """
    Print a formatted comparison table of baseline vs optimized performance.

    Args:
        baseline_metrics: Metrics dictionary for baseline
        optimized_metrics: Metrics dictionary for optimized
        baseline_cis: Confidence intervals for baseline
        optimized_cis: Confidence intervals for optimized
    """
    print("\n" + "=" * 90)
    print("CROSS-DOMAIN VALIDATION RESULTS (HOCON34k Dataset)")
    print("=" * 90)
    print()
    print(f"{'Metric':<20} {'Baseline':>15} {'Optimized':>15} {'Delta':>12} {'Δ %':>10}")
    print("-" * 90)

    # F1 Score
    f1_delta = optimized_metrics['F1'] - baseline_metrics['F1']
    f1_delta_pct = (f1_delta / baseline_metrics['F1'] * 100) if baseline_metrics['F1'] != 0 else 0
    print(f"{'F1 Score':<20} {baseline_metrics['F1']:>15.4f} {optimized_metrics['F1']:>15.4f} "
          f"{f1_delta:>+12.4f} {f1_delta_pct:>+9.1f}%")

    # F2 Score
    f2_delta = optimized_metrics['F2'] - baseline_metrics['F2']
    f2_delta_pct = (f2_delta / baseline_metrics['F2'] * 100) if baseline_metrics['F2'] != 0 else 0
    print(f"{'F2 Score':<20} {baseline_metrics['F2']:>15.4f} {optimized_metrics['F2']:>15.4f} "
          f"{f2_delta:>+12.4f} {f2_delta_pct:>+9.1f}%")
    print(f"{'  95% CI':<20} [{baseline_cis['F2'][0]:>6.4f}, {baseline_cis['F2'][1]:>6.4f}] "
          f"[{optimized_cis['F2'][0]:>6.4f}, {optimized_cis['F2'][1]:>6.4f}]")

    # MCC
    mcc_delta = optimized_metrics['MCC'] - baseline_metrics['MCC']
    print(f"{'MCC':<20} {baseline_metrics['MCC']:>15.4f} {optimized_metrics['MCC']:>15.4f} "
          f"{mcc_delta:>+12.4f}")
    print(f"{'  95% CI':<20} [{baseline_cis['MCC'][0]:>6.4f}, {baseline_cis['MCC'][1]:>6.4f}] "
          f"[{optimized_cis['MCC'][0]:>6.4f}, {optimized_cis['MCC'][1]:>6.4f}]")

    # S-Score
    s_delta = optimized_metrics['S_SCORE'] - baseline_metrics['S_SCORE']
    s_delta_pct = (s_delta / baseline_metrics['S_SCORE'] * 100) if baseline_metrics['S_SCORE'] != 0 else 0
    print(f"{'S-Score (F2+MCC)':<20} {baseline_metrics['S_SCORE']:>15.4f} {optimized_metrics['S_SCORE']:>15.4f} "
          f"{s_delta:>+12.4f} {s_delta_pct:>+9.1f}%")
    print(f"{'  95% CI':<20} [{baseline_cis['S_SCORE'][0]:>6.4f}, {baseline_cis['S_SCORE'][1]:>6.4f}] "
          f"[{optimized_cis['S_SCORE'][0]:>6.4f}, {optimized_cis['S_SCORE'][1]:>6.4f}]")

    # Accuracy
    acc_delta = optimized_metrics['ACC'] - baseline_metrics['ACC']
    acc_delta_pct = (acc_delta / baseline_metrics['ACC'] * 100) if baseline_metrics['ACC'] != 0 else 0
    print(f"{'Accuracy':<20} {baseline_metrics['ACC']:>15.4f} {optimized_metrics['ACC']:>15.4f} "
          f"{acc_delta:>+12.4f} {acc_delta_pct:>+9.1f}%")

    print("=" * 90)


# ========================================
# MAIN EXECUTION
# ========================================

def main() -> None:
    """
    Execute cross-domain validation on HOCON34k dataset.
    """
    logging.info("=" * 70)
    logging.info("CROSS-DOMAIN VALIDATION: gutefrage.net → HOCON34k")
    logging.info("Testing prompt generalization across different hate speech domains")
    logging.info("=" * 70)

    # Validation check: Ensure prompts are configured
    if "PASTE YOUR" in OPTIMIZED_PROMPT:
        logging.error("ERROR: You must paste your optimized prompt into OPTIMIZED_PROMPT variable!")
        logging.error("Open validate_on_hocon.py and replace the placeholder with your best prompt.")
        sys.exit(1)

    # Step 1: Load HOCON dataset
    logging.info("\n" + "=" * 70)
    logging.info("STEP 1: Loading HOCON34k Dataset")
    logging.info("=" * 70)
    logging.info(f"Data file: {HOCON_DATA_FILE}")
    logging.info(f"Sample size: {SAMPLE_SIZE}")

    try:
        hocon_df = load_and_prepare_data(
            HOCON_DATA_FILE,
            dataset_type='hocon',
            sample_size=SAMPLE_SIZE
        )
        logging.info(f"✓ Loaded {len(hocon_df)} samples from HOCON34k")
        logging.info(f"  Class distribution: {dict(hocon_df['label'].value_counts().sort_index())}")
    except FileNotFoundError:
        logging.error(f"ERROR: HOCON data file not found: {HOCON_DATA_FILE}")
        logging.error("Please ensure the HOCON34k dataset is in data/raw/")
        sys.exit(1)
    except Exception as e:
        logging.error(f"ERROR loading HOCON data: {e}")
        sys.exit(1)

    # Step 2: Baseline Inference
    logging.info("\n" + "=" * 70)
    logging.info("STEP 2: Evaluating BASELINE Prompt on HOCON34k")
    logging.info("=" * 70)
    logging.info(f"Prompt preview: {BASELINE_PROMPT[:100]}...")

    y_true_baseline, y_pred_baseline = run_inference(
        hocon_df,
        BASELINE_PROMPT,
        "Baseline inference"
    )

    baseline_metrics = calculate_metrics(y_true_baseline, y_pred_baseline)
    logging.info("Baseline Results:")
    logging.info(f"  F1-Score:  {baseline_metrics['F1']:.4f}")
    logging.info(f"  F2-Score:  {baseline_metrics['F2']:.4f}")
    logging.info(f"  MCC:       {baseline_metrics['MCC']:.4f}")
    logging.info(f"  S-Score:   {baseline_metrics['S_SCORE']:.4f}")
    logging.info(f"  Accuracy:  {baseline_metrics['ACC']:.4f}")

    # Step 3: Optimized Inference
    logging.info("\n" + "=" * 70)
    logging.info("STEP 3: Evaluating OPTIMIZED Prompt on HOCON34k")
    logging.info("=" * 70)
    logging.info(f"Prompt preview: {OPTIMIZED_PROMPT[:100]}...")

    y_true_optimized, y_pred_optimized = run_inference(
        hocon_df,
        OPTIMIZED_PROMPT,
        "Optimized inference"
    )

    optimized_metrics = calculate_metrics(y_true_optimized, y_pred_optimized)
    logging.info("Optimized Results:")
    logging.info(f"  F1-Score:  {optimized_metrics['F1']:.4f}")
    logging.info(f"  F2-Score:  {optimized_metrics['F2']:.4f}")
    logging.info(f"  MCC:       {optimized_metrics['MCC']:.4f}")
    logging.info(f"  S-Score:   {optimized_metrics['S_SCORE']:.4f}")
    logging.info(f"  Accuracy:  {optimized_metrics['ACC']:.4f}")

    # Step 4: Statistical Significance Testing
    logging.info("\n" + "=" * 70)
    logging.info("STEP 4: Statistical Significance Testing")
    logging.info("=" * 70)

    # Bootstrap confidence intervals
    logging.info("Calculating 95% confidence intervals (bootstrap, n=1000)...")

    # Baseline CIs
    _, baseline_f2_lower, baseline_f2_upper = bootstrap_confidence_interval(
        y_true_baseline, y_pred_baseline, 'F2', n_bootstrap=1000
    )
    _, baseline_mcc_lower, baseline_mcc_upper = bootstrap_confidence_interval(
        y_true_baseline, y_pred_baseline, 'MCC', n_bootstrap=1000
    )
    _, baseline_s_lower, baseline_s_upper = bootstrap_confidence_interval(
        y_true_baseline, y_pred_baseline, 'S_SCORE', n_bootstrap=1000
    )

    baseline_cis = {
        'F2': (baseline_f2_lower, baseline_f2_upper),
        'MCC': (baseline_mcc_lower, baseline_mcc_upper),
        'S_SCORE': (baseline_s_lower, baseline_s_upper)
    }

    # Optimized CIs
    _, optimized_f2_lower, optimized_f2_upper = bootstrap_confidence_interval(
        y_true_optimized, y_pred_optimized, 'F2', n_bootstrap=1000
    )
    _, optimized_mcc_lower, optimized_mcc_upper = bootstrap_confidence_interval(
        y_true_optimized, y_pred_optimized, 'MCC', n_bootstrap=1000
    )
    _, optimized_s_lower, optimized_s_upper = bootstrap_confidence_interval(
        y_true_optimized, y_pred_optimized, 'S_SCORE', n_bootstrap=1000
    )

    optimized_cis = {
        'F2': (optimized_f2_lower, optimized_f2_upper),
        'MCC': (optimized_mcc_lower, optimized_mcc_upper),
        'S_SCORE': (optimized_s_lower, optimized_s_upper)
    }

    logging.info("✓ Bootstrap confidence intervals calculated")

    # McNemar's test
    logging.info("\nRunning McNemar's test (paired predictions)...")
    mcnemar_p, mcnemar_interp = mcnemar_test(
        y_true_baseline,
        y_pred_baseline,
        y_pred_optimized
    )
    logging.info(f"McNemar's test p-value: {mcnemar_p:.6f}")
    logging.info(f"Interpretation: {mcnemar_interp}")

    # Step 5: Print Comparison Table
    logging.info("\n" + "=" * 70)
    logging.info("STEP 5: Results Summary")
    logging.info("=" * 70)

    print_comparison_table(baseline_metrics, optimized_metrics, baseline_cis, optimized_cis)

    # Step 6: Save Results
    logging.info("\n" + "=" * 70)
    logging.info("STEP 6: Saving Results")
    logging.info("=" * 70)

    # Calculate improvements
    f2_improvement = optimized_metrics['F2'] - baseline_metrics['F2']
    f2_improvement_pct = (f2_improvement / baseline_metrics['F2'] * 100) if baseline_metrics['F2'] != 0 else 0
    mcc_improvement = optimized_metrics['MCC'] - baseline_metrics['MCC']
    s_score_improvement = optimized_metrics['S_SCORE'] - baseline_metrics['S_SCORE']
    s_score_improvement_pct = (s_score_improvement / baseline_metrics['S_SCORE'] * 100) if baseline_metrics['S_SCORE'] != 0 else 0

    results = {
        "validation_type": "cross_domain",
        "training_domain": "gutefrage.net",
        "test_domain": "HOCON34k",
        "test_sample_size": int(len(hocon_df)),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "baseline_metrics": {
            "F1": float(baseline_metrics['F1']),
            "F2": float(baseline_metrics['F2']),
            "F2_CI_95": [float(baseline_f2_lower), float(baseline_f2_upper)],
            "MCC": float(baseline_metrics['MCC']),
            "MCC_CI_95": [float(baseline_mcc_lower), float(baseline_mcc_upper)],
            "S_SCORE": float(baseline_metrics['S_SCORE']),
            "S_SCORE_CI_95": [float(baseline_s_lower), float(baseline_s_upper)],
            "ACC": float(baseline_metrics['ACC'])
        },

        "optimized_metrics": {
            "F1": float(optimized_metrics['F1']),
            "F2": float(optimized_metrics['F2']),
            "F2_CI_95": [float(optimized_f2_lower), float(optimized_f2_upper)],
            "MCC": float(optimized_metrics['MCC']),
            "MCC_CI_95": [float(optimized_mcc_lower), float(optimized_mcc_upper)],
            "S_SCORE": float(optimized_metrics['S_SCORE']),
            "S_SCORE_CI_95": [float(optimized_s_lower), float(optimized_s_upper)],
            "ACC": float(optimized_metrics['ACC'])
        },

        "improvement": {
            "F2_absolute": float(f2_improvement),
            "F2_relative_percent": float(f2_improvement_pct),
            "MCC_absolute": float(mcc_improvement),
            "S_SCORE_absolute": float(s_score_improvement),
            "S_SCORE_relative_percent": float(s_score_improvement_pct)
        },

        "statistical_significance": {
            "mcnemar_p_value": float(mcnemar_p),
            "mcnemar_interpretation": mcnemar_interp,
            "is_significant_at_05": bool(mcnemar_p < 0.05)
        },

        "prompts": {
            "baseline_prompt": BASELINE_PROMPT,
            "optimized_prompt": OPTIMIZED_PROMPT
        },

        "interpretation": {
            "generalization_quality": "Strong" if s_score_improvement > 0.1 else "Moderate" if s_score_improvement > 0 else "Poor",
            "note": "This cross-domain validation tests whether optimization on gutefrage.net generalizes to HOCON34k, a completely different hate speech dataset. Positive improvement suggests domain-robust prompt optimization."
        }
    }

    # Ensure results directory exists
    RESULTS_DIR.mkdir(exist_ok=True)

    # Save to JSON
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"✓ Results saved to: {RESULTS_FILE}")

    # Final Summary
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"✅ Cross-domain evaluation on HOCON34k dataset ({len(hocon_df)} samples)")
    print(f"✅ Baseline S-Score:   {baseline_metrics['S_SCORE']:.4f} [{baseline_s_lower:.4f}, {baseline_s_upper:.4f}]")
    print(f"✅ Optimized S-Score:  {optimized_metrics['S_SCORE']:.4f} [{optimized_s_lower:.4f}, {optimized_s_upper:.4f}]")
    print(f"✅ Improvement:        {s_score_improvement:+.4f} ({s_score_improvement_pct:+.1f}%) - {mcnemar_interp}")
    print(f"✅ Results saved to:   {RESULTS_FILE}")
    print()
    print("THESIS INTERPRETATION:")
    if s_score_improvement > 0.1:
        print("  🎯 STRONG GENERALIZATION: Optimized prompt shows robust improvement")
        print("     across domains, suggesting domain-independent optimization gains.")
    elif s_score_improvement > 0:
        print("  ⚠️  MODERATE GENERALIZATION: Optimized prompt shows some improvement,")
        print("     but gains may be partially domain-specific.")
    else:
        print("  ❌ POOR GENERALIZATION: Optimized prompt does not generalize well to")
        print("     HOCON34k. Optimization may be overfitted to gutefrage.net domain.")
    print("=" * 70)


if __name__ == "__main__":
    main()
