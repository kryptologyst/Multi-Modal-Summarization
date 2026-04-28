"""Evaluation metrics for multi-modal summarization."""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
import logging

# Import evaluation libraries
try:
    from rouge_score import rouge_scorer
    from sacrebleu import BLEU
    from bert_score import score as bert_score
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError as e:
    logging.warning(f"Some evaluation libraries not available: {e}")


class MultiModalEvaluator:
    """Evaluator for multi-modal summarization tasks."""
    
    def __init__(self):
        """Initialize the evaluator."""
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            logging.warning(f"Could not download NLTK data: {e}")
        
        # Initialize scorers
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleu_scorer = BLEU()
        self.smoothing = SmoothingFunction().method1
        
        logging.info("Initialized multi-modal evaluator")
    
    def compute_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute ROUGE scores.
        
        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            
        Returns:
            Dictionary containing ROUGE scores
        """
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            for metric in rouge_scores:
                rouge_scores[metric].append(scores[metric].fmeasure)
        
        # Compute averages
        avg_scores = {}
        for metric in rouge_scores:
            avg_scores[f"rouge_{metric}"] = np.mean(rouge_scores[metric])
        
        return avg_scores
    
    def compute_bleu_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute BLEU scores.
        
        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            
        Returns:
            Dictionary containing BLEU scores
        """
        # Tokenize predictions and references
        pred_tokens = [pred.split() for pred in predictions]
        ref_tokens = [[ref.split()] for ref in references]
        
        # Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4
        bleu_scores = {}
        for n in range(1, 5):
            scores = []
            for pred, ref in zip(pred_tokens, ref_tokens):
                try:
                    score = sentence_bleu(ref, pred, weights=tuple([1/n] * n), smoothing_function=self.smoothing)
                    scores.append(score)
                except:
                    scores.append(0.0)
            bleu_scores[f"bleu_{n}"] = np.mean(scores)
        
        return bleu_scores
    
    def compute_bert_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute BERTScore.
        
        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            
        Returns:
            Dictionary containing BERTScore metrics
        """
        try:
            P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
            return {
                "bert_score_precision": P.mean().item(),
                "bert_score_recall": R.mean().item(),
                "bert_score_f1": F1.mean().item()
            }
        except Exception as e:
            logging.warning(f"Could not compute BERTScore: {e}")
            return {
                "bert_score_precision": 0.0,
                "bert_score_recall": 0.0,
                "bert_score_f1": 0.0
            }
    
    def compute_meteor_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute METEOR score.
        
        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            
        Returns:
            Dictionary containing METEOR score
        """
        try:
            from nltk.translate.meteor_score import meteor_score
            scores = []
            for pred, ref in zip(predictions, references):
                try:
                    score = meteor_score([ref.split()], pred.split())
                    scores.append(score)
                except:
                    scores.append(0.0)
            return {"meteor": np.mean(scores)}
        except ImportError:
            logging.warning("METEOR not available, skipping")
            return {"meteor": 0.0}
    
    def compute_cider_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute CIDEr score.
        
        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            
        Returns:
            Dictionary containing CIDEr score
        """
        try:
            from pycocoevalcap.cider.cider import Cider
            cider_scorer = Cider()
            
            # Format for CIDEr scorer
            pred_dict = {i: [pred] for i, pred in enumerate(predictions)}
            ref_dict = {i: [ref] for i, ref in enumerate(references)}
            
            score, _ = cider_scorer.compute_score(ref_dict, pred_dict)
            return {"cider": score}
        except ImportError:
            logging.warning("CIDEr not available, skipping")
            return {"cider": 0.0}
    
    def compute_all_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute all available metrics.
        
        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            
        Returns:
            Dictionary containing all computed metrics
        """
        all_metrics = {}
        
        # ROUGE scores
        rouge_scores = self.compute_rouge_scores(predictions, references)
        all_metrics.update(rouge_scores)
        
        # BLEU scores
        bleu_scores = self.compute_bleu_scores(predictions, references)
        all_metrics.update(bleu_scores)
        
        # BERTScore
        bert_scores = self.compute_bert_score(predictions, references)
        all_metrics.update(bert_scores)
        
        # METEOR
        meteor_scores = self.compute_meteor_score(predictions, references)
        all_metrics.update(meteor_scores)
        
        # CIDEr
        cider_scores = self.compute_cider_score(predictions, references)
        all_metrics.update(cider_scores)
        
        return all_metrics
    
    def create_leaderboard(self, results: Dict[str, Dict[str, float]]) -> str:
        """Create a formatted leaderboard from results.
        
        Args:
            results: Dictionary containing results for different models/configurations
            
        Returns:
            Formatted leaderboard string
        """
        if not results:
            return "No results to display."
        
        # Get all metric names
        all_metrics = set()
        for model_results in results.values():
            all_metrics.update(model_results.keys())
        
        all_metrics = sorted(list(all_metrics))
        
        # Create header
        header = "Model".ljust(20)
        for metric in all_metrics:
            header += metric.ljust(12)
        
        lines = [header, "=" * len(header)]
        
        # Add results for each model
        for model_name, model_results in results.items():
            line = model_name.ljust(20)
            for metric in all_metrics:
                value = model_results.get(metric, 0.0)
                line += f"{value:.4f}".ljust(12)
            lines.append(line)
        
        return "\n".join(lines)


class SummarizationMetrics:
    """Specialized metrics for summarization tasks."""
    
    @staticmethod
    def compression_ratio(predictions: List[str], references: List[str]) -> float:
        """Compute average compression ratio.
        
        Args:
            predictions: List of predicted summaries
            references: List of reference texts
            
        Returns:
            Average compression ratio
        """
        ratios = []
        for pred, ref in zip(predictions, references):
            pred_len = len(pred.split())
            ref_len = len(ref.split())
            if ref_len > 0:
                ratios.append(pred_len / ref_len)
        
        return np.mean(ratios) if ratios else 0.0
    
    @staticmethod
    def extractive_coverage(predictions: List[str], references: List[str]) -> float:
        """Compute how much of the summary is covered by the reference.
        
        Args:
            predictions: List of predicted summaries
            references: List of reference texts
            
        Returns:
            Average extractive coverage
        """
        coverages = []
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            
            if len(pred_words) > 0:
                coverage = len(pred_words.intersection(ref_words)) / len(pred_words)
                coverages.append(coverage)
        
        return np.mean(coverages) if coverages else 0.0
    
    @staticmethod
    def abstractiveness_score(predictions: List[str], references: List[str]) -> float:
        """Compute abstractiveness score (1 - extractive_coverage).
        
        Args:
            predictions: List of predicted summaries
            references: List of reference texts
            
        Returns:
            Average abstractiveness score
        """
        coverage = SummarizationMetrics.extractive_coverage(predictions, references)
        return 1.0 - coverage


def evaluate_model_predictions(
    model_predictions: List[str],
    ground_truth: List[str],
    reference_texts: Optional[List[str]] = None
) -> Dict[str, float]:
    """Evaluate model predictions comprehensively.
    
    Args:
        model_predictions: List of model-generated summaries
        ground_truth: List of reference summaries
        reference_texts: List of original texts (for compression ratio)
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    evaluator = MultiModalEvaluator()
    
    # Compute standard metrics
    metrics = evaluator.compute_all_metrics(model_predictions, ground_truth)
    
    # Compute summarization-specific metrics
    if reference_texts:
        metrics["compression_ratio"] = SummarizationMetrics.compression_ratio(
            model_predictions, reference_texts
        )
        metrics["extractive_coverage"] = SummarizationMetrics.extractive_coverage(
            model_predictions, reference_texts
        )
        metrics["abstractiveness_score"] = SummarizationMetrics.abstractiveness_score(
            model_predictions, reference_texts
        )
    
    return metrics
