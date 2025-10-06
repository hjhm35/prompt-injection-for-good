import logging
import statistics
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ConfidenceThresholds:
    """Configuration for confidence thresholds"""
    high_confidence: float = 0.8    # Above this = high confidence
    low_confidence: float = 0.3     # Below this = needs review
    score_variance_threshold: float = 1.0  # Max variance in scores for high confidence
    agreement_threshold: float = 0.7       # Min agreement between methods for confidence

class ConfidenceAnalyzer:
    """Analyze and calculate confidence scores for evaluations"""
    
    def __init__(self, thresholds: ConfidenceThresholds = None):
        self.thresholds = thresholds or ConfidenceThresholds()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_confidence(self, 
                           evaluation_scores: Dict[str, Any],
                           additional_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Calculate confidence score based on multiple factors
        
        Args:
            evaluation_scores: Dict containing various evaluation scores
            additional_context: Additional context for confidence calculation
            
        Returns:
            Dict with confidence analysis results
        """
        confidence_factors = []
        confidence_details = {}
        
        # Factor 1: LLM Judge Confidence (if available)
        if 'llm_judge' in evaluation_scores and 'confidence' in evaluation_scores['llm_judge']:
            judge_confidence = evaluation_scores['llm_judge']['confidence']
            confidence_factors.append(judge_confidence)
            confidence_details['judge_confidence'] = judge_confidence
        
        # Factor 2: Semantic Similarity Score Confidence
        if 'semantic_similarity' in evaluation_scores:
            semantic_score = evaluation_scores['semantic_similarity'].get('semantic_similarity_score', 0)
            # Higher confidence for scores closer to extremes (0 or 1)
            semantic_confidence = abs(semantic_score - 0.5) * 2
            confidence_factors.append(semantic_confidence)
            confidence_details['semantic_confidence'] = semantic_confidence
        
        # Factor 3: Score Agreement Analysis
        agreement_confidence = self._calculate_agreement_confidence(evaluation_scores)
        if agreement_confidence is not None:
            confidence_factors.append(agreement_confidence)
            confidence_details['agreement_confidence'] = agreement_confidence
        
        # Factor 4: Score Variance Analysis
        variance_confidence = self._calculate_variance_confidence(evaluation_scores)
        if variance_confidence is not None:
            confidence_factors.append(variance_confidence)
            confidence_details['variance_confidence'] = variance_confidence
        
        # Factor 5: Response Quality Indicators
        quality_confidence = self._calculate_quality_confidence(evaluation_scores, additional_context)
        if quality_confidence is not None:
            confidence_factors.append(quality_confidence)
            confidence_details['quality_confidence'] = quality_confidence
        
        # Calculate overall confidence
        if confidence_factors:
            overall_confidence = statistics.mean(confidence_factors)
        else:
            overall_confidence = 0.5  # Default neutral confidence
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(overall_confidence)
        
        # Determine if review is needed
        needs_review = self._needs_human_review(overall_confidence, evaluation_scores)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_confidence, confidence_level, evaluation_scores
        )
        
        return {
            'overall_confidence': overall_confidence,
            'confidence_level': confidence_level,
            'needs_review': needs_review,
            'confidence_factors': confidence_details,
            'factor_count': len(confidence_factors),
            'recommendations': recommendations,
            'thresholds_used': {
                'high_confidence': self.thresholds.high_confidence,
                'low_confidence': self.thresholds.low_confidence
            }
        }
    
    def _calculate_agreement_confidence(self, evaluation_scores: Dict[str, Any]) -> Optional[float]:
        """Calculate confidence based on agreement between different evaluation methods"""
        scores = []
        
        # Collect normalized scores from different methods
        if 'llm_judge' in evaluation_scores and 'overall_score' in evaluation_scores['llm_judge']:
            # Normalize LLM judge score (1-5) to (0-1)
            judge_score = (evaluation_scores['llm_judge']['overall_score'] - 1) / 4
            scores.append(judge_score)
        
        if 'semantic_similarity' in evaluation_scores:
            semantic_score = evaluation_scores['semantic_similarity'].get('semantic_similarity_score', 0)
            scores.append(semantic_score)
        
        # Need at least 2 scores for agreement analysis
        if len(scores) < 2:
            return None
        
        # Calculate standard deviation as disagreement measure
        score_std = statistics.stdev(scores)
        
        # Convert disagreement to agreement confidence
        # Lower std dev = higher agreement = higher confidence
        max_possible_std = 0.5  # Maximum std dev for normalized scores
        agreement_confidence = max(0, 1 - (score_std / max_possible_std))
        
        return agreement_confidence
    
    def _calculate_variance_confidence(self, evaluation_scores: Dict[str, Any]) -> Optional[float]:
        """Calculate confidence based on variance in criteria scores"""
        criteria_scores = []
        
        # Get criteria scores from LLM judge if available
        if ('llm_judge' in evaluation_scores and 
            'criteria_scores' in evaluation_scores['llm_judge']):
            
            scores_dict = evaluation_scores['llm_judge']['criteria_scores']
            criteria_scores = list(scores_dict.values())
        
        if len(criteria_scores) < 2:
            return None
        
        # Calculate variance in criteria scores
        score_variance = statistics.variance(criteria_scores)
        
        # Lower variance = higher confidence
        # Normalize based on threshold
        variance_confidence = max(0, 1 - (score_variance / self.thresholds.score_variance_threshold))
        
        return variance_confidence
    
    def _calculate_quality_confidence(self, 
                                    evaluation_scores: Dict[str, Any],
                                    additional_context: Dict[str, Any] = None) -> Optional[float]:
        """Calculate confidence based on response quality indicators"""
        quality_factors = []
        
        # Factor: Length similarity (if context provided)
        if additional_context:
            eval_length = additional_context.get('evaluation_output_length', 0)
            ref_length = additional_context.get('reference_output_length', 0)
            
            if eval_length > 0 and ref_length > 0:
                length_ratio = min(eval_length, ref_length) / max(eval_length, ref_length)
                quality_factors.append(length_ratio)
        
        # Factor: Evaluation method availability
        method_count = len([k for k in evaluation_scores.keys() 
                           if k in ['llm_judge', 'semantic_similarity']])
        method_confidence = min(1.0, method_count / 2)  # Max confidence with 2+ methods
        quality_factors.append(method_confidence)
        
        # Factor: Error indicators
        has_errors = any('error' in str(evaluation_scores.get(method, {})) 
                        for method in evaluation_scores.keys())
        error_confidence = 0.2 if has_errors else 1.0
        quality_factors.append(error_confidence)
        
        if quality_factors:
            return statistics.mean(quality_factors)
        
        return None
    
    def _determine_confidence_level(self, confidence_score: float) -> str:
        """Determine confidence level based on score"""
        if confidence_score >= self.thresholds.high_confidence:
            return "high"
        elif confidence_score <= self.thresholds.low_confidence:
            return "low"
        else:
            return "medium"
    
    def _needs_human_review(self, confidence_score: float, evaluation_scores: Dict[str, Any]) -> bool:
        """Determine if human review is needed"""
        # Low confidence always needs review
        if confidence_score <= self.thresholds.low_confidence:
            return True
        
        # Check for disagreement between methods
        if self._has_method_disagreement(evaluation_scores):
            return True
        
        # Check for edge case scores (very close to pass/fail boundaries)
        if self._has_boundary_scores(evaluation_scores):
            return True
        
        # Check for evaluation errors
        if self._has_evaluation_errors(evaluation_scores):
            return True
        
        return False
    
    def _has_method_disagreement(self, evaluation_scores: Dict[str, Any]) -> bool:
        """Check if different evaluation methods disagree significantly"""
        recommendations = []
        
        # Collect recommendations from different methods
        if 'llm_judge' in evaluation_scores:
            judge_rec = evaluation_scores['llm_judge'].get('recommendation', '').lower()
            if judge_rec:
                recommendations.append(judge_rec)
        
        if 'semantic_similarity' in evaluation_scores:
            semantic_rec = evaluation_scores['semantic_similarity'].get('recommendation', '').lower()
            if semantic_rec:
                recommendations.append(semantic_rec)
        
        # Check for disagreement
        if len(set(recommendations)) > 1:
            return True
        
        return False
    
    def _has_boundary_scores(self, evaluation_scores: Dict[str, Any]) -> bool:
        """Check if scores are close to decision boundaries"""
        boundary_threshold = 0.1  # Within 0.1 of a boundary
        
        # Check LLM judge scores
        if 'llm_judge' in evaluation_scores and 'overall_score' in evaluation_scores['llm_judge']:
            score = evaluation_scores['llm_judge']['overall_score']
            # Check if close to pass/fail boundaries (assuming 1-5 scale)
            if abs(score - 2.5) <= boundary_threshold or abs(score - 3.5) <= boundary_threshold:
                return True
        
        # Check semantic similarity scores
        if 'semantic_similarity' in evaluation_scores:
            score = evaluation_scores['semantic_similarity'].get('semantic_similarity_score', 0)
            threshold = evaluation_scores['semantic_similarity'].get('threshold_used', 0.8)
            # Check if close to threshold
            if abs(score - threshold) <= boundary_threshold:
                return True
        
        return False
    
    def _has_evaluation_errors(self, evaluation_scores: Dict[str, Any]) -> bool:
        """Check if any evaluation method reported errors"""
        for method_scores in evaluation_scores.values():
            if isinstance(method_scores, dict) and 'error' in method_scores:
                return True
        
        return False
    
    def _generate_recommendations(self, 
                                confidence_score: float,
                                confidence_level: str,
                                evaluation_scores: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on confidence analysis"""
        recommendations = []
        
        if confidence_level == "low":
            recommendations.append("Human review required due to low confidence")
            
            if self._has_method_disagreement(evaluation_scores):
                recommendations.append("Methods disagree - investigate evaluation criteria")
            
            if self._has_evaluation_errors(evaluation_scores):
                recommendations.append("Evaluation errors detected - check system configuration")
        
        elif confidence_level == "medium":
            recommendations.append("Consider human review for quality assurance")
            
            if self._has_boundary_scores(evaluation_scores):
                recommendations.append("Score near decision boundary - manual verification recommended")
        
        else:  # high confidence
            recommendations.append("High confidence - automated decision acceptable")
        
        # Method-specific recommendations
        if len(evaluation_scores) == 1:
            recommendations.append("Consider using multiple evaluation methods for better confidence")
        
        return recommendations
    
    def batch_analyze_confidence(self, 
                                evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze confidence for multiple evaluations in batch"""
        results = []
        
        for i, evaluation in enumerate(evaluations):
            try:
                confidence_analysis = self.calculate_confidence(evaluation)
                confidence_analysis['batch_index'] = i
                results.append(confidence_analysis)
            except Exception as e:
                self.logger.error(f"Confidence analysis failed for evaluation {i}: {e}")
                results.append({
                    'overall_confidence': 0.0,
                    'confidence_level': 'low',
                    'needs_review': True,
                    'batch_index': i,
                    'error': str(e)
                })
        
        return results
    
    def get_review_priority_order(self, 
                                 confidence_analyses: List[Dict[str, Any]]) -> List[int]:
        """
        Get indices of evaluations ordered by review priority
        
        Returns:
            List of indices ordered by priority (highest priority first)
        """
        # Create list of (index, priority_score) tuples
        priority_items = []
        
        for i, analysis in enumerate(confidence_analyses):
            confidence = analysis.get('overall_confidence', 0.5)
            needs_review = analysis.get('needs_review', True)
            
            # Calculate priority score (lower = higher priority)
            priority_score = confidence
            
            # Boost priority for items that need review
            if needs_review:
                priority_score -= 0.5
            
            priority_items.append((i, priority_score))
        
        # Sort by priority score (lowest first = highest priority)
        priority_items.sort(key=lambda x: x[1])
        
        return [item[0] for item in priority_items]

def create_confidence_analyzer(config: Dict[str, Any] = None) -> ConfidenceAnalyzer:
    """
    Factory function to create a confidence analyzer with custom configuration
    
    Args:
        config: Configuration dict with threshold values
        
    Returns:
        ConfidenceAnalyzer instance
    """
    if config:
        thresholds = ConfidenceThresholds(
            high_confidence=config.get('high_confidence', 0.8),
            low_confidence=config.get('low_confidence', 0.3),
            score_variance_threshold=config.get('score_variance_threshold', 1.0),
            agreement_threshold=config.get('agreement_threshold', 0.7)
        )
        return ConfidenceAnalyzer(thresholds)
    else:
        return ConfidenceAnalyzer()