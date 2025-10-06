import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod

class BaseSemanticScorer(ABC):
    """Abstract base class for semantic similarity scoring"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts"""
        pass
    
    @abstractmethod
    def batch_compute_similarity(self, texts1: List[str], texts2: List[str]) -> List[float]:
        """Compute similarities for multiple text pairs"""
        pass

class SentenceTransformerScorer(BaseSemanticScorer):
    """Semantic similarity scorer using sentence transformers"""
    
    DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
    
    def __init__(self, model_name: str = None):
        super().__init__()
        self.model_name = model_name or self.DEFAULT_MODEL
        self.model = None
        self._model_loaded = False
    
    def _load_model(self):
        """Load the sentence transformer model (lazy loading)"""
        if self._model_loaded:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            self.logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self._model_loaded = True
            self.logger.info("Model loaded successfully")
        except ImportError:
            raise ImportError(
                "sentence-transformers library not installed. "
                "Run: pip install sentence-transformers"
            )
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts using sentence embeddings
        
        Returns:
            Similarity score between 0 and 1
        """
        self._load_model()
        
        try:
            # Get embeddings
            embeddings = self.model.encode([text1, text2])
            
            # Compute cosine similarity
            similarity = self._cosine_similarity(embeddings[0], embeddings[1])
            
            # Convert from [-1, 1] to [0, 1] range
            normalized_similarity = (similarity + 1) / 2
            
            return float(normalized_similarity)
            
        except Exception as e:
            self.logger.error(f"Failed to compute similarity: {e}")
            return 0.0
    
    def batch_compute_similarity(self, texts1: List[str], texts2: List[str]) -> List[float]:
        """
        Compute similarities for multiple text pairs efficiently
        
        Args:
            texts1: First texts in each pair
            texts2: Second texts in each pair
            
        Returns:
            List of similarity scores
        """
        if len(texts1) != len(texts2):
            raise ValueError("texts1 and texts2 must have the same length")
        
        self._load_model()
        
        try:
            # Encode all texts at once for efficiency
            all_texts = texts1 + texts2
            embeddings = self.model.encode(all_texts)
            
            # Split embeddings back into two groups
            mid_point = len(texts1)
            embeddings1 = embeddings[:mid_point]
            embeddings2 = embeddings[mid_point:]
            
            # Compute similarities
            similarities = []
            for emb1, emb2 in zip(embeddings1, embeddings2):
                similarity = self._cosine_similarity(emb1, emb2)
                normalized_similarity = (similarity + 1) / 2
                similarities.append(float(normalized_similarity))
            
            return similarities
            
        except Exception as e:
            self.logger.error(f"Failed to compute batch similarities: {e}")
            return [0.0] * len(texts1)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts"""
        self._load_model()
        return self.model.encode(texts)
    
    def find_most_similar(self, query_text: str, candidate_texts: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find the most similar texts to a query
        
        Args:
            query_text: The query text
            candidate_texts: List of candidate texts
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples, sorted by similarity descending
        """
        similarities = []
        for candidate in candidate_texts:
            sim = self.compute_similarity(query_text, candidate)
            similarities.append(sim)
        
        # Get top-k indices sorted by similarity
        sorted_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(int(idx), similarities[idx]) for idx in sorted_indices]

class SimpleSemanticScorer(BaseSemanticScorer):
    """Simple semantic scorer using basic text similarity metrics"""
    
    def __init__(self):
        super().__init__()
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity using Jaccard similarity of words
        
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Tokenize and normalize
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            # Compute Jaccard similarity
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            if union == 0:
                return 1.0 if len(words1) == 0 and len(words2) == 0 else 0.0
            
            return intersection / union
            
        except Exception as e:
            self.logger.error(f"Failed to compute simple similarity: {e}")
            return 0.0
    
    def batch_compute_similarity(self, texts1: List[str], texts2: List[str]) -> List[float]:
        """Compute similarities for multiple text pairs"""
        if len(texts1) != len(texts2):
            raise ValueError("texts1 and texts2 must have the same length")
        
        return [self.compute_similarity(t1, t2) for t1, t2 in zip(texts1, texts2)]

class SemanticEvaluator:
    """High-level semantic evaluation system"""
    
    def __init__(self, scorer: BaseSemanticScorer = None, similarity_threshold: float = 0.8):
        self.scorer = scorer or self._get_default_scorer()
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _get_default_scorer(self) -> BaseSemanticScorer:
        """Get default scorer, fallback to simple if transformers not available"""
        try:
            return SentenceTransformerScorer()
        except ImportError:
            self.logger.warning("sentence-transformers not available, using simple scorer")
            return SimpleSemanticScorer()
    
    def evaluate_response(self, 
                         evaluation_output: str,
                         reference_output: str,
                         prompt: str = None) -> Dict[str, Any]:
        """
        Evaluate a response using semantic similarity
        
        Args:
            evaluation_output: The AI response to evaluate
            reference_output: The reference/gold standard response
            prompt: The original prompt (optional, for context)
            
        Returns:
            Dict with semantic evaluation results
        """
        try:
            # Compute similarity
            similarity_score = self.scorer.compute_similarity(evaluation_output, reference_output)
            
            # Determine pass/fail based on threshold
            passes_threshold = similarity_score >= self.similarity_threshold
            
            # Calculate confidence (higher for scores closer to extremes)
            confidence = abs(similarity_score - 0.5) * 2
            
            # Determine recommendation
            if similarity_score >= 0.8:
                recommendation = "Pass"
            elif similarity_score >= 0.6:
                recommendation = "Review" 
            else:
                recommendation = "Fail"
            
            # Analyze key differences (simple version)
            key_differences = self._analyze_differences(evaluation_output, reference_output)
            
            return {
                "semantic_similarity_score": similarity_score,
                "passes_threshold": passes_threshold,
                "threshold_used": self.similarity_threshold,
                "confidence": confidence,
                "recommendation": recommendation,
                "scorer_type": self.scorer.__class__.__name__,
                "key_differences": key_differences,
                "evaluation_method": "semantic_similarity"
            }
            
        except Exception as e:
            self.logger.error(f"Semantic evaluation failed: {e}")
            return {
                "semantic_similarity_score": 0.0,
                "passes_threshold": False,
                "threshold_used": self.similarity_threshold,
                "confidence": 0.0,
                "recommendation": "Review",
                "scorer_type": "error",
                "key_differences": f"Evaluation failed: {str(e)}",
                "evaluation_method": "semantic_similarity",
                "error": str(e)
            }
    
    def _analyze_differences(self, text1: str, text2: str) -> str:
        """Simple analysis of key differences between texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        only_in_1 = words1 - words2
        only_in_2 = words2 - words1
        
        differences = []
        
        if only_in_1:
            differences.append(f"AI response contains: {', '.join(list(only_in_1)[:5])}")
        
        if only_in_2:
            differences.append(f"Reference contains: {', '.join(list(only_in_2)[:5])}")
        
        if not differences:
            return "High word overlap, minimal differences"
        
        return "; ".join(differences)
    
    def batch_evaluate_responses(self, 
                                evaluation_outputs: List[str],
                                reference_outputs: List[str],
                                prompts: List[str] = None) -> List[Dict[str, Any]]:
        """
        Evaluate multiple responses in batch
        
        Args:
            evaluation_outputs: List of AI responses to evaluate
            reference_outputs: List of reference responses
            prompts: List of original prompts (optional)
            
        Returns:
            List of evaluation results
        """
        if len(evaluation_outputs) != len(reference_outputs):
            raise ValueError("evaluation_outputs and reference_outputs must have same length")
        
        if prompts and len(prompts) != len(evaluation_outputs):
            raise ValueError("prompts must have same length as outputs")
        
        # Compute all similarities in batch for efficiency
        try:
            similarity_scores = self.scorer.batch_compute_similarity(
                evaluation_outputs, reference_outputs
            )
        except Exception as e:
            self.logger.error(f"Batch similarity computation failed: {e}")
            similarity_scores = [0.0] * len(evaluation_outputs)
        
        # Process results
        results = []
        for i, (eval_output, ref_output, similarity) in enumerate(
            zip(evaluation_outputs, reference_outputs, similarity_scores)
        ):
            prompt = prompts[i] if prompts else None
            
            # Use the pre-computed similarity
            result = self.evaluate_response(eval_output, ref_output, prompt)
            result["semantic_similarity_score"] = similarity
            result["batch_index"] = i
            
            results.append(result)
        
        return results
    
    def set_threshold(self, threshold: float):
        """Update the similarity threshold"""
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.similarity_threshold = threshold
        self.logger.info(f"Updated similarity threshold to {threshold}")

def get_semantic_scorer(scorer_type: str = "auto", **kwargs) -> BaseSemanticScorer:
    """
    Factory function to get a semantic scorer
    
    Args:
        scorer_type: "auto", "transformer", or "simple"
        **kwargs: Additional arguments for scorer initialization
        
    Returns:
        Semantic scorer instance
    """
    if scorer_type == "auto":
        try:
            return SentenceTransformerScorer(**kwargs)
        except ImportError:
            return SimpleSemanticScorer(**kwargs)
    elif scorer_type == "transformer":
        return SentenceTransformerScorer(**kwargs)
    elif scorer_type == "simple":
        return SimpleSemanticScorer(**kwargs)
    else:
        raise ValueError(f"Unknown scorer type: {scorer_type}")