import os
import click
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..database.operations import DatabaseManager
from ..database.models import Evaluation, EvaluationScore
from .confidence import ConfidenceAnalyzer

class HumanReviewInterface:
    """Interface for human review of evaluations"""
    
    def __init__(self, database_manager: DatabaseManager = None):
        self.db = database_manager or DatabaseManager()
        self.confidence_analyzer = ConfidenceAnalyzer()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize database
        self.db.init_database()
    
    def get_items_for_review(self, 
                           run_number: Optional[int] = None,
                           confidence_threshold: float = 0.7,
                           limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get evaluation items that need human review
        
        Args:
            run_number: Specific run to review (optional)
            confidence_threshold: Only show items below this confidence
            limit: Maximum number of items to return
            
        Returns:
            List of evaluation items with their scores and context
        """
        # Get evaluations that need review
        evaluations = self.db.get_evaluations_needing_review(limit=limit * 2)  # Get more to filter
        
        # If run_number specified, filter by it
        if run_number:
            evaluations = [e for e in evaluations if e.get('run_number') == run_number]
        
        # Get corresponding scores and add confidence analysis
        review_items = []
        for eval_data in evaluations[:limit]:
            eval_id = eval_data['id']
            
            # Get evaluation scores
            scores = self.db.query_evaluation_scores(evaluation_id=eval_id)
            
            # Format for review
            review_item = {
                'evaluation': eval_data,
                'scores': [score.to_dict() for score in scores],
                'review_context': self._create_review_context(eval_data, scores)
            }
            
            review_items.append(review_item)
        
        return review_items
    
    def _create_review_context(self, 
                              evaluation: Dict[str, Any], 
                              scores: List[EvaluationScore]) -> Dict[str, Any]:
        """Create context information for human review"""
        context = {
            'file_name': evaluation.get('file_name'),
            'prompt': evaluation.get('prompt'),
            'prompt_description': evaluation.get('prompt_description'),
            'provider': evaluation.get('provider'),
            'model': evaluation.get('model'),
            'output_preview': evaluation.get('output', '')[:200] + "..." if len(evaluation.get('output', '')) > 200 else evaluation.get('output', ''),
            'response_time': evaluation.get('response_time'),
            'token_usage': {
                'input': evaluation.get('input_tokens'),
                'output': evaluation.get('output_tokens')
            }
        }
        
        # Add scoring context
        if scores:
            context['automated_scores'] = []
            for score in scores:
                score_dict = score.to_dict()
                context['automated_scores'].append({
                    'method': score_dict.get('scoring_method'),
                    'score': score_dict.get('automated_score'),
                    'confidence': score_dict.get('confidence_score'),
                    'needs_review': score_dict.get('needs_review')
                })
        
        return context
    
    def conduct_interactive_review(self, 
                                  run_number: Optional[int] = None,
                                  confidence_threshold: float = 0.7,
                                  reviewer_name: str = None) -> Dict[str, Any]:
        """
        Conduct an interactive review session
        
        Args:
            run_number: Specific run to review
            confidence_threshold: Confidence threshold for filtering
            reviewer_name: Name of the reviewer
            
        Returns:
            Summary of the review session
        """
        reviewer_name = reviewer_name or os.getenv('USER', 'anonymous')
        
        # Get items for review
        review_items = self.get_items_for_review(
            run_number=run_number,
            confidence_threshold=confidence_threshold,
            limit=50  # Allow more items for interactive review
        )
        
        if not review_items:
            click.echo("No items need review!")
            return {'reviewed_count': 0, 'skipped_count': 0}
        
        click.echo(f"\nFound {len(review_items)} items for review")
        click.echo("Commands: [s]core, [p]ass, [f]ail, [sk]ip, [q]uit, [h]elp")
        
        reviewed_count = 0
        skipped_count = 0
        
        for i, item in enumerate(review_items):
            if not self._review_single_item(item, i + 1, len(review_items), reviewer_name):
                # User chose to quit
                break
            
            if item.get('reviewed'):
                reviewed_count += 1
            else:
                skipped_count += 1
        
        summary = {
            'reviewed_count': reviewed_count,
            'skipped_count': skipped_count,
            'total_items': len(review_items),
            'reviewer_name': reviewer_name
        }
        
        click.echo(f"\nReview session complete!")
        click.echo(f"Reviewed: {reviewed_count}, Skipped: {skipped_count}")
        
        return summary
    
    def _review_single_item(self, 
                           item: Dict[str, Any], 
                           current: int, 
                           total: int,
                           reviewer_name: str) -> bool:
        """
        Review a single item interactively
        
        Returns:
            True to continue, False to quit
        """
        evaluation = item['evaluation']
        context = item['review_context']
        
        click.echo(f"\n" + "="*80)
        click.echo(f"Review Item {current}/{total}")
        click.echo(f"="*80)
        
        # Display evaluation info
        click.echo(f"File: {context['file_name']}")
        click.echo(f"Provider/Model: {context['provider']}/{context['model']}")
        click.echo(f"Prompt: {context['prompt']}")
        if context['prompt_description']:
            click.echo(f"Description: {context['prompt_description']}")
        
        click.echo(f"\nOutput Preview:")
        click.echo(f"{context['output_preview']}")
        
        # Display automated scores
        if context.get('automated_scores'):
            click.echo(f"\nAutomated Scores:")
            for score in context['automated_scores']:
                click.echo(f"  {score['method']}: {score['score']:.2f} (confidence: {score['confidence']:.2f})")
        
        # Get user input
        while True:
            try:
                command = click.prompt(
                    "\nAction",
                    type=click.Choice(['s', 'score', 'p', 'pass', 'f', 'fail', 'sk', 'skip', 'q', 'quit', 'h', 'help']),
                    show_choices=False
                ).lower()
                
                if command in ['q', 'quit']:
                    return False
                
                elif command in ['h', 'help']:
                    self._show_help()
                    continue
                
                elif command in ['sk', 'skip']:
                    item['reviewed'] = False
                    return True
                
                elif command in ['p', 'pass']:
                    self._record_human_score(evaluation['id'], 5.0, "Pass", reviewer_name)
                    item['reviewed'] = True
                    return True
                
                elif command in ['f', 'fail']:
                    self._record_human_score(evaluation['id'], 1.0, "Fail", reviewer_name)
                    item['reviewed'] = True
                    return True
                
                elif command in ['s', 'score']:
                    score = click.prompt("Score (1-5)", type=float)
                    if 1 <= score <= 5:
                        notes = click.prompt("Notes (optional)", default="", show_default=False)
                        self._record_human_score(evaluation['id'], score, notes, reviewer_name)
                        item['reviewed'] = True
                        return True
                    else:
                        click.echo("Score must be between 1 and 5")
                        continue
                
            except click.Abort:
                return False
            except Exception as e:
                click.echo(f"Error: {e}")
                continue
    
    def _show_help(self):
        """Show help for review commands"""
        click.echo("\nReview Commands:")
        click.echo("  s, score  - Give a detailed score (1-5) with notes")
        click.echo("  p, pass   - Mark as pass (score 5)")
        click.echo("  f, fail   - Mark as fail (score 1)")
        click.echo("  sk, skip  - Skip this item")
        click.echo("  q, quit   - Quit review session")
        click.echo("  h, help   - Show this help")
    
    def _record_human_score(self, 
                           evaluation_id: int,
                           score: float,
                           notes: str,
                           reviewer_name: str):
        """Record a human score for an evaluation"""
        try:
            # Get existing scores for this evaluation
            existing_scores = self.db.query_evaluation_scores(evaluation_id=evaluation_id)
            
            # Check if human score already exists
            human_score_exists = any(score.human_score is not None for score in existing_scores)
            
            if human_score_exists:
                # Update existing human score
                for existing_score in existing_scores:
                    if existing_score.human_score is not None:
                        updates = {
                            'human_score': score,
                            'reviewer_name': reviewer_name,
                            'reviewer_notes': notes,
                            'review_date': datetime.utcnow(),
                            'needs_review': False
                        }
                        self.db.update_evaluation_score(existing_score.id, updates)
                        break
            else:
                # Create new score record
                score_data = {
                    'evaluation_id': evaluation_id,
                    'human_score': score,
                    'reviewer_name': reviewer_name,
                    'reviewer_notes': notes,
                    'review_date': datetime.utcnow(),
                    'scoring_method': 'human_review',
                    'needs_review': False
                }
                self.db.insert_evaluation_score(score_data)
            
            self.logger.info(f"Recorded human score {score} for evaluation {evaluation_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to record human score: {e}")
            click.echo(f"Error recording score: {e}")
    
    def batch_review_interface(self, 
                              items: List[Dict[str, Any]],
                              reviewer_name: str = None) -> List[Dict[str, Any]]:
        """
        Simplified batch review interface for programmatic use
        
        Args:
            items: List of review items with 'evaluation_id', 'score', 'notes'
            reviewer_name: Name of the reviewer
            
        Returns:
            List of review results
        """
        reviewer_name = reviewer_name or os.getenv('USER', 'anonymous')
        results = []
        
        for item in items:
            try:
                evaluation_id = item['evaluation_id']
                score = item['score']
                notes = item.get('notes', '')
                
                self._record_human_score(evaluation_id, score, notes, reviewer_name)
                
                results.append({
                    'evaluation_id': evaluation_id,
                    'status': 'success',
                    'score': score,
                    'reviewer': reviewer_name
                })
                
            except Exception as e:
                results.append({
                    'evaluation_id': item.get('evaluation_id'),
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
    
    def get_review_statistics(self, 
                             run_number: Optional[int] = None,
                             reviewer_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about human reviews"""
        # Get all scores with human reviews
        scores = self.db.query_evaluation_scores()
        human_scores = [s for s in scores if s.human_score is not None]
        
        # Filter by run number if specified
        if run_number:
            evaluations = self.db.query_evaluations(run_number=run_number)
            eval_ids = {e.id for e in evaluations}
            human_scores = [s for s in human_scores if s.evaluation_id in eval_ids]
        
        # Filter by reviewer if specified
        if reviewer_name:
            human_scores = [s for s in human_scores if s.reviewer_name == reviewer_name]
        
        if not human_scores:
            return {'total_reviews': 0}
        
        # Calculate statistics
        scores_list = [s.human_score for s in human_scores]
        
        # Group by reviewer
        by_reviewer = {}
        for score in human_scores:
            reviewer = score.reviewer_name or 'anonymous'
            if reviewer not in by_reviewer:
                by_reviewer[reviewer] = []
            by_reviewer[reviewer].append(score.human_score)
        
        # Score distribution
        score_distribution = {i: scores_list.count(i) for i in range(1, 6)}
        
        return {
            'total_reviews': len(human_scores),
            'average_score': sum(scores_list) / len(scores_list),
            'score_distribution': score_distribution,
            'reviewers': list(by_reviewer.keys()),
            'reviews_by_reviewer': {k: len(v) for k, v in by_reviewer.items()},
            'average_by_reviewer': {k: sum(v)/len(v) for k, v in by_reviewer.items()},
            'latest_review': max(s.review_date for s in human_scores if s.review_date),
            'pass_rate': len([s for s in scores_list if s >= 4]) / len(scores_list) if scores_list else 0
        }
    
    def export_review_data(self, 
                          output_file: str,
                          run_number: Optional[int] = None,
                          format: str = 'json') -> int:
        """
        Export human review data
        
        Args:
            output_file: Output file path
            run_number: Optional run number filter
            format: 'json' or 'csv'
            
        Returns:
            Number of records exported
        """
        # Get all human scores
        scores = self.db.query_evaluation_scores()
        human_scores = [s for s in scores if s.human_score is not None]
        
        # Filter by run number if specified
        if run_number:
            evaluations = self.db.query_evaluations(run_number=run_number)
            eval_ids = {e.id for e in evaluations}
            human_scores = [s for s in human_scores if s.evaluation_id in eval_ids]
        
        # Get evaluation details
        export_data = []
        for score in human_scores:
            evaluation = self.db.get_evaluation_by_id(score.evaluation_id)
            
            export_data.append({
                'evaluation_id': score.evaluation_id,
                'human_score': score.human_score,
                'reviewer_name': score.reviewer_name,
                'reviewer_notes': score.reviewer_notes,
                'review_date': score.review_date.isoformat() if score.review_date else None,
                'automated_score': score.automated_score,
                'confidence_score': score.confidence_score,
                'scoring_method': score.scoring_method
            })
        
        # Export data
        if format.lower() == 'json':
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        elif format.lower() == 'csv':
            import csv
            if export_data:
                with open(output_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=export_data[0].keys())
                    writer.writeheader()
                    writer.writerows(export_data)
        
        self.logger.info(f"Exported {len(export_data)} review records to {output_file}")
        return len(export_data)