import logging
import statistics
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter

from ..database.operations import DatabaseManager

class PerformanceAnalyzer:
    """Analyze and compare LLM performance across evaluations"""
    
    def __init__(self, database_manager: DatabaseManager = None):
        self.db = database_manager or DatabaseManager()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def compare_runs(self, run_numbers: List[int]) -> Dict[str, Any]:
        """
        Compare performance across multiple evaluation runs
        
        Args:
            run_numbers: List of run numbers to compare
            
        Returns:
            Comparison analysis
        """
        comparison = {
            'run_numbers': run_numbers,
            'run_summaries': {},
            'provider_comparison': {},
            'model_comparison': {},
            'performance_trends': {},
            'recommendations': []
        }
        
        # Get data for each run
        all_evaluations = []
        for run_num in run_numbers:
            run_evals = self.db.query_evaluations(run_number=run_num)
            run_stats = self.db.get_run_statistics(run_num)
            
            comparison['run_summaries'][run_num] = {
                'statistics': run_stats,
                'evaluations': [e.to_dict() for e in run_evals]
            }
            
            all_evaluations.extend(run_evals)
        
        # Analyze by provider
        comparison['provider_comparison'] = self._analyze_by_provider(all_evaluations, run_numbers)
        
        # Analyze by model
        comparison['model_comparison'] = self._analyze_by_model(all_evaluations, run_numbers)
        
        # Analyze performance trends
        comparison['performance_trends'] = self._analyze_trends(comparison['run_summaries'])
        
        # Generate recommendations
        comparison['recommendations'] = self._generate_comparison_recommendations(comparison)
        
        return comparison
    
    def _analyze_by_provider(self, evaluations: List, run_numbers: List[int]) -> Dict[str, Any]:
        """Analyze performance by provider"""
        provider_stats = defaultdict(lambda: {
            'total_evaluations': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'average_response_time': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'response_times': [],
            'success_rates_by_run': {}
        })
        
        # Group evaluations by provider and run
        for eval_data in evaluations:
            provider = eval_data.llm_provider
            run_num = eval_data.evaluation_run_number
            
            provider_stats[provider]['total_evaluations'] += 1
            
            if eval_data.status == 'success':
                provider_stats[provider]['successful_evaluations'] += 1
                if eval_data.response_time_seconds:
                    provider_stats[provider]['response_times'].append(eval_data.response_time_seconds)
                if eval_data.input_token_usage:
                    provider_stats[provider]['total_input_tokens'] += eval_data.input_token_usage
                if eval_data.output_token_usage:
                    provider_stats[provider]['total_output_tokens'] += eval_data.output_token_usage
            else:
                provider_stats[provider]['failed_evaluations'] += 1
            
            # Track success rate by run
            if run_num not in provider_stats[provider]['success_rates_by_run']:
                provider_stats[provider]['success_rates_by_run'][run_num] = {'success': 0, 'total': 0}
            
            provider_stats[provider]['success_rates_by_run'][run_num]['total'] += 1
            if eval_data.status == 'success':
                provider_stats[provider]['success_rates_by_run'][run_num]['success'] += 1
        
        # Calculate derived metrics
        for provider, stats in provider_stats.items():
            total = stats['total_evaluations']
            if total > 0:
                stats['success_rate'] = stats['successful_evaluations'] / total
                
            if stats['response_times']:
                stats['average_response_time'] = statistics.mean(stats['response_times'])
                stats['median_response_time'] = statistics.median(stats['response_times'])
                stats['response_time_std'] = statistics.stdev(stats['response_times']) if len(stats['response_times']) > 1 else 0
            
            # Calculate success rates by run
            for run_num, run_data in stats['success_rates_by_run'].items():
                run_data['success_rate'] = run_data['success'] / run_data['total'] if run_data['total'] > 0 else 0
        
        return dict(provider_stats)
    
    def _analyze_by_model(self, evaluations: List, run_numbers: List[int]) -> Dict[str, Any]:
        """Analyze performance by model"""
        model_stats = defaultdict(lambda: {
            'provider': '',
            'total_evaluations': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'average_response_time': 0,
            'total_tokens': 0,
            'response_times': [],
            'success_rates_by_run': {}
        })
        
        for eval_data in evaluations:
            model_key = f"{eval_data.llm_provider}/{eval_data.model_version}"
            run_num = eval_data.evaluation_run_number
            
            model_stats[model_key]['provider'] = eval_data.llm_provider
            model_stats[model_key]['total_evaluations'] += 1
            
            if eval_data.status == 'success':
                model_stats[model_key]['successful_evaluations'] += 1
                if eval_data.response_time_seconds:
                    model_stats[model_key]['response_times'].append(eval_data.response_time_seconds)
                
                total_tokens = (eval_data.input_token_usage or 0) + (eval_data.output_token_usage or 0)
                model_stats[model_key]['total_tokens'] += total_tokens
            else:
                model_stats[model_key]['failed_evaluations'] += 1
            
            # Track by run
            if run_num not in model_stats[model_key]['success_rates_by_run']:
                model_stats[model_key]['success_rates_by_run'][run_num] = {'success': 0, 'total': 0}
            
            model_stats[model_key]['success_rates_by_run'][run_num]['total'] += 1
            if eval_data.status == 'success':
                model_stats[model_key]['success_rates_by_run'][run_num]['success'] += 1
        
        # Calculate derived metrics
        for model, stats in model_stats.items():
            total = stats['total_evaluations']
            if total > 0:
                stats['success_rate'] = stats['successful_evaluations'] / total
                
            if stats['response_times']:
                stats['average_response_time'] = statistics.mean(stats['response_times'])
                stats['median_response_time'] = statistics.median(stats['response_times'])
            
            for run_num, run_data in stats['success_rates_by_run'].items():
                run_data['success_rate'] = run_data['success'] / run_data['total'] if run_data['total'] > 0 else 0
        
        return dict(model_stats)
    
    def _analyze_trends(self, run_summaries: Dict[int, Dict]) -> Dict[str, Any]:
        """Analyze performance trends across runs"""
        trends = {
            'success_rate_trend': [],
            'response_time_trend': [],
            'token_usage_trend': [],
            'provider_dominance_trend': [],
            'improvement_metrics': {}
        }
        
        sorted_runs = sorted(run_summaries.keys())
        
        for run_num in sorted_runs:
            stats = run_summaries[run_num]['statistics']
            
            # Success rate trend
            trends['success_rate_trend'].append({
                'run': run_num,
                'success_rate': stats.get('success_rate', 0)
            })
            
            # Response time trend
            trends['response_time_trend'].append({
                'run': run_num,
                'avg_response_time': stats.get('average_response_time', 0)
            })
            
            # Token usage trend
            trends['token_usage_trend'].append({
                'run': run_num,
                'total_tokens': (stats.get('total_input_tokens', 0) + stats.get('total_output_tokens', 0))
            })
            
            # Provider dominance (most used provider)
            evaluations = run_summaries[run_num]['evaluations']
            provider_counts = Counter(e['provider'] for e in evaluations)
            most_used = provider_counts.most_common(1)[0] if provider_counts else ('none', 0)
            
            trends['provider_dominance_trend'].append({
                'run': run_num,
                'dominant_provider': most_used[0],
                'usage_count': most_used[1]
            })
        
        # Calculate improvement metrics
        if len(sorted_runs) >= 2:
            first_run = trends['success_rate_trend'][0]
            last_run = trends['success_rate_trend'][-1]
            
            trends['improvement_metrics'] = {
                'success_rate_change': last_run['success_rate'] - first_run['success_rate'],
                'response_time_change': (trends['response_time_trend'][-1]['avg_response_time'] - 
                                       trends['response_time_trend'][0]['avg_response_time']),
                'total_runs_analyzed': len(sorted_runs),
                'overall_trend': 'improving' if last_run['success_rate'] > first_run['success_rate'] else 'declining'
            }
        
        return trends
    
    def _generate_comparison_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on comparison"""
        recommendations = []
        
        # Provider recommendations
        provider_comparison = comparison['provider_comparison']
        if provider_comparison:
            # Find best performing provider
            best_provider = max(provider_comparison.items(), 
                              key=lambda x: x[1]['success_rate'])[0]
            best_success_rate = provider_comparison[best_provider]['success_rate']
            
            recommendations.append(f"Best performing provider: {best_provider} ({best_success_rate:.1%} success rate)")
            
            # Find fastest provider
            providers_with_times = {p: s for p, s in provider_comparison.items() 
                                  if s['average_response_time'] > 0}
            if providers_with_times:
                fastest_provider = min(providers_with_times.items(), 
                                     key=lambda x: x[1]['average_response_time'])[0]
                fastest_time = providers_with_times[fastest_provider]['average_response_time']
                
                recommendations.append(f"Fastest provider: {fastest_provider} ({fastest_time:.2f}s avg response)")
        
        # Model recommendations
        model_comparison = comparison['model_comparison']
        if model_comparison:
            # Find best performing model
            best_model = max(model_comparison.items(), 
                           key=lambda x: x[1]['success_rate'])[0]
            best_model_success = model_comparison[best_model]['success_rate']
            
            recommendations.append(f"Best performing model: {best_model} ({best_model_success:.1%} success rate)")
        
        # Trend recommendations
        trends = comparison['performance_trends']
        if trends['improvement_metrics']:
            trend = trends['improvement_metrics']['overall_trend']
            success_change = trends['improvement_metrics']['success_rate_change']
            
            if trend == 'improving':
                recommendations.append(f"Performance is improving (+{success_change:.1%} success rate)")
            else:
                recommendations.append(f"Performance is declining ({success_change:.1%} success rate change)")
        
        # Consistency recommendations
        provider_consistency = self._analyze_consistency(provider_comparison)
        if provider_consistency:
            most_consistent = min(provider_consistency.items(), key=lambda x: x[1])[0]
            recommendations.append(f"Most consistent provider: {most_consistent}")
        
        return recommendations
    
    def _analyze_consistency(self, provider_comparison: Dict[str, Any]) -> Dict[str, float]:
        """Analyze consistency of providers across runs"""
        consistency_scores = {}
        
        for provider, stats in provider_comparison.items():
            run_success_rates = [run_data['success_rate'] 
                               for run_data in stats['success_rates_by_run'].values()]
            
            if len(run_success_rates) > 1:
                # Lower standard deviation = more consistent
                consistency_scores[provider] = statistics.stdev(run_success_rates)
        
        return consistency_scores
    
    def generate_performance_report(self, 
                                  run_numbers: List[int],
                                  output_format: str = 'json') -> str:
        """
        Generate a performance report
        
        Args:
            run_numbers: List of run numbers to analyze
            output_format: 'json' or 'text'
            
        Returns:
            Formatted report string
        """
        comparison = self.compare_runs(run_numbers)
        
        if output_format == 'json':
            return json.dumps(comparison, indent=2, default=str)
        
        # Generate text report
        report_lines = []
        report_lines.append("LLM EVALUATION PERFORMANCE REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append(f"Runs Analyzed: {', '.join(map(str, run_numbers))}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 20)
        for rec in comparison['recommendations']:
            report_lines.append(f"• {rec}")
        report_lines.append("")
        
        # Provider Performance
        report_lines.append("PROVIDER PERFORMANCE")
        report_lines.append("-" * 20)
        for provider, stats in comparison['provider_comparison'].items():
            report_lines.append(f"{provider.upper()}:")
            report_lines.append(f"  Success Rate: {stats['success_rate']:.1%}")
            report_lines.append(f"  Avg Response Time: {stats['average_response_time']:.2f}s")
            report_lines.append(f"  Total Evaluations: {stats['total_evaluations']}")
            report_lines.append("")
        
        # Model Performance
        report_lines.append("MODEL PERFORMANCE")
        report_lines.append("-" * 20)
        sorted_models = sorted(comparison['model_comparison'].items(), 
                             key=lambda x: x[1]['success_rate'], reverse=True)
        
        for model, stats in sorted_models[:10]:  # Top 10 models
            report_lines.append(f"{model}:")
            report_lines.append(f"  Success Rate: {stats['success_rate']:.1%}")
            report_lines.append(f"  Avg Response Time: {stats['average_response_time']:.2f}s")
            report_lines.append("")
        
        # Performance Trends
        if comparison['performance_trends']['improvement_metrics']:
            report_lines.append("PERFORMANCE TRENDS")
            report_lines.append("-" * 20)
            metrics = comparison['performance_trends']['improvement_metrics']
            report_lines.append(f"Overall Trend: {metrics['overall_trend'].title()}")
            report_lines.append(f"Success Rate Change: {metrics['success_rate_change']:+.1%}")
            report_lines.append(f"Response Time Change: {metrics['response_time_change']:+.2f}s")
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def get_model_rankings(self, 
                          run_numbers: List[int] = None,
                          metric: str = 'success_rate') -> List[Tuple[str, float]]:
        """
        Get model rankings by specified metric
        
        Args:
            run_numbers: List of run numbers to include (None for all)
            metric: 'success_rate', 'avg_response_time', or 'total_evaluations'
            
        Returns:
            List of (model_name, metric_value) tuples sorted by performance
        """
        # Get evaluations
        if run_numbers:
            all_evaluations = []
            for run_num in run_numbers:
                evaluations = self.db.query_evaluations(run_number=run_num)
                all_evaluations.extend(evaluations)
        else:
            all_evaluations = self.db.query_evaluations()
        
        # Analyze by model
        model_comparison = self._analyze_by_model(all_evaluations, run_numbers or [])
        
        # Extract rankings based on metric
        rankings = []
        for model, stats in model_comparison.items():
            if metric == 'success_rate':
                value = stats['success_rate']
                rankings.append((model, value))
            elif metric == 'avg_response_time':
                if stats['average_response_time'] > 0:
                    rankings.append((model, stats['average_response_time']))
            elif metric == 'total_evaluations':
                rankings.append((model, stats['total_evaluations']))
        
        # Sort rankings
        if metric == 'avg_response_time':
            rankings.sort(key=lambda x: x[1])  # Lower is better
        else:
            rankings.sort(key=lambda x: x[1], reverse=True)  # Higher is better
        
        return rankings
    
    def analyze_cost_efficiency(self, run_numbers: List[int]) -> Dict[str, Any]:
        """
        Analyze cost efficiency across providers/models
        
        Note: This is a placeholder for cost analysis.
        Actual implementation would require cost data.
        """
        comparison = self.compare_runs(run_numbers)
        
        cost_analysis = {
            'token_efficiency': {},
            'time_efficiency': {},
            'success_efficiency': {}
        }
        
        # Token efficiency (tokens per successful evaluation)
        for provider, stats in comparison['provider_comparison'].items():
            if stats['successful_evaluations'] > 0:
                total_tokens = stats['total_input_tokens'] + stats['total_output_tokens']
                cost_analysis['token_efficiency'][provider] = {
                    'tokens_per_success': total_tokens / stats['successful_evaluations'],
                    'total_tokens': total_tokens,
                    'successful_evaluations': stats['successful_evaluations']
                }
        
        # Time efficiency (time per successful evaluation)
        for provider, stats in comparison['provider_comparison'].items():
            if stats['successful_evaluations'] > 0 and stats['response_times']:
                avg_time = statistics.mean(stats['response_times'])
                cost_analysis['time_efficiency'][provider] = {
                    'avg_time_per_success': avg_time,
                    'total_time': sum(stats['response_times']),
                    'successful_evaluations': stats['successful_evaluations']
                }
        
        # Success efficiency (success rate vs resources)
        for provider, stats in comparison['provider_comparison'].items():
            cost_analysis['success_efficiency'][provider] = {
                'success_rate': stats['success_rate'],
                'total_attempts': stats['total_evaluations'],
                'efficiency_score': stats['success_rate'] * 100 / stats['total_evaluations'] if stats['total_evaluations'] > 0 else 0
            }
        
        return cost_analysis