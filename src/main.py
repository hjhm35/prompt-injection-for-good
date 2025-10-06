import click
import os
import logging
from dotenv import load_dotenv
from typing import List, Optional

from .core.evaluator import SimpleEvaluator, FastEvaluator
from .core.batch_evaluator import UltraFastEvaluator
from .core.file_handler import FileHandler
from .core.reference_manager import ReferenceManager
from .core.config_parser import ConfigurationParser
from .providers import list_providers, get_all_models
from .evaluation.human_review import HumanReviewInterface
from .evaluation.analytics import PerformanceAnalyzer
from .utils.cli import (
    validate_files_option, 
    validate_models_option,
    print_models_table,
    print_results_table,
    print_run_statistics,
    export_results_to_csv,
    export_results_to_json
)

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.getenv('LOG_FILE', 'llm_eval.log')
)

@click.group()
@click.version_option(version='1.0.0')
def cli():
    """Prompt Injection For Good - Test security prompts with multiple LLM providers"""
    pass

@cli.command()
@click.option('--files', '-f', multiple=True, callback=validate_files_option,
              help='Files to evaluate (can specify multiple times or comma-separated)')
@click.option('--prompt', '-p',
              help='Prompt to send with the files')
@click.option('--run-number', '-r', type=int,
              help='Run number for grouping evaluations')
@click.option('--prompt-description', '-d',
              help='Description of what this prompt tests')
@click.option('--providers', callback=validate_models_option,
              help='Comma-separated list of providers to use (default: all)')
@click.option('--models', callback=validate_models_option,
              help='Comma-separated list of specific models to use (default: all)')
@click.option('--export-csv', 
              help='Export results to CSV file after evaluation')
@click.option('--export-json',
              help='Export results to JSON file after evaluation')
@click.option('--config-file', '-c', type=click.Path(exists=True),
              help='JSON configuration file (overrides other options)')
@click.option('--fast/--no-fast', default=True,
              help='Use fast concurrent evaluation (default: True)')
@click.option('--ultra/--no-ultra', default=False,
              help='Use ultra-fast batch evaluation (processes ALL tasks simultaneously)')
@click.option('--max-concurrent', type=int, default=15,
              help='Maximum concurrent tasks when using fast evaluation (default: 15)')
def evaluate(files: List[str], prompt: str, run_number: int, 
            prompt_description: Optional[str] = None,
            providers: Optional[List[str]] = None,
            models: Optional[List[str]] = None,
            export_csv: Optional[str] = None,
            export_json: Optional[str] = None,
            config_file: Optional[str] = None,
            fast: bool = True,
            ultra: bool = False,
            max_concurrent: int = 15):
    """Evaluate files with LLM providers"""
    
    # Validate required arguments
    if not config_file:
        if not prompt:
            click.echo("Error: --prompt is required when not using --config-file")
            return
        if run_number is None:
            click.echo("Error: --run-number is required when not using --config-file")
            return
    
    # Handle JSON configuration
    if config_file:
        try:
            parser = ConfigurationParser()
            config = parser.parse_config_file(config_file)
            
            # Resolve relative paths
            config_dir = os.path.dirname(os.path.abspath(config_file))
            config = parser.resolve_relative_paths(config, config_dir)
            
            # Validate file paths
            missing_files = parser.validate_file_paths(config)
            if missing_files:
                click.echo(f"Error: Missing files in configuration: {missing_files}")
                return
            
            # Extract parameters from config
            params = parser.extract_evaluation_params(config)
            
            # Override command line arguments with config
            files = [f['file_path'] for f in params['files']]
            run_number = params['run_number']
            providers = params.get('providers') or providers
            models = params.get('models') or models
            
            # Handle export settings from config
            export_config = params.get('export_config', {})
            export_csv = export_config.get('csv_file') or export_csv
            export_json = export_config.get('json_file') or export_json
            
            click.echo(f"Using configuration from {config_file}")
            click.echo(f"Evaluating {len(files)} file(s) from config")
            
            # Show queued files
            for file_config, file_prompt in zip(params['files'], params['prompts']):
                file_path = file_config['file_path']
                file_description = file_config['description']
                click.echo(f"Queued: {os.path.basename(file_path)} - {file_description or 'No description'}")
            
            # Now evaluate ALL tasks concurrently
            click.echo(f"\nStarting evaluation of {len(params['files'])} files with {len(providers or [])} providers and {len(models or [])} models")
            
            # Choose evaluator based on options
            if ultra:
                evaluator = UltraFastEvaluator(max_concurrent_tasks=max_concurrent)
                click.echo(f"Using ULTRA-FAST batch evaluation (max {max_concurrent} concurrent tasks)")
                
                try:
                    summary = evaluator.evaluate_config_batch(
                        file_configs=params['files'],
                        prompts=params['prompts'],
                        run_number=run_number,
                        providers=providers,
                        models=models
                    )
                    
                    click.echo(f"\nBatch completed: {summary['successful_tasks']}/{summary['total_tasks']} successful")
                    click.echo(f"Total time: {summary['total_time']:.2f}s")
                    click.echo(f"Average time per task: {summary['average_time_per_task']:.2f}s")
                    
                    if summary['failed_tasks'] > 0:
                        click.echo(f"Failed tasks: {summary['failed_tasks']}")
                    
                except Exception as e:
                    click.echo(f"Error during ultra-fast batch evaluation: {e}")
                finally:
                    evaluator.close()
            
            elif fast:
                evaluator = FastEvaluator(max_concurrent_tasks=max_concurrent)
                click.echo(f"Using fast concurrent evaluation (max {max_concurrent} concurrent tasks)")
                
                try:
                    total_successful = 0
                    total_tasks = 0
                    
                    for file_config, file_prompt in zip(params['files'], params['prompts']):
                        file_path = file_config['file_path']
                        file_description = file_config['description']
                        
                        summary = evaluator.evaluate_files(
                            files=[file_path],
                            prompt=file_prompt,
                            run_number=run_number,
                            prompt_description=file_description,
                            providers=providers,
                            models=models
                        )
                        
                        total_successful += summary['successful_tasks']
                        total_tasks += summary['total_tasks']
                        
                        click.echo(f"✓ {os.path.basename(file_path)}: {summary['successful_tasks']}/{summary['total_tasks']} successful")
                    
                    click.echo(f"\nOverall: {total_successful}/{total_tasks} successful")
                    
                except Exception as e:
                    click.echo(f"Error during fast evaluation: {e}")
                finally:
                    evaluator.close()
            
            else:
                evaluator = SimpleEvaluator()
                click.echo("Using sequential evaluation")
                
                try:
                    total_successful = 0
                    total_tasks = 0
                    
                    for file_config, file_prompt in zip(params['files'], params['prompts']):
                        file_path = file_config['file_path']
                        file_description = file_config['description']
                        
                        summary = evaluator.evaluate_files(
                            files=[file_path],
                            prompt=file_prompt,
                            run_number=run_number,
                            prompt_description=file_description,
                            providers=providers,
                            models=models
                        )
                        
                        total_successful += summary['successful_tasks']
                        total_tasks += summary['total_tasks']
                        
                        click.echo(f"✓ {os.path.basename(file_path)}: {summary['successful_tasks']}/{summary['total_tasks']} successful")
                    
                    click.echo(f"\nOverall: {total_successful}/{total_tasks} successful")
                    
                except Exception as e:
                    click.echo(f"Error during sequential evaluation: {e}")
            
            # Export results if requested
            if export_csv or export_json:
                if ultra:
                    temp_evaluator = UltraFastEvaluator(max_concurrent_tasks=max_concurrent)
                elif fast:
                    temp_evaluator = FastEvaluator(max_concurrent_tasks=max_concurrent)
                else:
                    temp_evaluator = SimpleEvaluator()
                
                try:
                    results = temp_evaluator.query_results(run_number=run_number)
                    
                    if export_csv:
                        export_results_to_csv(results, export_csv, run_number)
                    
                    if export_json:
                        export_results_to_json(results, export_json)
                finally:
                    if hasattr(temp_evaluator, 'close'):
                        temp_evaluator.close()
            
            click.echo(f"\nConfiguration-based evaluation completed!")
            return
            
        except Exception as e:
            click.echo(f"Error processing configuration file: {e}")
            logging.error(f"Config file processing failed: {e}")
            return
    
    # Original command-line based evaluation
    if not files:
        click.echo("Error: No files specified. Use --files option or --config-file.")
        return
    
    # Check if any API keys are available
    available_providers = list_providers()
    if not available_providers:
        click.echo("Error: No providers available. Please check your API keys in .env file.")
        return
    
    click.echo(f"Starting evaluation of {len(files)} file(s) with run number {run_number}")
    click.echo(f"Prompt: {prompt}")
    if prompt_description:
        click.echo(f"Description: {prompt_description}")
    
    try:
        if ultra:
            evaluator = UltraFastEvaluator(max_concurrent_tasks=max_concurrent)
            click.echo(f"Using ULTRA-FAST batch evaluation (max {max_concurrent} concurrent tasks)")
        elif fast:
            evaluator = FastEvaluator(max_concurrent_tasks=max_concurrent)
            click.echo(f"Using fast concurrent evaluation (max {max_concurrent} concurrent tasks)")
        else:
            evaluator = SimpleEvaluator()
            click.echo("Using sequential evaluation")
        
        summary = evaluator.evaluate_files(
            files=files,
            prompt=prompt,
            run_number=run_number,
            prompt_description=prompt_description,
            providers=providers,
            models=models
        )
        
        # Display summary
        click.echo(f"\nEvaluation Summary:")
        click.echo(f"Total tasks: {summary['total_tasks']}")
        click.echo(f"Successful: {summary['successful_tasks']}")
        click.echo(f"Failed: {summary['failed_tasks']}")
        click.echo(f"Success rate: {summary['success_rate']:.1%}")
        click.echo(f"Total time: {summary['total_time']:.2f}s")
        click.echo(f"Average time per task: {summary['average_time_per_task']:.2f}s")
        
        if summary['failed_results']:
            click.echo(f"\nSome failures occurred:")
            for error in summary['failed_results'][:5]:
                click.echo(f"  - {error}")
        
        # Export results if requested
        if export_csv or export_json:
            results = evaluator.query_results(run_number=run_number)
            
            if export_csv:
                export_results_to_csv(results, export_csv, run_number)
            
            if export_json:
                export_results_to_json(results, export_json)
        
        # Clean up resources if using FastEvaluator or UltraFastEvaluator
        if ultra or fast:
            evaluator.close()
        
        click.echo(f"\nUse 'llm-eval query --run-number {run_number}' to view detailed results.")
        
    except Exception as e:
        click.echo(f"Error during evaluation: {e}")
        logging.error(f"Evaluation failed: {e}")

@cli.command()
def list_models():
    """List all available models from all providers"""
    print_models_table()

@cli.command()
@click.option('--run-number', '-r', type=int,
              help='Query specific run number')
@click.option('--file-name', '-f',
              help='Filter by file name (partial match)')
@click.option('--provider', '-p',
              help='Filter by provider name')
@click.option('--status', '-s', type=click.Choice(['success', 'error']),
              help='Filter by status')
@click.option('--limit', '-l', type=int, default=50,
              help='Limit number of results (default: 50)')
@click.option('--export-csv',
              help='Export results to CSV file')
@click.option('--export-json',
              help='Export results to JSON file')
@click.option('--show-stats', is_flag=True,
              help='Show run statistics')
def query(run_number: Optional[int] = None,
          file_name: Optional[str] = None,
          provider: Optional[str] = None,
          status: Optional[str] = None,
          limit: int = 50,
          export_csv: Optional[str] = None,
          export_json: Optional[str] = None,
          show_stats: bool = False):
    """Query evaluation results"""
    
    try:
        evaluator = SimpleEvaluator()
        
        # Query results
        results = evaluator.query_results(
            run_number=run_number,
            file_name=file_name,
            provider=provider,
            status=status,
            limit=limit
        )
        
        # Show statistics if requested and run_number specified
        if show_stats and run_number:
            run_results = evaluator.get_run_results(run_number)
            print_run_statistics(run_results['statistics'])
        
        # Display results
        print_results_table(results, limit)
        
        # Export if requested
        if export_csv:
            export_results_to_csv(results, export_csv, run_number)
        
        if export_json:
            export_results_to_json(results, export_json)
            
    except Exception as e:
        click.echo(f"Error querying results: {e}")
        logging.error(f"Query failed: {e}")

@cli.command()
@click.argument('run_number', type=int)
def stats(run_number: int):
    """Show detailed statistics for a specific run"""
    
    try:
        evaluator = SimpleEvaluator()
        run_results = evaluator.get_run_results(run_number)
        
        if not run_results['statistics'].get('total_evaluations'):
            click.echo(f"No evaluations found for run {run_number}")
            return
        
        print_run_statistics(run_results['statistics'])
        
    except Exception as e:
        click.echo(f"Error getting statistics: {e}")
        logging.error(f"Stats failed: {e}")

@cli.command()
def setup():
    """Setup and validate environment"""
    
    click.echo("Prompt Injection For Good Evaluation System Setup")
    click.echo("=" * 30)
    
    # Check .env file
    if os.path.exists('.env'):
        click.echo("✓ .env file found")
    else:
        click.echo("✗ .env file not found")
        click.echo("  Create .env file from .env.example template")
    
    # Check API keys
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    click.echo(f"\nAPI Keys:")
    click.echo(f"OpenAI: {'✓ Set' if openai_key else '✗ Not set'}")
    click.echo(f"Anthropic: {'✓ Set' if anthropic_key else '✗ Not set'}")
    
    if not openai_key and not anthropic_key:
        click.echo("\nWarning: No API keys configured. You won't be able to run evaluations.")
    
    # Test database
    try:
        from .database.operations import DatabaseManager
        db = DatabaseManager()
        db.init_database()
        click.echo(f"\n✓ Database initialized at: {db.database_url}")
    except Exception as e:
        click.echo(f"\n✗ Database error: {e}")
    
    # Show available providers
    available_providers = list_providers()
    click.echo(f"\nAvailable providers: {', '.join(available_providers)}")
    
    click.echo(f"\nSetup complete! Try: llm-eval list-models")

@cli.group()
def reference():
    """Manage reference outputs for evaluation scoring"""
    pass

@reference.command()
@click.option('--file', '-f', required=True, help='File name this reference applies to')
@click.option('--prompt', '-p', required=True, help='The prompt text')
@click.option('--reference', '-r', required=True, help='The reference/ideal output')
@click.option('--description', '-d', help='Description of what this prompt tests')
@click.option('--notes', '-n', help='Additional notes')
def add(file: str, prompt: str, reference: str, description: str = None, notes: str = None):
    """Add a new reference output"""
    try:
        manager = ReferenceManager()
        ref_id = manager.add_reference_output(
            file_name=file,
            prompt_text=prompt,
            reference_output=reference,
            prompt_description=description,
            notes=notes
        )
        click.echo(f"Added reference output {ref_id} for file '{file}'")
    except Exception as e:
        click.echo(f"Error adding reference: {e}")

@reference.command()
@click.option('--file', '-f', help='Filter by file name')
@click.option('--prompt', '-p', help='Filter by prompt text')
@click.option('--limit', '-l', type=int, default=20, help='Limit number of results')
def list(file: str = None, prompt: str = None, limit: int = 20):
    """List reference outputs"""
    try:
        manager = ReferenceManager()
        references = manager.list_reference_outputs(file_name=file, prompt_text=prompt)
        
        if not references:
            click.echo("No reference outputs found")
            return
        
        click.echo(f"\nFound {len(references)} reference output(s):")
        click.echo("=" * 80)
        
        for ref in references[:limit]:
            click.echo(f"\nID: {ref['id']}")
            click.echo(f"File: {ref['file_name']}")
            click.echo(f"Prompt: {ref['prompt_text'][:100]}...")
            click.echo(f"Reference: {ref['reference_output'][:150]}...")
            if ref['prompt_description']:
                click.echo(f"Description: {ref['prompt_description']}")
            click.echo(f"Created: {ref['created_date']}")
            click.echo("-" * 40)
    except Exception as e:
        click.echo(f"Error listing references: {e}")

@reference.command()
@click.argument('csv_file', type=click.Path(exists=True))
def import_csv(csv_file: str):
    """Import reference outputs from CSV file"""
    try:
        manager = ReferenceManager()
        imported, errors = manager.import_from_csv(csv_file)
        
        click.echo(f"Imported {imported} reference outputs")
        
        if errors:
            click.echo(f"\nErrors encountered:")
            for error in errors[:10]:
                click.echo(f"  - {error}")
            if len(errors) > 10:
                click.echo(f"  ... and {len(errors) - 10} more errors")
    except Exception as e:
        click.echo(f"Error importing CSV: {e}")

@reference.command()
@click.argument('output_file')
@click.option('--format', type=click.Choice(['csv', 'json']), default='csv')
def export(output_file: str, format: str):
    """Export reference outputs to file"""
    try:
        manager = ReferenceManager()
        
        if format == 'csv':
            count = manager.export_to_csv(output_file)
        else:
            count = manager.export_to_json(output_file)
        
        click.echo(f"Exported {count} reference outputs to {output_file}")
    except Exception as e:
        click.echo(f"Error exporting: {e}")

@reference.command()
def stats():
    """Show reference output statistics"""
    try:
        manager = ReferenceManager()
        stats = manager.get_statistics()
        
        click.echo("\nReference Output Statistics:")
        click.echo("=" * 30)
        click.echo(f"Total References: {stats['total_references']}")
        click.echo(f"Active References: {stats['active_references']}")
        click.echo(f"Files with References: {stats['files_with_references']}")
        
        if stats['references_by_file']:
            click.echo(f"\nTop Files by Reference Count:")
            for file_name, count in list(stats['references_by_file'].items())[:5]:
                click.echo(f"  {file_name}: {count}")
        
        if stats['references_by_creator']:
            click.echo(f"\nReferences by Creator:")
            for creator, count in stats['references_by_creator'].items():
                click.echo(f"  {creator}: {count}")
    except Exception as e:
        click.echo(f"Error getting statistics: {e}")

@cli.command()
@click.option('--run-number', '-r', type=int, help='Review specific run number')
@click.option('--confidence-threshold', '-c', type=float, default=0.7, 
              help='Only review items below this confidence')
@click.option('--limit', '-l', type=int, default=10, help='Maximum items to review')
@click.option('--reviewer-name', help='Name of the reviewer (defaults to system user)')
def review(run_number: int = None, confidence_threshold: float = 0.7, 
          limit: int = 10, reviewer_name: str = None):
    """Start interactive human review of evaluations"""
    try:
        reviewer = HumanReviewInterface()
        
        # Get items for review
        items = reviewer.get_items_for_review(
            run_number=run_number,
            confidence_threshold=confidence_threshold,
            limit=limit
        )
        
        if not items:
            click.echo("No items need review!")
            return
        
        click.echo(f"Found {len(items)} items for review")
        
        if click.confirm("Start interactive review?"):
            summary = reviewer.conduct_interactive_review(
                run_number=run_number,
                confidence_threshold=confidence_threshold,
                reviewer_name=reviewer_name
            )
            
            click.echo(f"\nReview completed!")
            click.echo(f"Items reviewed: {summary['reviewed_count']}")
            click.echo(f"Items skipped: {summary['skipped_count']}")
        
    except Exception as e:
        click.echo(f"Error during review: {e}")
        logging.error(f"Review failed: {e}")

@cli.command()
@click.option('--run-number', '-r', type=int, help='Get review stats for specific run')
@click.option('--reviewer', help='Get stats for specific reviewer')
def review_stats(run_number: int = None, reviewer: str = None):
    """Show human review statistics"""
    try:
        reviewer_interface = HumanReviewInterface()
        stats = reviewer_interface.get_review_statistics(
            run_number=run_number,
            reviewer_name=reviewer
        )
        
        if stats['total_reviews'] == 0:
            click.echo("No reviews found")
            return
        
        click.echo(f"\nHuman Review Statistics:")
        click.echo("=" * 30)
        click.echo(f"Total Reviews: {stats['total_reviews']}")
        click.echo(f"Average Score: {stats['average_score']:.2f}")
        click.echo(f"Pass Rate: {stats['pass_rate']:.1%}")
        
        click.echo(f"\nScore Distribution:")
        for score, count in stats['score_distribution'].items():
            click.echo(f"  Score {score}: {count}")
        
        if len(stats['reviewers']) > 1:
            click.echo(f"\nReviews by Reviewer:")
            for reviewer_name, count in stats['reviews_by_reviewer'].items():
                avg_score = stats['average_by_reviewer'][reviewer_name]
                click.echo(f"  {reviewer_name}: {count} reviews (avg: {avg_score:.2f})")
        
        if stats['latest_review']:
            click.echo(f"\nLatest Review: {stats['latest_review']}")
    
    except Exception as e:
        click.echo(f"Error getting review statistics: {e}")

@cli.command()
@click.argument('run_numbers', nargs=-1, type=int, required=True)
@click.option('--format', type=click.Choice(['text', 'json']), default='text', help='Output format')
@click.option('--output', '-o', help='Save report to file')
def compare(run_numbers: List[int], format: str, output: str = None):
    """Compare performance across multiple evaluation runs"""
    try:
        analyzer = PerformanceAnalyzer()
        
        if len(run_numbers) < 2:
            click.echo("Error: At least 2 run numbers required for comparison")
            return
        
        click.echo(f"Comparing runs: {', '.join(map(str, run_numbers))}")
        
        # Generate report
        report = analyzer.generate_performance_report(list(run_numbers), format)
        
        if output:
            with open(output, 'w') as f:
                f.write(report)
            click.echo(f"Report saved to {output}")
        else:
            click.echo("\n" + report)
    
    except Exception as e:
        click.echo(f"Error generating comparison: {e}")
        logging.error(f"Comparison failed: {e}")

@cli.command()
@click.option('--run-numbers', '-r', help='Comma-separated run numbers (default: all runs)')
@click.option('--metric', type=click.Choice(['success_rate', 'avg_response_time', 'total_evaluations']), 
              default='success_rate', help='Ranking metric')
@click.option('--limit', '-l', type=int, default=10, help='Number of top models to show')
def rankings(run_numbers: str = None, metric: str = 'success_rate', limit: int = 10):
    """Show model rankings by performance metric"""
    try:
        analyzer = PerformanceAnalyzer()
        
        # Parse run numbers
        run_list = None
        if run_numbers:
            run_list = [int(x.strip()) for x in run_numbers.split(',')]
        
        # Get rankings
        rankings_list = analyzer.get_model_rankings(run_list, metric)
        
        if not rankings_list:
            click.echo("No model data found")
            return
        
        click.echo(f"\nModel Rankings by {metric.replace('_', ' ').title()}:")
        click.echo("=" * 60)
        
        for i, (model, value) in enumerate(rankings_list[:limit], 1):
            if metric == 'success_rate':
                click.echo(f"{i:2d}. {model:<40} {value:.1%}")
            elif metric == 'avg_response_time':
                click.echo(f"{i:2d}. {model:<40} {value:.2f}s")
            else:
                click.echo(f"{i:2d}. {model:<40} {value}")
    
    except Exception as e:
        click.echo(f"Error generating rankings: {e}")

@cli.command()
@click.argument('run_numbers', nargs=-1, type=int, required=True)
@click.option('--output', '-o', help='Save analysis to file')
def cost_analysis(run_numbers: List[int], output: str = None):
    """Analyze cost efficiency across providers and models"""
    try:
        analyzer = PerformanceAnalyzer()
        
        click.echo(f"Analyzing cost efficiency for runs: {', '.join(map(str, run_numbers))}")
        
        analysis = analyzer.analyze_cost_efficiency(list(run_numbers))
        
        # Display results
        click.echo("\nCOST EFFICIENCY ANALYSIS")
        click.echo("=" * 40)
        
        # Token efficiency
        if analysis['token_efficiency']:
            click.echo("\nToken Efficiency (tokens per successful evaluation):")
            for provider, data in analysis['token_efficiency'].items():
                click.echo(f"  {provider}: {data['tokens_per_success']:.0f} tokens/success")
        
        # Time efficiency
        if analysis['time_efficiency']:
            click.echo("\nTime Efficiency (average time per successful evaluation):")
            for provider, data in analysis['time_efficiency'].items():
                click.echo(f"  {provider}: {data['avg_time_per_success']:.2f}s/success")
        
        # Success efficiency
        if analysis['success_efficiency']:
            click.echo("\nSuccess Efficiency:")
            for provider, data in analysis['success_efficiency'].items():
                click.echo(f"  {provider}: {data['success_rate']:.1%} success rate ({data['total_attempts']} attempts)")
        
        # Save to file if requested
        if output:
            import json
            with open(output, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            click.echo(f"\nDetailed analysis saved to {output}")
    
    except Exception as e:
        click.echo(f"Error performing cost analysis: {e}")

@cli.command()
@click.option('--run-number', '-r', type=int, help='Specific run to analyze (default: latest)')
@click.option('--provider', '-p', help='Filter by provider')
@click.option('--model', '-m', help='Filter by model')
def performance(run_number: int = None, provider: str = None, model: str = None):
    """Show detailed performance metrics"""
    try:
        evaluator = SimpleEvaluator()
        
        # Get evaluations
        filters = {}
        if run_number:
            filters['run_number'] = run_number
        if provider:
            filters['provider'] = provider
        
        evaluations = evaluator.query_results(**filters)
        
        if model:
            evaluations = [e for e in evaluations if model.lower() in e.get('model', '').lower()]
        
        if not evaluations:
            click.echo("No evaluations found matching criteria")
            return
        
        # Calculate metrics
        total = len(evaluations)
        successful = len([e for e in evaluations if e['status'] == 'success'])
        failed = total - successful
        
        response_times = [e['response_time'] for e in evaluations if e['response_time']]
        input_tokens = [e['input_tokens'] for e in evaluations if e['input_tokens']]
        output_tokens = [e['output_tokens'] for e in evaluations if e['output_tokens']]
        
        click.echo(f"\nPerformance Metrics")
        click.echo("=" * 25)
        click.echo(f"Total Evaluations: {total}")
        click.echo(f"Successful: {successful} ({successful/total:.1%})")
        click.echo(f"Failed: {failed} ({failed/total:.1%})")
        
        if response_times:
            import statistics
            click.echo(f"\nResponse Times:")
            click.echo(f"  Average: {statistics.mean(response_times):.2f}s")
            click.echo(f"  Median: {statistics.median(response_times):.2f}s")
            click.echo(f"  Min: {min(response_times):.2f}s")
            click.echo(f"  Max: {max(response_times):.2f}s")
        
        if input_tokens:
            click.echo(f"\nToken Usage:")
            click.echo(f"  Total Input Tokens: {sum(input_tokens):,}")
            click.echo(f"  Total Output Tokens: {sum(output_tokens):,}")
            click.echo(f"  Avg Input per Eval: {sum(input_tokens)/len(input_tokens):.0f}")
            click.echo(f"  Avg Output per Eval: {sum(output_tokens)/len(output_tokens):.0f}")
        
        # Provider/Model breakdown
        providers = {}
        models = {}
        
        for eval in evaluations:
            prov = eval['provider']
            mod = eval['model']
            
            if prov not in providers:
                providers[prov] = {'success': 0, 'total': 0}
            if mod not in models:
                models[mod] = {'success': 0, 'total': 0}
            
            providers[prov]['total'] += 1
            models[mod]['total'] += 1
            
            if eval['status'] == 'success':
                providers[prov]['success'] += 1
                models[mod]['success'] += 1
        
        if len(providers) > 1:
            click.echo(f"\nBy Provider:")
            for prov, stats in providers.items():
                rate = stats['success'] / stats['total']
                click.echo(f"  {prov}: {stats['success']}/{stats['total']} ({rate:.1%})")
        
        if len(models) > 1:
            click.echo(f"\nBy Model:")
            for mod, stats in list(models.items())[:10]:  # Top 10
                rate = stats['success'] / stats['total']
                click.echo(f"  {mod}: {stats['success']}/{stats['total']} ({rate:.1%})")
    
    except Exception as e:
        click.echo(f"Error analyzing performance: {e}")

@cli.command()
@click.argument('output_file', default='config.json')
def create_config(output_file: str):
    """Create an example JSON configuration file"""
    try:
        parser = ConfigurationParser()
        parser.create_example_config(output_file)
        
        click.echo(f"Created example configuration file: {output_file}")
        click.echo(f"\nTo use the configuration:")
        click.echo(f"  1. Edit {output_file} with your files and settings")
        click.echo(f"  2. Run: python3 -m src.main evaluate --config-file {output_file}")
        
    except Exception as e:
        click.echo(f"Error creating configuration file: {e}")

if __name__ == '__main__':
    cli()