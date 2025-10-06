import click
import json
from typing import List, Dict, Any
from ..providers import get_all_models

def validate_files_option(ctx, param, value):
    """Validate files option"""
    if not value:
        return []
    
    # Handle both comma-separated and space-separated files
    if isinstance(value, tuple):
        files = list(value)
    else:
        files = [f.strip() for f in str(value).split(',') if f.strip()]
    
    return files

def validate_models_option(ctx, param, value):
    """Validate models option"""
    if not value:
        return None
    
    if isinstance(value, str):
        models = [m.strip() for m in value.split(',') if m.strip()]
    else:
        models = list(value)
    
    return models

def print_models_table():
    """Print available models in a nice table format"""
    all_models = get_all_models()
    
    if not all_models:
        click.echo("No providers available. Please check your API keys.")
        return
    
    click.echo("\nAvailable Models:")
    click.echo("=" * 50)
    
    for provider, models in all_models.items():
        click.echo(f"\n{provider.upper()}:")
        if models:
            for model in models:
                click.echo(f"  • {model}")
        else:
            click.echo("  No models available (check API key)")

def print_results_table(results: List[Dict[str, Any]], limit: int = None):
    """Print evaluation results in a table format"""
    if not results:
        click.echo("No results found.")
        return
    
    # Limit results if specified
    if limit:
        results = results[:limit]
    
    click.echo(f"\nFound {len(results)} evaluation(s):")
    click.echo("=" * 80)
    
    for result in results:
        click.echo(f"\nID: {result.get('id', 'N/A')}")
        click.echo(f"Run: {result.get('run_number', 'N/A')}")
        click.echo(f"File: {result.get('file_name', 'N/A')}")
        click.echo(f"Provider: {result.get('provider', 'N/A')}")
        click.echo(f"Model: {result.get('model', 'N/A')}")
        click.echo(f"Status: {result.get('status', 'N/A')}")
        
        if result.get('status') == 'success':
            click.echo(f"Response Time: {result.get('response_time', 'N/A'):.2f}s")
            click.echo(f"Input Tokens: {result.get('input_tokens', 'N/A')}")
            click.echo(f"Output Tokens: {result.get('output_tokens', 'N/A')}")
            
            # Show first 200 chars of output
            output = result.get('output', '')
            if output:
                preview = output[:200] + "..." if len(output) > 200 else output
                click.echo(f"Output Preview: {preview}")
        else:
            error = result.get('error', 'Unknown error')
            click.echo(f"Error: {error}")
        
        if result.get('prompt_description'):
            click.echo(f"Description: {result.get('prompt_description')}")
        
        click.echo("-" * 40)

def print_run_statistics(stats: Dict[str, Any]):
    """Print statistics for a run"""
    if not stats:
        click.echo("No statistics available.")
        return
    
    click.echo(f"\nRun {stats.get('run_number')} Statistics:")
    click.echo("=" * 40)
    click.echo(f"Total Evaluations: {stats.get('total_evaluations', 0)}")
    click.echo(f"Successful: {stats.get('successful_evaluations', 0)}")
    click.echo(f"Failed: {stats.get('failed_evaluations', 0)}")
    click.echo(f"Success Rate: {stats.get('success_rate', 0):.1%}")
    
    if stats.get('average_response_time'):
        click.echo(f"Average Response Time: {stats.get('average_response_time'):.2f}s")
    
    if stats.get('total_input_tokens'):
        click.echo(f"Total Input Tokens: {stats.get('total_input_tokens')}")
    
    if stats.get('total_output_tokens'):
        click.echo(f"Total Output Tokens: {stats.get('total_output_tokens')}")
    
    if stats.get('providers_tested'):
        click.echo(f"Providers: {', '.join(stats.get('providers_tested', []))}")
    
    if stats.get('models_tested'):
        click.echo(f"Models: {', '.join(stats.get('models_tested', []))}")

def export_results_to_csv(results: List[Dict[str, Any]], filename: str, run_number: int = None):
    """Export results to CSV file using database manager"""
    from ..database.operations import DatabaseManager
    
    if not results:
        click.echo("No results to export.")
        return
    
    # If run_number is provided, query directly from database for consistency
    if run_number:
        db = DatabaseManager()
        evaluations = db.query_evaluations(run_number=run_number)
        db.export_evaluations_to_csv_file(evaluations, filename)
    else:
        # Fallback to direct CSV writing for dictionary data
        import csv
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
    
    click.echo(f"Results exported to {filename}")

def export_results_to_json(results: List[Dict[str, Any]], filename: str):
    """Export results to JSON file"""
    if not results:
        click.echo("No results to export.")
        return
    
    with open(filename, 'w', encoding='utf-8') as jsonfile:
        json.dump(results, jsonfile, indent=2, ensure_ascii=False, default=str)
    
    click.echo(f"Results exported to {filename}")