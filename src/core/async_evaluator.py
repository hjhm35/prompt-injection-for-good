import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

from ..providers import get_provider, list_providers, get_all_models
from ..database.operations import DatabaseManager
from .file_handler import FileHandler


@dataclass
class RateLimiter:
    """Rate limiter for API calls"""
    max_requests: int
    time_window: float
    requests: List[float]
    lock: threading.Lock

    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = threading.Lock()

    async def acquire(self):
        """Acquire permission to make a request"""
        with self.lock:
            now = time.time()
            # Remove old requests outside the time window
            self.requests = [req_time for req_time in self.requests if now - req_time < self.time_window]
            
            if len(self.requests) >= self.max_requests:
                # Calculate how long to wait
                oldest_request = min(self.requests)
                wait_time = self.time_window - (now - oldest_request)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    return await self.acquire()  # Try again
            
            self.requests.append(now)


class AsyncEvaluator:
    """Async evaluation engine with concurrent processing"""
    
    # Rate limits per provider (requests per minute)
    RATE_LIMITS = {
        'openai': RateLimiter(max_requests=50, time_window=60.0),  # 50 requests per minute
        'anthropic': RateLimiter(max_requests=50, time_window=60.0),  # 50 requests per minute
        'default': RateLimiter(max_requests=30, time_window=60.0)  # Conservative default
    }
    
    def __init__(self, database_manager: DatabaseManager = None, max_concurrent_tasks: int = 10):
        self.db = database_manager or DatabaseManager()
        self.file_handler = FileHandler()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Initialize database
        self.db.init_database()
        
        # Thread pool for blocking operations
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
    
    async def evaluate_files(self, 
                           files: List[str],
                           prompt: str,
                           run_number: int,
                           prompt_description: Optional[str] = None,
                           providers: List[str] = None,
                           models: List[str] = None,
                           progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Evaluate multiple files with multiple providers/models concurrently
        
        Args:
            files: List of file paths to evaluate
            prompt: Text prompt to send with files
            run_number: Evaluation run number for grouping
            prompt_description: Description of what the prompt tests
            providers: List of provider names (default: all available)
            models: List of specific models (default: all available)
            progress_callback: Optional callback for progress updates
        
        Returns:
            Summary of evaluation results
        """
        start_time = time.time()
        
        # Validate all files first
        validation_results = self.file_handler.validate_files(files)
        invalid_files = {path: error for path, error in validation_results.items() if error}
        
        if invalid_files:
            self.logger.error(f"Invalid files detected: {invalid_files}")
            raise ValueError(f"Invalid files found:\n" + 
                           "\n".join(f"  {path}: {error}" for path, error in invalid_files.items()))
        
        # Determine providers and models to use
        available_providers = list_providers()
        if providers is None:
            providers = available_providers
        else:
            # Filter to only valid providers
            providers = [p for p in providers if p in available_providers]
            if not providers:
                raise ValueError(f"No valid providers specified. Available: {available_providers}")
        
        all_models = get_all_models()
        
        # Create evaluation tasks
        tasks = []
        for file_path in files:
            for provider_name in providers:
                provider_models = all_models.get(provider_name, [])
                
                if models:
                    # Use specified models if they're available for this provider
                    provider_models = [m for m in models if m in provider_models]
                
                if not provider_models:
                    self.logger.warning(f"No models available for provider {provider_name}")
                    continue
                
                for model_name in provider_models:
                    tasks.append({
                        'file_path': file_path,
                        'prompt': prompt,
                        'prompt_description': prompt_description,
                        'provider_name': provider_name,
                        'model_name': model_name,
                        'run_number': run_number
                    })
        
        if not tasks:
            raise ValueError("No valid provider/model combinations found")
        
        self.logger.info(f"Starting concurrent evaluation of {len(tasks)} tasks")
        
        # Execute tasks concurrently with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        results = await asyncio.gather(
            *[self._execute_task_with_semaphore(task, semaphore, progress_callback) 
              for task in tasks],
            return_exceptions=True
        )
        
        # Process results
        successful_results = []
        failed_results = []
        
        for result in results:
            if isinstance(result, Exception):
                failed_results.append(str(result))
            elif result.get('status') == 'success':
                successful_results.append(result)
            else:
                failed_results.append(result.get('error_message', 'Unknown error'))
        
        end_time = time.time()
        
        # Generate summary
        summary = {
            'run_number': run_number,
            'total_tasks': len(tasks),
            'successful_tasks': len(successful_results),
            'failed_tasks': len(failed_results),
            'success_rate': len(successful_results) / len(tasks) if tasks else 0,
            'total_time': end_time - start_time,
            'average_time_per_task': (end_time - start_time) / len(tasks) if tasks else 0,
            'providers_used': list(set(task['provider_name'] for task in tasks)) if tasks else [],
            'models_used': list(set(task['model_name'] for task in tasks)) if tasks else [],
            'files_processed': list(set(task['file_path'] for task in tasks)) if tasks else [],
            'failed_results': failed_results[:10]  # Limit error details
        }
        
        self.logger.info(f"Concurrent evaluation completed: {summary}")
        return summary
    
    async def _execute_task_with_semaphore(self, task: Dict[str, Any], semaphore: asyncio.Semaphore, progress_callback: Optional[Callable] = None):
        """Execute a task with semaphore protection"""
        async with semaphore:
            # Apply rate limiting
            provider_name = task['provider_name']
            rate_limiter = self.RATE_LIMITS.get(provider_name, self.RATE_LIMITS['default'])
            await rate_limiter.acquire()
            
            if progress_callback:
                progress_callback(f"Starting {Path(task['file_path']).name} with {provider_name}/{task['model_name']}")
            
            # Execute the task in thread pool since providers are sync
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, self._execute_single_task, task)
            
            if progress_callback:
                status = "✓" if result.get('status') == 'success' else "✗"
                progress_callback(f"{status} {Path(task['file_path']).name} with {provider_name}/{task['model_name']}")
            
            return result
    
    def _execute_single_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single evaluation task (synchronous)"""
        try:
            # Get provider instance
            provider = get_provider(task['provider_name'])
            
            # Execute evaluation
            result = provider.evaluate_with_file(
                file_path=task['file_path'],
                prompt=task['prompt'],
                model=task['model_name']
            )
            
            # Prepare database record
            evaluation_data = {
                'evaluation_run_number': task['run_number'],
                'file_name': Path(task['file_path']).name,
                'input_prompt_text': task['prompt'],
                'manual_prompt_description': task['prompt_description'],
                'llm_provider': result.provider,
                'model_version': result.model,
                'output_text': result.output_text,
                'input_token_usage': result.input_tokens,
                'output_token_usage': result.output_tokens,
                'response_time_seconds': result.response_time,
                'status': result.status,
                'error_message': result.error_message
            }
            
            # Insert into database
            evaluation_id = self.db.insert_evaluation(evaluation_data)
            evaluation_data['id'] = evaluation_id
            
            return evaluation_data
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Task failed: {error_msg}")
            
            # Still try to save error to database
            try:
                error_data = {
                    'evaluation_run_number': task['run_number'],
                    'file_name': Path(task['file_path']).name,
                    'input_prompt_text': task['prompt'],
                    'manual_prompt_description': task['prompt_description'],
                    'llm_provider': task['provider_name'],
                    'model_version': task['model_name'],
                    'output_text': '',
                    'status': 'error',
                    'error_message': error_msg,
                    'response_time_seconds': 0.0
                }
                
                evaluation_id = self.db.insert_evaluation(error_data)
                error_data['id'] = evaluation_id
                return error_data
                
            except Exception as db_error:
                self.logger.error(f"Failed to save error to database: {db_error}")
                return {
                    'status': 'error',
                    'error_message': f"Task failed: {error_msg}, DB save failed: {db_error}"
                }
    
    def get_run_results(self, run_number: int) -> Dict[str, Any]:
        """Get results for a specific run"""
        evaluations = self.db.query_evaluations(run_number=run_number)
        statistics = self.db.get_run_statistics(run_number)
        
        return {
            'statistics': statistics,
            'evaluations': self.db.export_to_dict(evaluations)
        }
    
    def query_results(self, **filters) -> List[Dict[str, Any]]:
        """Query results with filters"""
        evaluations = self.db.query_evaluations(**filters)
        return self.db.export_to_dict(evaluations)
    
    def close(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)


class ConcurrentEvaluator:
    """Wrapper that provides synchronous interface to async evaluator"""
    
    def __init__(self, database_manager: DatabaseManager = None, max_concurrent_tasks: int = 10):
        self.async_evaluator = AsyncEvaluator(database_manager, max_concurrent_tasks)
        self.loop = None
        self.thread = None
    
    def evaluate_files(self, 
                      files: List[str],
                      prompt: str,
                      run_number: int,
                      prompt_description: Optional[str] = None,
                      providers: List[str] = None,
                      models: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate multiple files with multiple providers/models concurrently
        (Synchronous interface)
        """
        # Simple progress tracking
        completed_tasks = 0
        total_tasks = 0
        
        def progress_callback(message: str):
            nonlocal completed_tasks
            if message.startswith("✓") or message.startswith("✗"):
                completed_tasks += 1
                progress = completed_tasks / total_tasks if total_tasks > 0 else 0
                print(f"\r[{progress:.1%}] {message}", end="", flush=True)
        
        # Run the async evaluation
        return asyncio.run(self.async_evaluator.evaluate_files(
            files=files,
            prompt=prompt,
            run_number=run_number,
            prompt_description=prompt_description,
            providers=providers,
            models=models,
            progress_callback=progress_callback
        ))
    
    def get_run_results(self, run_number: int) -> Dict[str, Any]:
        """Get results for a specific run"""
        return self.async_evaluator.get_run_results(run_number)
    
    def query_results(self, **filters) -> List[Dict[str, Any]]:
        """Query results with filters"""
        return self.async_evaluator.query_results(**filters)
    
    def close(self):
        """Clean up resources"""
        self.async_evaluator.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()