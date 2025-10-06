import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

from ..providers import get_provider, list_providers, get_all_models
from ..providers.base import BulkUploadResult
from ..database.operations import DatabaseManager
from .file_handler import FileHandler
from .async_evaluator import RateLimiter


@dataclass
class BatchTask:
    """Individual evaluation task in a batch"""
    file_path: str
    prompt: str
    prompt_description: Optional[str]
    provider_name: str
    model_name: str
    run_number: int
    task_id: str  # Unique identifier for this task
    file_id: Optional[str] = None  # File ID from bulk upload (if available)


class BatchEvaluator:
    """High-performance batch evaluator that processes ALL tasks simultaneously"""
    
    # Rate limits per provider (requests per minute)
    RATE_LIMITS = {
        'openai': RateLimiter(max_requests=50, time_window=60.0),
        'anthropic': RateLimiter(max_requests=50, time_window=60.0),
        'deepseek': RateLimiter(max_requests=30, time_window=60.0),  # Conservative limit for DeepSeek
        'together': RateLimiter(max_requests=40, time_window=60.0),     # Together AI rate limit
        'gemini': RateLimiter(max_requests=60, time_window=60.0),    # Google Gemini rate limit
        'default': RateLimiter(max_requests=30, time_window=60.0)
    }
    
    def __init__(self, database_manager: DatabaseManager = None, max_concurrent_tasks: int = 20):
        self.db = database_manager or DatabaseManager()
        self.file_handler = FileHandler()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Initialize database
        self.db.init_database()
        
        # Thread pool for blocking operations
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        
        # Progress tracking
        self.progress_queue = queue.Queue()
        self.completed_tasks = 0
        self.total_tasks = 0
    
    def create_batch_tasks(self, 
                          file_configs: List[Dict[str, Any]],
                          prompts: List[str],
                          run_number: int,
                          providers: List[str] = None,
                          models: List[str] = None) -> List[BatchTask]:
        """Create all individual tasks for batch processing"""
        
        # Determine providers and models to use
        available_providers = list_providers()
        self.logger.info(f"Available providers: {available_providers}")
        self.logger.info(f"Requested providers: {providers}")
        
        if providers is None:
            providers = available_providers
        else:
            providers = [p for p in providers if p in available_providers]
            if not providers:
                raise ValueError(f"No valid providers specified. Available: {available_providers}")
        
        all_models = get_all_models()
        self.logger.info(f"All available models: {all_models}")
        self.logger.info(f"Requested models: {models}")
        
        # Validate that models were explicitly specified (not None or empty)
        if models is not None and len(models) == 0:
            raise ValueError("No models specified for evaluation. Please select at least one model.")
        
        # Create all tasks
        tasks = []
        task_counter = 0
        
        for file_config, prompt in zip(file_configs, prompts):
            file_path = file_config['file_path']
            description = file_config.get('description', '')
            
            for provider_name in providers:
                provider_models = all_models.get(provider_name, [])
                
                if models:
                    # Use specified models if they're available for this provider
                    provider_models = [m for m in models if m in provider_models]
                
                if not provider_models:
                    self.logger.warning(f"No models available for provider {provider_name}")
                    continue
                
                for model_name in provider_models:
                    task_counter += 1
                    tasks.append(BatchTask(
                        file_path=file_path,
                        prompt=prompt,
                        prompt_description=description,
                        provider_name=provider_name,
                        model_name=model_name,
                        run_number=run_number,
                        task_id=f"task_{task_counter}"
                    ))
        
        self.logger.info(f"Created {len(tasks)} tasks total")
        return tasks
    
    def bulk_upload_files_for_providers(self, 
                                       file_paths: List[str], 
                                       providers: List[str]) -> Dict[str, BulkUploadResult]:
        """
        Upload all files to all selected providers in bulk
        
        Returns a mapping of provider -> BulkUploadResult
        """
        upload_results = {}
        
        self.logger.info(f"Starting bulk upload phase for {len(file_paths)} files across {len(providers)} providers")
        
        for provider_name in providers:
            try:
                self.logger.info(f"Bulk uploading {len(file_paths)} files to {provider_name}...")
                
                # Get provider instance
                provider = get_provider(provider_name)
                
                # Perform bulk upload
                upload_result = provider.bulk_upload_files(file_paths)
                upload_results[provider_name] = upload_result
                
                if upload_result.status == "success":
                    self.logger.info(f"✅ {provider_name}: {upload_result.successful_uploads} files uploaded successfully in {upload_result.upload_time:.2f}s")
                elif upload_result.status == "partial":
                    self.logger.warning(f"⚠️ {provider_name}: {upload_result.successful_uploads} successful, {upload_result.failed_uploads} failed in {upload_result.upload_time:.2f}s")
                else:
                    self.logger.error(f"❌ {provider_name}: Upload failed - {upload_result.error_message}")
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to upload files to {provider_name}: {e}")
                upload_results[provider_name] = BulkUploadResult(
                    provider=provider_name,
                    file_mappings={},
                    successful_uploads=0,
                    failed_uploads=len(file_paths),
                    status="error",
                    error_message=str(e)
                )
        
        # Log summary
        total_successful = sum(result.successful_uploads for result in upload_results.values())
        total_failed = sum(result.failed_uploads for result in upload_results.values())
        self.logger.info(f"Bulk upload phase completed: {total_successful} successful, {total_failed} failed across all providers")
        
        return upload_results
    
    def create_batch_tasks_with_uploads(self, 
                                       file_configs: List[Dict[str, Any]],
                                       prompts: List[str],
                                       run_number: int,
                                       upload_results: Dict[str, BulkUploadResult],
                                       providers: List[str] = None,
                                       models: List[str] = None) -> List[BatchTask]:
        """Create batch tasks using pre-uploaded file references"""
        
        # Get the same provider/model logic from original method
        available_providers = list_providers()
        
        if providers is None:
            providers = available_providers
        else:
            providers = [p for p in providers if p in available_providers]
            if not providers:
                raise ValueError(f"No valid providers specified. Available: {available_providers}")
        
        all_models = get_all_models()
        
        if models is not None and len(models) == 0:
            raise ValueError("No models specified for evaluation. Please select at least one model.")
        
        # Create tasks with file references
        tasks = []
        task_counter = 0
        
        for file_config, prompt in zip(file_configs, prompts):
            file_path = file_config['file_path']
            description = file_config.get('description', '')
            
            for provider_name in providers:
                # Skip providers that failed upload
                if provider_name not in upload_results:
                    self.logger.warning(f"Skipping {provider_name} - no upload results")
                    continue
                    
                upload_result = upload_results[provider_name]
                
                # Get file ID for this provider (fallback to file path if not uploaded)
                file_id = upload_result.file_mappings.get(file_path, file_path)
                
                # Skip files that failed to upload to this provider
                if upload_result.status == "error" or file_path not in upload_result.file_mappings:
                    if upload_result.errors and file_path in upload_result.errors:
                        self.logger.warning(f"Skipping {file_path} for {provider_name}: {upload_result.errors[file_path]}")
                    continue
                
                provider_models = all_models.get(provider_name, [])
                
                if models:
                    # Use specified models if they're available for this provider
                    provider_models = [m for m in models if m in provider_models]
                
                if not provider_models:
                    self.logger.warning(f"No models available for provider {provider_name}")
                    continue
                
                for model_name in provider_models:
                    task_counter += 1
                    tasks.append(BatchTask(
                        file_path=file_path,
                        prompt=prompt,
                        prompt_description=description,
                        provider_name=provider_name,
                        model_name=model_name,
                        run_number=run_number,
                        task_id=f"task_{task_counter}",
                        file_id=file_id  # Include the uploaded file ID
                    ))
        
        self.logger.info(f"Created {len(tasks)} tasks with file references")
        return tasks
    
    async def evaluate_batch(self, 
                           tasks: List[BatchTask],
                           progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Evaluate all tasks concurrently"""
        
        if not tasks:
            return {'total_tasks': 0, 'successful_tasks': 0, 'failed_tasks': 0}
        
        start_time = time.time()
        self.total_tasks = len(tasks)
        self.completed_tasks = 0
        
        self.logger.info(f"Starting batch evaluation of {len(tasks)} tasks")
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        
        # Process all tasks concurrently
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
            'total_tasks': len(tasks),
            'successful_tasks': len(successful_results),
            'failed_tasks': len(failed_results),
            'success_rate': len(successful_results) / len(tasks) if tasks else 0,
            'total_time': end_time - start_time,
            'average_time_per_task': (end_time - start_time) / len(tasks) if tasks else 0,
            'providers_used': list(set(task.provider_name for task in tasks)) if tasks else [],
            'models_used': list(set(task.model_name for task in tasks)) if tasks else [],
            'files_processed': list(set(task.file_path for task in tasks)) if tasks else [],
            'failed_results': failed_results[:10]  # Limit error details
        }
        
        self.logger.info(f"Batch evaluation completed: {summary}")
        return summary
    
    async def _execute_task_with_semaphore(self, 
                                         task: BatchTask, 
                                         semaphore: asyncio.Semaphore, 
                                         progress_callback: Optional[Callable] = None):
        """Execute a task with semaphore protection"""
        async with semaphore:
            # Apply rate limiting
            rate_limiter = self.RATE_LIMITS.get(task.provider_name, self.RATE_LIMITS['default'])
            await rate_limiter.acquire()
            
            if progress_callback:
                progress_callback(f"Starting {Path(task.file_path).name} → {task.provider_name}/{task.model_name}")
            
            # Execute the task in thread pool since providers are sync
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, self._execute_single_task, task)
            
            # Update progress
            self.completed_tasks += 1
            progress_percent = (self.completed_tasks / self.total_tasks) * 100
            
            if progress_callback:
                status = "✓" if result.get('status') == 'success' else "✗"
                progress_callback(f"{status} [{progress_percent:.1f}%] {Path(task.file_path).name} → {task.provider_name}/{task.model_name}")
            
            return result
    
    def _execute_single_task(self, task: BatchTask) -> Dict[str, Any]:
        """Execute a single evaluation task (synchronous)"""
        try:
            # Get provider instance
            provider = get_provider(task.provider_name)
            
            # Execute evaluation using file reference if available, otherwise fall back to file path
            if task.file_id and task.file_id != task.file_path:
                # Use the optimized file reference method
                result = provider.evaluate_with_file_reference(
                    file_id=task.file_id,
                    file_path=task.file_path,  # Keep as fallback
                    prompt=task.prompt,
                    model=task.model_name
                )
                self.logger.debug(f"Used file reference {task.file_id} for {task.file_path}")
            else:
                # Fall back to original file upload method
                result = provider.evaluate_with_file(
                    file_path=task.file_path,
                    prompt=task.prompt,
                    model=task.model_name
                )
                self.logger.debug(f"Used direct file upload for {task.file_path}")
            
            # Prepare database record
            evaluation_data = {
                'evaluation_run_number': task.run_number,
                'file_name': Path(task.file_path).name,
                'input_prompt_text': task.prompt,
                'manual_prompt_description': task.prompt_description,
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
            evaluation_data['task_id'] = task.task_id
            
            return evaluation_data
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Task {task.task_id} failed: {error_msg}")
            
            # Still try to save error to database
            try:
                error_data = {
                    'evaluation_run_number': task.run_number,
                    'file_name': Path(task.file_path).name,
                    'input_prompt_text': task.prompt,
                    'manual_prompt_description': task.prompt_description,
                    'llm_provider': task.provider_name,
                    'model_version': task.model_name,
                    'output_text': '',
                    'status': 'error',
                    'error_message': error_msg,
                    'response_time_seconds': 0.0
                }
                
                evaluation_id = self.db.insert_evaluation(error_data)
                error_data['id'] = evaluation_id
                error_data['task_id'] = task.task_id
                return error_data
                
            except Exception as db_error:
                self.logger.error(f"Failed to save error to database: {db_error}")
                return {
                    'task_id': task.task_id,
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


class UltraFastEvaluator:
    """Ultra-fast evaluator with improved CLI feedback"""
    
    def __init__(self, database_manager: DatabaseManager = None, max_concurrent_tasks: int = 20):
        self.batch_evaluator = BatchEvaluator(database_manager, max_concurrent_tasks)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def evaluate_files(self, 
                      files: List[str],
                      prompt: str,
                      run_number: int,
                      prompt_description: Optional[str] = None,
                      providers: List[str] = None,
                      models: List[str] = None) -> Dict[str, Any]:
        """Evaluate files with ultra-fast concurrent processing"""
        
        # Create file configs
        file_configs = [{'file_path': f, 'description': prompt_description} for f in files]
        prompts = [prompt] * len(files)
        
        # Create all tasks
        tasks = self.batch_evaluator.create_batch_tasks(
            file_configs=file_configs,
            prompts=prompts,
            run_number=run_number,
            providers=providers,
            models=models
        )
        
        print(f"Created {len(tasks)} concurrent tasks")
        
        # Track progress
        def progress_callback(message: str):
            print(f"\r{message}", end="", flush=True)
        
        # Run batch evaluation
        summary = asyncio.run(self.batch_evaluator.evaluate_batch(
            tasks=tasks,
            progress_callback=progress_callback
        ))
        
        print()  # New line after progress
        return summary
    
    def evaluate_config_batch(self,
                            file_configs: List[Dict[str, Any]],
                            prompts: List[str],
                            run_number: int,
                            providers: List[str] = None,
                            models: List[str] = None) -> Dict[str, Any]:
        """Evaluate config-based batch with optimized bulk file uploads"""
        
        # Extract file paths for bulk upload
        file_paths = [config['file_path'] for config in file_configs]
        
        # Determine final providers list
        available_providers = list_providers()
        if providers is None:
            providers = available_providers
        else:
            providers = [p for p in providers if p in available_providers]
            if not providers:
                raise ValueError(f"No valid providers specified. Available: {available_providers}")
        
        print(f"🚀 Starting optimized evaluation with bulk uploads for {len(file_configs)} files across {len(providers)} providers")
        
        # Phase 1: Bulk upload all files to all providers
        print(f"📤 Phase 1: Bulk uploading {len(file_paths)} files to {len(providers)} providers...")
        upload_results = self.batch_evaluator.bulk_upload_files_for_providers(file_paths, providers)
        
        # Phase 2: Create tasks with file references
        print(f"⚙️ Phase 2: Creating evaluation tasks with file references...")
        tasks = self.batch_evaluator.create_batch_tasks_with_uploads(
            file_configs=file_configs,
            prompts=prompts,
            run_number=run_number,
            upload_results=upload_results,
            providers=providers,
            models=models
        )
        
        print(f"✅ Created {len(tasks)} optimized tasks (files already uploaded!)")
        
        # Track progress with improved feedback
        def progress_callback(message: str):
            print(f"\r{message}", end="", flush=True)
        
        # Phase 3: Run batch evaluation using file references
        print(f"🔥 Phase 3: Running evaluations with pre-uploaded files...")
        summary = asyncio.run(self.batch_evaluator.evaluate_batch(
            tasks=tasks,
            progress_callback=progress_callback
        ))
        
        print()  # New line after progress
        
        # Add upload summary to results
        total_upload_time = sum(result.upload_time for result in upload_results.values())
        summary['upload_phase'] = {
            'upload_time': total_upload_time,
            'providers_processed': len(upload_results),
            'total_files_uploaded': sum(result.successful_uploads for result in upload_results.values()),
            'upload_efficiency_gain': "Significant - files uploaded once per provider instead of once per model"
        }
        
        print(f"🎉 Bulk upload optimization completed! Upload time: {total_upload_time:.2f}s")
        return summary
    
    def get_run_results(self, run_number: int) -> Dict[str, Any]:
        """Get results for a specific run"""
        return self.batch_evaluator.get_run_results(run_number)
    
    def query_results(self, **filters) -> List[Dict[str, Any]]:
        """Query results with filters"""
        return self.batch_evaluator.query_results(**filters)
    
    def close(self):
        """Clean up resources"""
        self.batch_evaluator.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()