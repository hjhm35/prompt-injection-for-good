from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from dataclasses import dataclass
import time
import logging

@dataclass
class EvaluationResult:
    """Result of an LLM evaluation"""
    provider: str
    model: str
    output_text: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    response_time: float = 0.0
    status: str = "success"
    error_message: Optional[str] = None
    prompt_text: Optional[str] = None
    prompt_hash: Optional[str] = None

@dataclass
class BulkUploadResult:
    """Result of bulk file upload operation"""
    provider: str
    file_mappings: Dict[str, str]  # file_path -> file_id
    successful_uploads: int
    failed_uploads: int
    upload_time: float = 0.0
    status: str = "success"
    error_message: Optional[str] = None
    errors: Optional[Dict[str, str]] = None  # file_path -> error_message

class BaseProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
    @abstractmethod
    def evaluate_with_file(self, 
                          file_path: str, 
                          prompt: str, 
                          model: str) -> EvaluationResult:
        """
        Evaluate a file with the given prompt using specified model
        
        Args:
            file_path: Path to the file to evaluate
            prompt: Text prompt to send with the file
            model: Model name to use for evaluation
            
        Returns:
            EvaluationResult with the response and metadata
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models that support file upload"""
        pass
    
    @abstractmethod
    def validate_file(self, file_path: str) -> bool:
        """Validate if file format is supported by this provider"""
        pass
    
    def bulk_upload_files(self, file_paths: List[str]) -> BulkUploadResult:
        """
        Upload multiple files to the provider in bulk (optional optimization)
        
        Default implementation uploads files one by one.
        Providers can override this for true bulk upload optimization.
        
        Args:
            file_paths: List of file paths to upload
            
        Returns:
            BulkUploadResult with file mappings and upload metadata
        """
        start_time = time.time()
        file_mappings = {}
        errors = {}
        successful_uploads = 0
        failed_uploads = 0
        
        provider_name = self.__class__.__name__.replace('Provider', '')
        
        self.logger.info(f"Starting bulk upload of {len(file_paths)} files for {provider_name}")
        
        for file_path in file_paths:
            try:
                if not self.validate_file(file_path):
                    errors[file_path] = f"Unsupported file type"
                    failed_uploads += 1
                    continue
                
                # Default: use file path as file ID (no actual upload)
                # Providers that support true upload should override this method
                file_mappings[file_path] = file_path
                successful_uploads += 1
                self.logger.debug(f"Mapped file {file_path} (no upload needed)")
                
            except Exception as e:
                errors[file_path] = str(e)
                failed_uploads += 1
                self.logger.error(f"Failed to process file {file_path}: {e}")
        
        upload_time = time.time() - start_time
        status = "success" if failed_uploads == 0 else ("partial" if successful_uploads > 0 else "error")
        
        result = BulkUploadResult(
            provider=provider_name,
            file_mappings=file_mappings,
            successful_uploads=successful_uploads,
            failed_uploads=failed_uploads,
            upload_time=upload_time,
            status=status,
            errors=errors if errors else None
        )
        
        self.logger.info(f"Bulk upload completed: {successful_uploads} successful, {failed_uploads} failed in {upload_time:.2f}s")
        return result
    
    def evaluate_with_file_reference(self, 
                                   file_id: str, 
                                   file_path: str,
                                   prompt: str, 
                                   model: str) -> EvaluationResult:
        """
        Evaluate using a previously uploaded file reference
        
        Default implementation falls back to evaluate_with_file using file_path.
        Providers with true upload support should override this method.
        
        Args:
            file_id: File ID from bulk upload result
            file_path: Original file path (fallback for providers without upload)
            prompt: Text prompt to send with the file
            model: Model name to use for evaluation
            
        Returns:
            EvaluationResult with the response and metadata
        """
        # Default fallback: use the original file path
        return self.evaluate_with_file(file_path, prompt, model)
    
    @staticmethod
    def _measure_time(func):
        """Decorator to measure execution time"""
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            try:
                result = func(self, *args, **kwargs)
                if hasattr(result, 'response_time'):
                    result.response_time = time.time() - start_time
                return result
            except Exception as e:
                error_result = EvaluationResult(
                    provider=self.__class__.__name__.replace('Provider', ''),
                    model="unknown",
                    output_text="",
                    response_time=time.time() - start_time,
                    status="error",
                    error_message=str(e)
                )
                return error_result
        return wrapper