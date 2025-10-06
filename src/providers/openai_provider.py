from typing import List, Dict
import os
from pathlib import Path
import mimetypes
from datetime import datetime, timedelta

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("OpenAI library not installed. Run: pip install openai")

from .base import BaseProvider, EvaluationResult, BulkUploadResult

class OpenAIProvider(BaseProvider):
    """OpenAI provider implementation"""
    
    # No class-level caching - app-level cache handles this
    
    SUPPORTED_FILE_TYPES = {
        'application/pdf': '.pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
        'application/msword': '.doc',
        'text/plain': '.txt',
        'text/markdown': '.md',
        'message/rfc822': '.eml',
        'text/email': '.eml',
        'image/jpeg': '.jpg',
        'image/png': '.png',
        'image/gif': '.gif',
        'image/webp': '.webp',
    }
    
    # No static models - all models are fetched dynamically from OpenAI API
    
    def __init__(self, api_key: str = None):
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        super().__init__(api_key)
        # Configure client with timeouts
        self.client = OpenAI(
            api_key=api_key,
            timeout=120.0,  # 60 second timeout for all requests
            max_retries=6   # Retry failed requests up to 3 times
        )
        # File processing lock to prevent concurrent processing of same file
        self._file_locks = {}
        import threading
        self._lock_manager = threading.Lock()
        
    def get_available_models(self) -> List[str]:
        """Get list of available OpenAI models from OpenAI API"""
        try:
            dynamic_models = self.get_available_models_from_api()
            if dynamic_models:
                # Extract model names from API response
                model_names = [model['id'] for model in dynamic_models if model['id']]
                self.logger.info(f"Fetched {len(model_names)} dynamic OpenAI models from API")
                return model_names
            else:
                self.logger.warning("No models returned from OpenAI API")
                return []
        except Exception as e:
            self.logger.error(f"Failed to get dynamic models from OpenAI API: {e}")
            return []
    
    def get_available_models_from_api(self) -> List[dict]:
        """Get list of available OpenAI models from OpenAI API (no caching - handled by app)"""
        try:
            # Fetch models from OpenAI API
            models_response = self.client.models.list()
            
            # Process models and extract information
            available_models = []
            for model in models_response.data:
                # Log all available attributes to understand the model structure
                self.logger.debug(f"Model {model.id} attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
                
                # Extract capabilities from API response
                capabilities = getattr(model, 'capabilities', None)
                permissions = getattr(model, 'permissions', None)
                
                # Determine supported endpoint and file capabilities
                preferred_endpoint = self._determine_supported_endpoints(model)
                
                # Skip models where we couldn't determine endpoint from API data
                if preferred_endpoint is None:
                    self.logger.debug(f"Skipping model {model.id} - insufficient API data for endpoint determination")
                    continue
                
                supports_vision = self._determine_vision_capability_from_api(model)
                
                # Only include models that support file uploads
                # - Responses API models support everything
                # - Chat Completions models need vision capability for file inputs  
                # - Legacy Completions models are filtered out
                supports_file_upload = False
                supports_file_search = False
                
                if preferred_endpoint == 'responses':
                    supports_file_upload = True
                    supports_file_search = True
                    self.logger.debug(f"Model {model.id} supports responses API - full file capabilities")
                elif preferred_endpoint == 'chat_completions' and supports_vision:
                    supports_file_upload = True
                    supports_file_search = False
                    self.logger.debug(f"Model {model.id} supports chat completions + vision - file input capabilities")
                elif preferred_endpoint == 'chat_completions':
                    self.logger.debug(f"Skipping model {model.id} - chat completions without vision")
                    continue
                elif preferred_endpoint == 'completions':
                    self.logger.debug(f"Skipping model {model.id} - legacy completions only")
                    continue
                
                if supports_file_upload:
                    context_length = self._get_context_length_from_api(model)
                    
                    model_info = {
                        'id': model.id,
                        'object': getattr(model, 'object', 'model'),
                        'created': getattr(model, 'created', 0),
                        'owned_by': getattr(model, 'owned_by', 'openai'),
                        'description': getattr(model, 'description', ''),
                        'capabilities': capabilities,
                        'permissions': permissions,
                        'preferred_endpoint': preferred_endpoint,
                        'supports_file_upload': supports_file_upload,
                        'supports_file_search': supports_file_search,
                        'supports_vision': supports_vision,
                        'context_length': context_length  # Now always returns a number
                    }
                    available_models.append(model_info)
                
            # Sort by model ID for consistent ordering
            available_models.sort(key=lambda x: x['id'])
            
            self.logger.info(f"Fetched {len(available_models)} OpenAI models from API")
            return available_models
            
        except Exception as e:
            self.logger.error(f"Error fetching models from OpenAI API: {e}")
            return []
    
    def _determine_supported_endpoints(self, model) -> str:
        """Determine which API endpoint this model supports
        
        Note: OpenAI API rarely provides 'supported_methods' field, so this mostly uses
        intelligent inference based on model names and known capabilities.
        """
        try:
            # Check if model has supported_methods field (rare, but most reliable when available)
            if hasattr(model, 'supported_methods') and model.supported_methods:
                supported_methods = model.supported_methods
                self.logger.debug(f"Model {model.id} has API supported_methods: {supported_methods}")
                
                if 'responses' in supported_methods:
                    return 'responses'
                elif 'chat.completions' in supported_methods:
                    return 'chat_completions'
                elif 'completions' in supported_methods:
                    return 'completions'
                else:
                    self.logger.warning(f"Model {model.id} has unsupported methods: {supported_methods}")
            
            # Check capabilities object for hints (sometimes available)
            capabilities = getattr(model, 'capabilities', None)
            if capabilities and (hasattr(capabilities, 'vision') or hasattr(capabilities, 'multimodal')):
                if (hasattr(capabilities, 'vision') and capabilities.vision) or \
                   (hasattr(capabilities, 'multimodal') and capabilities.multimodal):
                    self.logger.debug(f"Model {model.id} has API vision/multimodal capability - chat_completions")
                    return 'chat_completions'
            
            # Since API data is limited, use informed inference based on known model characteristics
            model_id = model.id.lower()
            
            # Modern models likely supporting Responses API
            if 'gpt-5' in model_id or any(pattern in model_id for pattern in ['o1', 'o3', 'o4']):
                self.logger.debug(f"Model {model.id} inferred as modern - likely supports responses")
                return 'responses'
            
            # Standard models supporting Chat Completions
            elif any(pattern in model_id for pattern in ['gpt-4o', 'gpt-4', 'gpt-3.5', 'turbo']):
                self.logger.debug(f"Model {model.id} inferred as standard - likely supports chat_completions")
                return 'chat_completions'
            
            # Skip truly unknown models
            else:
                self.logger.debug(f"Model {model.id} unknown - skipping")
                return None
            
        except Exception as e:
            self.logger.error(f"Error determining supported endpoints for {model.id}: {e}")
            return None
    
    
    def _determine_vision_capability_from_api(self, model) -> bool:
        """Determine if a model supports vision/image processing based on API response data"""
        try:
            # Check if model has capabilities object
            capabilities = getattr(model, 'capabilities', None)
            if capabilities:
                # Check for vision capability
                if hasattr(capabilities, 'vision') and capabilities.vision:
                    return True
                
                # Check for multimodal capability (includes vision)
                if hasattr(capabilities, 'multimodal') and capabilities.multimodal:
                    return True
            
            # Fallback: Check model name patterns for known vision models
            model_id = model.id.lower()
            vision_patterns = [
                'gpt-4o', 'gpt-4-vision', 'gpt-4-turbo', 'vision', 'multimodal'
            ]
            
            for pattern in vision_patterns:
                if pattern in model_id:
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.debug(f"Error determining vision capability for {model.id}: {e}")
            return False
    
    def _get_context_length_from_api(self, model) -> int:
        """Get context length for a model
        
        Note: OpenAI API rarely provides context length data, so this mostly uses
        known defaults for common model types.
        """
        try:
            # Try to get from API capabilities (rarely available)
            capabilities = getattr(model, 'capabilities', None)
            if capabilities:
                if hasattr(capabilities, 'context_length') and capabilities.context_length:
                    self.logger.debug(f"Model {model.id} has API context_length: {capabilities.context_length}")
                    return capabilities.context_length
                
                if hasattr(capabilities, 'max_tokens') and capabilities.max_tokens:
                    self.logger.debug(f"Model {model.id} has API max_tokens: {capabilities.max_tokens}")
                    return capabilities.max_tokens
            
            # Try model object directly (also rarely available)
            if hasattr(model, 'context_length') and model.context_length:
                self.logger.debug(f"Model {model.id} has direct context_length: {model.context_length}")
                return model.context_length
            
            # Since API data is usually unavailable, use known defaults
            model_id = model.id.lower()
            
            if any(pattern in model_id for pattern in ['gpt-5', 'o1', 'o3', 'o4']):
                self.logger.debug(f"Model {model.id} using known default: 200K (modern model)")
                return 200000
            elif 'gpt-4o' in model_id or 'gpt-4-turbo' in model_id:
                self.logger.debug(f"Model {model.id} using known default: 128K")
                return 128000
            elif 'gpt-4' in model_id:
                self.logger.debug(f"Model {model.id} using known default: 8K")
                return 8192
            elif 'gpt-3.5' in model_id:
                self.logger.debug(f"Model {model.id} using known default: 16K")
                return 16384
            else:
                self.logger.debug(f"Model {model.id} using fallback default: 4K")
                return 4096
            
        except Exception as e:
            self.logger.debug(f"Error getting context length for {model.id}: {e}")
            return 4096
    
    def validate_file(self, file_path: str) -> bool:
        """Validate if file is supported by OpenAI"""
        if not os.path.exists(file_path):
            return False
            
        mime_type, _ = mimetypes.guess_type(file_path)
        file_extension = Path(file_path).suffix.lower()
        
        return (mime_type in self.SUPPORTED_FILE_TYPES or 
                file_extension in self.SUPPORTED_FILE_TYPES.values())
    
    def _supports_file_upload(self, model: str) -> bool:
        """Check if model supports file uploads - all returned models support file uploads"""
        # Since we filter models to only include those with file upload capabilities,
        # if a model is in our list, it supports file uploads
        try:
            dynamic_models = self.get_available_models_from_api()
            for model_info in dynamic_models:
                if model_info['id'] == model:
                    return True  # All models in list support file uploads
            return False  # Model not in our filtered list
        except Exception as e:
            self.logger.debug(f"Error checking file upload support for {model}: {e}")
            return False
    
    def _get_model_info(self, model: str) -> dict:
        """Get model information including preferred endpoint"""
        # First try to get from API
        try:
            dynamic_models = self.get_available_models_from_api()
            for model_info in dynamic_models:
                if model_info['id'] == model:
                    self.logger.debug(f"Found {model} in API response")
                    return model_info
        except Exception as e:
            self.logger.debug(f"Error getting model info from API for {model}: {e}")
        
        # If not found in API, provide intelligent defaults based on model name patterns
        model_lower = model.lower()
        
        # GPT-5 models (all variants) - prefer Responses API with file search
        if 'gpt-5' in model_lower or 'gpt5' in model_lower:
            return {
                'id': model,
                'preferred_endpoint': 'responses',
                'supports_file_search': True,
                'supports_vision': True,
                'supports_file_upload': True
            }
        
        # o1, o3, o4 - prefer Responses API with file search  
        elif any(pattern in model_lower for pattern in ['o1', 'o3', 'o4', '4o']):
            return {
                'id': model,
                'preferred_endpoint': 'responses',
                'supports_file_search': True,
                'supports_vision': True, 
                'supports_file_upload': True
            }
        # GPT-4 models - Chat Completions with file upload
        elif 'gpt-4' in model_lower:
            return {
                'id': model,
                'preferred_endpoint': 'chat_completions',
                'supports_file_search': False,
                'supports_vision': 'vision' in model_lower or 'gpt-4o' in model_lower,
                'supports_file_upload': True
            }
        
        # GPT-3.5 and older models - Chat Completions, limited file support
        elif 'gpt-3.5' in model_lower or 'gpt-3' in model_lower:
            return {
                'id': model,
                'preferred_endpoint': 'chat_completions',
                'supports_file_search': False,
                'supports_vision': False,
                'supports_file_upload': False  # Most GPT-3.5 models don't support file upload
            }
        
        # Default for unknown models - try Chat Completions
        else:
            self.logger.warning(f"Unknown model {model}, using default configuration")
            return {
                'id': model,
                'preferred_endpoint': 'chat_completions',
                'supports_file_search': False,
                'supports_vision': False,
                'supports_file_upload': True  # Assume modern models support file upload
            }
    
    def _get_file_lock(self, file_path: str):
        """Get or create a lock for the given file path to prevent concurrent processing"""
        import threading
        import os
        
        # Use absolute path for consistent locking
        abs_file_path = os.path.abspath(file_path)
        
        with self._lock_manager:
            if abs_file_path not in self._file_locks:
                self._file_locks[abs_file_path] = threading.Lock()
            return self._file_locks[abs_file_path]
    
    def bulk_upload_files(self, file_paths: List[str]) -> BulkUploadResult:
        """
        Upload multiple files to OpenAI Files API in bulk
        
        This method uploads all files to OpenAI and returns file IDs for later use.
        Significantly more efficient than uploading files individually for each evaluation.
        """
        import time
        from typing import List
        
        start_time = time.time()
        file_mappings = {}
        errors = {}
        successful_uploads = 0
        failed_uploads = 0
        
        self.logger.info(f"Starting bulk upload of {len(file_paths)} files to OpenAI")
        
        for file_path in file_paths:
            try:
                self.logger.debug(f"Uploading file: {file_path}")
                
                # Validate file before upload
                if not self.validate_file(file_path):
                    errors[file_path] = f"Unsupported file type for OpenAI"
                    failed_uploads += 1
                    self.logger.warning(f"Skipping unsupported file: {file_path}")
                    continue
                
                # Upload file to OpenAI Files API
                with open(file_path, 'rb') as file:
                    uploaded_file = self.client.files.create(
                        file=file,
                        purpose='assistants'
                    )
                
                # Wait for file to be processed (with timeout)
                max_wait_time = 90  # 90 seconds max wait per file
                wait_time = 0
                check_interval = 1
                
                while uploaded_file.status != 'processed' and wait_time < max_wait_time:
                    if uploaded_file.status == 'failed':
                        error_details = getattr(uploaded_file, 'status_details', 'Unknown error')
                        errors[file_path] = f"OpenAI file processing failed: {error_details}"
                        failed_uploads += 1
                        break
                    
                    time.sleep(check_interval)
                    try:
                        uploaded_file = self.client.files.retrieve(uploaded_file.id)
                        wait_time += check_interval
                    except Exception as api_error:
                        self.logger.warning(f"API error checking file status for {file_path}: {api_error}")
                        wait_time += check_interval
                        if wait_time >= max_wait_time:
                            errors[file_path] = f"File processing timeout after {max_wait_time}s"
                            failed_uploads += 1
                            break
                
                # Check final status
                if uploaded_file.status == 'processed':
                    file_mappings[file_path] = uploaded_file.id
                    successful_uploads += 1
                    self.logger.debug(f"Successfully uploaded {file_path} -> {uploaded_file.id}")
                elif file_path not in errors:
                    # Timeout case
                    errors[file_path] = f"File processing timeout after {max_wait_time}s. Final status: {uploaded_file.status}"
                    failed_uploads += 1
                    
            except Exception as e:
                errors[file_path] = f"Upload failed: {str(e)}"
                failed_uploads += 1
                self.logger.error(f"Failed to upload file {file_path}: {e}")
        
        upload_time = time.time() - start_time
        status = "success" if failed_uploads == 0 else ("partial" if successful_uploads > 0 else "error")
        
        result = BulkUploadResult(
            provider="OpenAI",
            file_mappings=file_mappings,
            successful_uploads=successful_uploads,
            failed_uploads=failed_uploads,
            upload_time=upload_time,
            status=status,
            errors=errors if errors else None
        )
        
        self.logger.info(f"OpenAI bulk upload completed: {successful_uploads} successful, {failed_uploads} failed in {upload_time:.2f}s")
        return result
    
    def evaluate_with_file_reference(self, 
                                   file_id: str, 
                                   file_path: str,
                                   prompt: str, 
                                   model: str) -> EvaluationResult:
        """
        Evaluate using a previously uploaded OpenAI file ID with proper strategy precedence
        
        This method replicates the same strategy system as evaluate_with_file but using
        pre-uploaded file IDs for maximum efficiency.
        """
        try:
            self.logger.debug(f"Evaluating with OpenAI file ID: {file_id} (original: {file_path})")
            
            # Get model information to determine strategy (same as original method)
            model_info = self._get_model_info(model)
            preferred_endpoint = model_info.get('preferred_endpoint', 'chat_completions')
            supports_file_search = model_info.get('supports_file_search', False)
            supports_vision = model_info.get('supports_vision', False)
            
            # Determine file processing strategy based on file type and model capabilities
            # (Same logic as original evaluate_with_file method)
            file_extension = file_path.lower()
            is_text_file = file_extension.endswith(('.txt', '.md', '.eml'))
            is_image_file = file_extension.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp'))
            is_document_file = file_extension.endswith(('.pdf', '.docx', '.doc'))
            
            self.logger.debug(f"Processing file ID {file_id} with model {model} (endpoint: {preferred_endpoint}, supports_file_search: {supports_file_search})")
            
            # Strategy 1: Text files - prioritize file search for Responses API models, otherwise inline
            if is_text_file:
                if preferred_endpoint == 'responses' and supports_file_search:
                    self.logger.debug("STRATEGY 1A: Text file with Responses API + file search (using file ID)")
                    return self._responses_api_process_with_file_id(file_id, prompt, model)
                else:
                    self.logger.debug("STRATEGY 1B: Text file - fallback to original method for inline processing")
                    # For text files that need inline processing, we need the actual file content
                    # Fall back to original method
                    return self.evaluate_with_file(file_path, prompt, model)
            
            # Strategy 2: Image files with vision models - may need original file for base64
            elif is_image_file and supports_vision:
                self.logger.debug("STRATEGY 2: Vision API for image file - fallback to original method")
                # Vision API typically needs base64 encoding, so fall back to original method
                return self.evaluate_with_file(file_path, prompt, model)
            
            # Strategy 3: Responses API with file search (best for documents)
            elif preferred_endpoint == 'responses' and supports_file_search:
                self.logger.debug("STRATEGY 3: Responses API with file search for document (using file ID)")
                return self._responses_api_process_with_file_id(file_id, prompt, model)
            
            # Strategy 4: Chat Completions with file upload (modern approach)
            elif preferred_endpoint == 'chat_completions':
                self.logger.debug("STRATEGY 4: Chat Completions with file ID")
                try:
                    return self._chat_completions_with_file_id(file_id, prompt, model)
                except Exception as e:
                    self.logger.warning(f"STRATEGY 4 FAILED: Chat Completions with file ID failed: {e}")
                    # Fallback to original method
                    self.logger.info("Falling back to original evaluate_with_file method")
                    return self.evaluate_with_file(file_path, prompt, model)
            
            # Strategy 5: Responses API models fallback - try Chat Completions with file ID
            elif preferred_endpoint == 'responses':
                self.logger.debug("STRATEGY 5: Responses API model fallback to Chat Completions with file ID")
                try:
                    return self._chat_completions_with_file_id(file_id, prompt, model)
                except Exception as e:
                    self.logger.warning(f"STRATEGY 5 FAILED: Chat Completions fallback with file ID failed: {e}")
                    # Fallback to original method
                    self.logger.info("Falling back to original evaluate_with_file method")
                    return self.evaluate_with_file(file_path, prompt, model)
            
            # Strategy 6: Assistants API fallback (only for models that support it)
            else:
                # Check if model supports Assistants API (modern models like GPT-5, o1, etc. don't)
                model_id = model.lower()
                if any(pattern in model_id for pattern in ['gpt-5', 'o1', 'o3', 'o4', 'o5']):
                    self.logger.debug(f"STRATEGY 6A: Model {model} doesn't support Assistants API, using Chat Completions with file ID")
                    return self._chat_completions_with_file_id(file_id, prompt, model)
                else:
                    self.logger.debug("STRATEGY 6B: Using Assistants API with file ID")
                    try:
                        return self._assistants_api_with_file_id(file_id, prompt, model)
                    except Exception as e:
                        self.logger.warning(f"STRATEGY 6B FAILED: Assistants API with file ID failed: {e}")
                        # Fallback to original method
                        self.logger.info("Falling back to original evaluate_with_file method")
                        return self.evaluate_with_file(file_path, prompt, model)
                        
        except Exception as e:
            self.logger.error(f"File reference evaluation failed: {e}")
            
            # Final fallback to original file path method
            self.logger.info(f"Final fallback to original file upload method for {file_path}")
            return self.evaluate_with_file(file_path, prompt, model)
    
    def _responses_api_process_with_file_id(self, file_id: str, prompt: str, model: str) -> EvaluationResult:
        """Process using Responses API with pre-uploaded file ID"""
        try:
            self.logger.debug(f"Using Responses API with pre-uploaded file ID: {file_id}")
            
            # Create vector store for the file
            vector_store = self.client.vector_stores.create(
                name=f"File analysis for {file_id}",
                file_ids=[file_id]
            )
            
            self.logger.debug(f"Created vector store: {vector_store.id}")
            
            # Wait for vector store to be ready
            import time
            max_wait_time = 90
            wait_time = 0
            check_interval = 2
            
            while vector_store.status != 'completed' and wait_time < max_wait_time:
                if vector_store.status == 'failed':
                    raise Exception(f"Vector store creation failed: {getattr(vector_store, 'last_error', 'Unknown error')}")
                
                time.sleep(check_interval)
                try:
                    vector_store = self.client.vector_stores.retrieve(vector_store.id)
                    wait_time += check_interval
                except Exception as api_error:
                    self.logger.warning(f"API error checking vector store status: {api_error}")
                    wait_time += check_interval
            
            if vector_store.status != 'completed':
                raise Exception(f"Vector store timeout after {max_wait_time}s. Status: {vector_store.status}")
            
            # Use Responses API
            response = self.client.responses.create(
                model=model,
                input=prompt,
                tools=[{
                    "type": "file_search",
                    "vector_store_ids":[vector_store.id]
                    
                }]
            )
            
            output_text = response.output_text
            input_tokens = response.usage.prompt_tokens if response.usage else None
            output_tokens = response.usage.completion_tokens if response.usage else None
            
            # Cleanup
            try:
                self.client.vector_stores.delete(vector_store.id)
                self.logger.debug("Cleaned up vector store")
            except Exception as cleanup_error:
                self.logger.warning(f"Failed to cleanup vector store: {cleanup_error}")
            
            return EvaluationResult(
                provider="OpenAI",
                model=model,
                output_text=output_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                status="success"
            )
            
        except Exception as e:
            self.logger.error(f"Responses API with file ID failed: {e}")
            # Cleanup any resources
            try:
                if 'vector_store' in locals() and vector_store:
                    self.client.vector_stores.delete(vector_store.id)
            except:
                pass
            raise e
    
    def _chat_completions_with_file_id(self, file_id: str, prompt: str, model: str) -> EvaluationResult:
        """Process using Chat Completions API with pre-uploaded file ID"""
        try:
            self.logger.debug(f"Using Chat Completions with pre-uploaded file ID: {file_id}")
            
            # Use Chat Completions API with the uploaded file
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "file",
                                "file": {
                                    "file_id": file_id
                                }
                            }
                        ]
                    }
                ]
            )
            
            output_text = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens if response.usage else None
            output_tokens = response.usage.completion_tokens if response.usage else None
            
            return EvaluationResult(
                provider="OpenAI",
                model=model,
                output_text=output_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                status="success"
            )
            
        except Exception as e:
            self.logger.error(f"Chat Completions with file ID failed: {e}")
            raise e
    
    def _assistants_api_with_file_id(self, file_id: str, prompt: str, model: str) -> EvaluationResult:
        """Process using Assistants API with pre-uploaded file ID"""
        try:
            self.logger.debug(f"Using Assistants API with pre-uploaded file ID: {file_id}")
            
            # Create assistant with file search tool
            assistant = self.client.beta.assistants.create(
                name="File Evaluator",
                instructions="You are a helpful assistant that analyzes uploaded files.",
                model=model,
                tools=[{"type": "file_search"}]
            )
            
            # Create thread and message with file attachment
            thread = self.client.beta.threads.create()
            
            message = self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=prompt,
                attachments=[{
                    "file_id": file_id,
                    "tools": [{"type": "file_search"}]
                }]
            )
            
            # Run assistant
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id
            )
            
            # Wait for completion
            import time
            max_wait_time = 300  # 5 minutes for Assistants API
            wait_time = 0
            check_interval = 3
            
            while run.status in ['queued', 'in_progress'] and wait_time < max_wait_time:
                time.sleep(check_interval)
                try:
                    run = self.client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
                    wait_time += check_interval
                except Exception as api_error:
                    self.logger.warning(f"API error checking run status: {api_error}")
                    wait_time += check_interval
            
            if run.status == 'completed':
                # Get the response
                messages = self.client.beta.threads.messages.list(thread_id=thread.id)
                assistant_message = messages.data[0]
                output_text = assistant_message.content[0].text.value
                
                # Cleanup
                try:
                    self.client.beta.assistants.delete(assistant.id)
                    self.client.beta.threads.delete(thread.id)
                except Exception as cleanup_error:
                    self.logger.warning(f"Failed to cleanup Assistants API resources: {cleanup_error}")
                
                return EvaluationResult(
                    provider="OpenAI",
                    model=model,
                    output_text=output_text,
                    status="success"
                )
            else:
                raise Exception(f"Assistant run failed or timed out. Status: {run.status}")
                
        except Exception as e:
            self.logger.error(f"Assistants API with file ID failed: {e}")
            # Cleanup any resources
            try:
                if 'assistant' in locals() and assistant:
                    self.client.beta.assistants.delete(assistant.id)
                if 'thread' in locals() and thread:
                    self.client.beta.threads.delete(thread.id)
            except:
                pass
            raise e
    
    @BaseProvider._measure_time
    def evaluate_with_file(self, 
                          file_path: str, 
                          prompt: str, 
                          model: str) -> EvaluationResult:
        """Evaluate file with OpenAI using preferred endpoint based on model capabilities"""
        if not self.validate_file(file_path):
            return EvaluationResult(
                provider="OpenAI",
                model=model,
                output_text="",
                status="error",
                error_message=f"Unsupported file type: {file_path}"
            )
        
        # Use file-level locking to prevent concurrent processing of same file
        file_lock = self._get_file_lock(file_path)
        
        # Try to acquire lock with timeout to prevent indefinite blocking
        lock_acquired = file_lock.acquire(timeout=30)  # 30 second timeout for lock
        if not lock_acquired:
            self.logger.warning(f"Could not acquire file lock for {file_path} within 30 seconds")
            return EvaluationResult(
                provider="OpenAI",
                model=model,
                output_text="",
                status="error",
                error_message=f"File processing lock timeout - another process may be using this file"
            )
        
        try:
            self.logger.debug(f"Acquired file lock for {file_path}")
            
            # Get model information to determine preferred endpoint
            model_info = self._get_model_info(model)
            # model_info now always returns a dict with defaults, so no need to check if empty
            
            preferred_endpoint = model_info.get('preferred_endpoint', 'chat_completions')
            supports_file_search = model_info.get('supports_file_search', False)
            # Determine file processing strategy based on file type and model capabilities
            file_extension = file_path.lower()
            is_text_file = file_extension.endswith(('.txt', '.md', '.eml'))
            is_image_file = file_extension.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp'))
            is_document_file = file_extension.endswith(('.pdf', '.docx', '.doc'))
            
            self.logger.debug(f"Processing {file_path} with model {model} (endpoint: {preferred_endpoint}, supports_file_search: {supports_file_search})")
            
            # Strategy 1: Text files - prioritize file search for Responses API models, otherwise inline
            if is_text_file:
                if preferred_endpoint == 'responses' and supports_file_search:
                    self.logger.debug("STRATEGY 1A: Text file with Responses API + file search")
                    return self._process_text_inline(file_path, prompt, model, preferred_endpoint)
                else:
                    self.logger.debug("STRATEGY 1B: Text file with inline processing")
                    return self._process_text_inline(file_path, prompt, model, preferred_endpoint)
            
            # Strategy 2: Image files with vision models - use chat completions with base64
            elif is_image_file and model_info.get('supports_vision', False):
                self.logger.debug("STRATEGY 2: Vision API for image file")
                return self._process_image_vision(file_path, prompt, model)
            
            # Strategy 3: Responses API with file search (best for documents)
            elif preferred_endpoint == 'responses' and supports_file_search:
                self.logger.debug("STRATEGY 3: Responses API with file search for document")
                return self._responses_api_process(file_path, prompt, model)
            
            # Strategy 4: Chat Completions with file upload (modern approach)
            elif preferred_endpoint == 'chat_completions':
                self.logger.debug("STRATEGY 4: Chat Completions with file upload")
                try:
                    return self._chat_completions_file_upload(file_path, prompt, model)
                except Exception as e:
                    self.logger.warning(f"STRATEGY 4 FAILED: Chat Completions file upload failed: {e}")
                    return self._intelligent_fallback(file_path, prompt, model, "chat_completions_failed", 1)
            
            # Strategy 5: Responses API models fallback - try Chat Completions but handle failures intelligently
            elif preferred_endpoint == 'responses':
                self.logger.debug("STRATEGY 5: Responses API model fallback to Chat Completions")
                try:
                    return self._chat_completions_file_upload(file_path, prompt, model)
                except Exception as e:
                    self.logger.warning(f"STRATEGY 5 FAILED: Chat Completions fallback failed: {e}")
                    return self._intelligent_fallback(file_path, prompt, model, "no_responses_api_fallback_failed", 1)
            
            # Strategy 6: Assistants API fallback (only for models that support it)
            else:
                # Check if model supports Assistants API (modern models like GPT-5, o1, etc. don't)
                model_id = model.lower()
                if any(pattern in model_id for pattern in ['gpt-5', 'o1', 'o3', 'o4', 'o5']):
                    self.logger.debug(f"STRATEGY 6A: Model {model} doesn't support Assistants API, using Chat Completions")
                    return self._chat_completions_file_upload(file_path, prompt, model)
                else:
                    self.logger.debug("STRATEGY 6B: Using Assistants API fallback")
                    return self._assistants_api_fallback(file_path, prompt, model)
            
        except Exception as e:
            self.logger.error(f"OpenAI evaluation failed: {e}")
            
            return EvaluationResult(
                provider="OpenAI",
                model=model,
                output_text="",
                status="error",
                error_message=str(e)
            )
        finally:
            # Always release the file lock
            file_lock.release()
            self.logger.debug(f"Released file lock for {file_path}")
    
    def _process_text_inline(self, file_path: str, prompt: str, model: str, endpoint: str) -> EvaluationResult:
        """Process text files by including content inline in the message"""
        try:
            # For Responses API models, try to use file search even for text files
            if endpoint == 'responses':
                model_info = self._get_model_info(model)
                supports_file_search = model_info.get('supports_file_search', False)
                
                if supports_file_search:
                    self.logger.debug(f"Using Responses API with file search for text file {file_path}")
                    try:
                        return self._responses_api_process(file_path, prompt, model)
                    except Exception as e:
                        self.logger.warning(f"Responses API failed for text file: {e}, falling back to inline processing")
            
            # Fallback to inline processing
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
            
            self.logger.debug(f"Using inline text processing for {model}")
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nFile content:\n{file_content}"
                    }
                ]
            )
            
            output_text = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens if response.usage else None
            output_tokens = response.usage.completion_tokens if response.usage else None
            
            return EvaluationResult(
                provider="OpenAI",
                model=model,
                output_text=output_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                status="success"
            )
                
        except Exception as e:
            self.logger.error(f"Text inline processing failed for {model}: {e}")
            return EvaluationResult(
                provider="OpenAI",
                model=model,
                output_text="",
                status="error",
                error_message=f"Text processing failed: {str(e)}"
            )
    
    def _process_image_vision(self, file_path: str, prompt: str, model: str) -> EvaluationResult:
        """Process image files using vision API with base64 encoding"""
        try:
            import base64
            
            # Read and encode image as base64
            with open(file_path, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Determine MIME type
            file_extension = file_path.lower().split('.')[-1]
            mime_types = {
                'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
                'png': 'image/png', 'gif': 'image/gif',
                'webp': 'image/webp'
            }
            mime_type = mime_types.get(file_extension, 'image/jpeg')
            
            # Create vision request
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}"
                                }
                            }
                        ]
                    }
                ]
            )
            
            output_text = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens if response.usage else None
            output_tokens = response.usage.completion_tokens if response.usage else None
            
            return EvaluationResult(
                provider="OpenAI",
                model=model,
                output_text=output_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                status="success"
            )
            
        except Exception as e:
            self.logger.error(f"Vision processing failed for {model}: {e}")
            return EvaluationResult(
                provider="OpenAI",
                model=model,
                output_text="",
                status="error",
                error_message=f"Vision processing failed: {str(e)}"
            )
    
    def _chat_completions_file_upload(self, file_path: str, prompt: str, model: str) -> EvaluationResult:
        """Process files using Chat Completions API with file upload"""
        try:
            # This is essentially the same as _upload_file_and_process but with better naming
            return self._upload_file_and_process(file_path, prompt, model)
            
        except Exception as e:
            self.logger.error(f"Chat completions file upload failed for {model}: {e}")
            return EvaluationResult(
                provider="OpenAI",
                model=model,
                output_text="",
                status="error",
                error_message=f"File upload processing failed: {str(e)}"
            )
    
    def _responses_api_process(self, file_path: str, prompt: str, model: str) -> EvaluationResult:
        """Process file using the Responses API with file search tool"""
        try:
            self.logger.debug(f"Using Responses API with file search for {model}")
            
            # Step 1: Upload file to OpenAI
            with open(file_path, 'rb') as file:
                uploaded_file = self.client.files.create(
                    file=file,
                    purpose='assistants'
                )
            
            self.logger.debug(f"Uploaded file {file_path} with ID: {uploaded_file.id}")
            
            # Step 2: Create a vector store for the file
            vector_store = self.client.vector_stores.create(
                name=f"File analysis for {uploaded_file.id}",
                file_ids=[uploaded_file.id]
            )
            
            self.logger.debug(f"Created vector store: {vector_store.id}")
            
            # Step 3: Wait for vector store to be ready with improved timeout handling
            import time
            max_wait_time = 90  # 90 seconds max wait (increased for large PDFs)
            wait_time = 0
            check_interval = 2  # Start with 2 second intervals
            consecutive_same_status = 0
            last_status = None
            
            while vector_store.status != 'completed' and wait_time < max_wait_time:
                if vector_store.status == 'failed':
                    error_details = getattr(vector_store, 'last_error', 'Unknown error')
                    raise Exception(f"Vector store processing failed: {error_details}")
                
                # Check for stuck processing (same status for too long)
                if vector_store.status == last_status:
                    consecutive_same_status += 1
                    if consecutive_same_status > 10:  # Same status for 20+ seconds
                        self.logger.warning(f"Vector store stuck in {vector_store.status} status for {consecutive_same_status * check_interval}s")
                        # Increase check interval for stuck processing
                        check_interval = min(5, check_interval + 1)
                else:
                    consecutive_same_status = 0
                    check_interval = 2  # Reset to normal interval
                
                last_status = vector_store.status
                self.logger.debug(f"Vector store status: {vector_store.status} (wait: {wait_time}s)")
                
                time.sleep(check_interval)
                try:
                    vector_store = self.client.vector_stores.retrieve(vector_store.id)
                    wait_time += check_interval
                except Exception as api_error:
                    self.logger.warning(f"API error checking vector store status: {api_error}")
                    wait_time += check_interval
                    if wait_time >= max_wait_time:
                        raise Exception(f"Vector store status check failed: {api_error}")
            
            if vector_store.status != 'completed':
                raise Exception(f"Vector store processing timeout after {max_wait_time}s. Final status: {vector_store.status}")
            
            # Step 4: Use Responses API with file search tool
            response = self.client.responses.create(
                model=model,
                input=prompt,
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": [vector_store.id]
                }],
                tool_choice="required"  # Force file search usage
            )
            
            # Step 5: Extract results
            output_text = response.output_text if hasattr(response, 'output_text') else str(response.output)
            
            # Get token usage if available
            input_tokens = getattr(response, 'input_tokens', None)
            output_tokens = getattr(response, 'output_tokens', None)
            
            try:
                self.client.vector_stores.delete(vector_store.id)
                self.client.files.delete(uploaded_file.id)
                self.logger.debug("Cleaned up vector store and uploaded file")
            except Exception as cleanup_error:
                self.logger.warning(f"Failed to cleanup resources: {cleanup_error}")
            
            self.logger.info(f"Successfully processed {file_path} using Responses API")
            
            return EvaluationResult(
                provider="OpenAI",
                model=model,
                output_text=output_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                status="success"
            )
            
        except Exception as e:
            self.logger.error(f"Responses API processing failed: {e}")
            
            try:
                if 'vector_store' in locals() and vector_store:
                    self.client.vector_stores.delete(vector_store.id)
                if 'uploaded_file' in locals() and uploaded_file:
                    self.client.files.delete(uploaded_file.id)
            except:
                pass
            
            # Intelligent fallback based on file type and model capabilities
            return self._intelligent_fallback(file_path, prompt, model, "responses_api_failed", 1)
    
    def _upload_file_and_process(self, file_path: str, prompt: str, model: str) -> EvaluationResult:
        """Upload file to OpenAI and process with Chat Completions API"""
        try:
            # Upload file to OpenAI Files API
            with open(file_path, 'rb') as file:
                uploaded_file = self.client.files.create(
                    file=file,
                    purpose='assistants'
                )
            
            # Wait for file to be processed with improved timeout handling
            import time
            max_wait_time = 90  # 90 seconds max wait (increased for large files)
            wait_time = 0
            check_interval = 1  # Start with 1 second intervals
            consecutive_same_status = 0
            last_status = None
            
            while uploaded_file.status != 'processed' and wait_time < max_wait_time:
                if uploaded_file.status == 'failed':
                    error_details = getattr(uploaded_file, 'status_details', 'Unknown error')
                    return EvaluationResult(
                        provider="OpenAI",
                        model=model,
                        output_text="",
                        status="error",
                        error_message=f"File processing failed: {error_details}"
                    )
                
                # Check for stuck processing (same status for too long)
                if uploaded_file.status == last_status:
                    consecutive_same_status += 1
                    if consecutive_same_status > 30:  # Same status for 30+ seconds
                        self.logger.warning(f"File upload stuck in {uploaded_file.status} status for {consecutive_same_status}s")
                        # Increase check interval for stuck processing
                        check_interval = min(3, check_interval + 1)
                else:
                    consecutive_same_status = 0
                    check_interval = 1  # Reset to normal interval
                
                last_status = uploaded_file.status
                self.logger.debug(f"File upload status: {uploaded_file.status} (wait: {wait_time}s)")
                
                time.sleep(check_interval)
                try:
                    uploaded_file = self.client.files.retrieve(uploaded_file.id)
                    wait_time += check_interval
                except Exception as api_error:
                    self.logger.warning(f"API error checking file status: {api_error}")
                    wait_time += check_interval
                    if wait_time >= max_wait_time:
                        return EvaluationResult(
                            provider="OpenAI",
                            model=model,
                            output_text="",
                            status="error",
                            error_message=f"File status check failed: {api_error}"
                        )
            
            if uploaded_file.status != 'processed':
                return EvaluationResult(
                    provider="OpenAI",
                    model=model,
                    output_text="",
                    status="error",
                    error_message=f"File processing timeout after {max_wait_time}s. Final status: {uploaded_file.status}"
                )
            
            # Create chat completion with file attachment using the correct 'file' format
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "file",
                                "file": {
                                    "file_id": uploaded_file.id
                                }
                            }
                        ]
                    }
                ]
            )
            
            output_text = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens if response.usage else None
            output_tokens = response.usage.completion_tokens if response.usage else None
            
            try:
                self.client.files.delete(uploaded_file.id)
            except Exception as e:
                self.logger.warning(f"Failed to clean up uploaded file: {e}")
            
            return EvaluationResult(
                provider="OpenAI",
                model=model,
                output_text=output_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                status="success"
            )
            
        except Exception as e:
            self.logger.error(f"File upload and processing failed: {e}")
            
            # If the file format doesn't work, try fallbacks based on model capability
            if ("file" in str(e).lower() and ("unsupported" in str(e).lower() or "invalid" in str(e).lower())) or "mime type" in str(e).lower():
                # Check if model supports Assistants API
                model_id = model.lower()
                if any(pattern in model_id for pattern in ['gpt-5', 'o1', 'o3', 'o4', 'o5']):
                    self.logger.warning(f"Model {model} doesn't support Assistants API and Chat Completions file upload failed")
                    # For modern models, we have no other fallbacks - return error
                    return EvaluationResult(
                        provider="OpenAI",
                        model=model,
                        output_text="",
                        status="error",
                        error_message=f"File format not supported by {model}. Chat Completions file upload failed and model doesn't support Assistants API: {str(e)}"
                    )
                else:
                    self.logger.warning("Chat Completions API file format not supported for this file type, trying Assistants API fallback")
                    return self._assistants_api_fallback(file_path, prompt, model)
            
            return EvaluationResult(
                provider="OpenAI",
                model=model,
                output_text="",
                status="error",
                error_message=f"File processing failed: {str(e)}"
            )
    
    def _assistants_api_fallback(self, file_path: str, prompt: str, model: str) -> EvaluationResult:
        """Fallback to Assistants API if input_file format is not supported"""
        try:
            # Upload file to OpenAI Files API
            with open(file_path, 'rb') as file:
                uploaded_file = self.client.files.create(
                    file=file,
                    purpose='assistants'
                )
            
            # Create assistant with file search tool
            assistant = self.client.beta.assistants.create(
                name="File Evaluator",
                instructions="You are a helpful assistant that analyzes uploaded files.",
                model=model,
                tools=[{"type": "file_search"}]
            )
            
            # Create thread and message with file attachment
            thread = self.client.beta.threads.create()
            
            message = self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=prompt,
                attachments=[{
                    "file_id": uploaded_file.id,
                    "tools": [{"type": "file_search"}]
                }]
            )
            
            # Run assistant
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id
            )
            
            # Wait for completion
            import time
            while run.status in ['queued', 'in_progress', 'cancelling']:
                time.sleep(1)
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
            
            if run.status == 'completed':
                # Get messages
                messages = self.client.beta.threads.messages.list(
                    thread_id=thread.id
                )
                
                # Get the latest assistant message
                assistant_messages = [msg for msg in messages.data if msg.role == 'assistant']
                if assistant_messages:
                    output_text = assistant_messages[0].content[0].text.value
                else:
                    output_text = "No response received"
                
                # Get token usage
                input_tokens = getattr(run.usage, 'prompt_tokens', None) if hasattr(run, 'usage') else None
                output_tokens = getattr(run.usage, 'completion_tokens', None) if hasattr(run, 'usage') else None
                
            else:
                return EvaluationResult(
                    provider="OpenAI",
                    model=model,
                    output_text="",
                    status="error",
                    error_message=f"Run failed with status: {run.status}"
                )
            
            try:
                self.client.beta.assistants.delete(assistant.id)
                self.client.files.delete(uploaded_file.id)
            except Exception as e:
                self.logger.warning(f"Failed to clean up resources: {e}")
            
            return EvaluationResult(
                provider="OpenAI",
                model=model,
                output_text=output_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                status="success"
            )
            
        except Exception as e:
            self.logger.error(f"Assistants API fallback failed: {e}")
            return EvaluationResult(
                provider="OpenAI",
                model=model,
                output_text="",
                status="error",
                error_message=f"Both Chat Completions and Assistants API failed: {str(e)}"
            )
    
    def _intelligent_fallback(self, file_path: str, prompt: str, model: str, failure_reason: str, recursion_depth: int = 0) -> EvaluationResult:
        """
        Intelligent fallback logic for when primary processing methods fail.
        Chooses the best alternative based on file type and model capabilities.
        """
        # Prevent infinite recursion
        max_recursion_depth = 3
        if recursion_depth >= max_recursion_depth:
            self.logger.error(f"Maximum fallback recursion depth ({max_recursion_depth}) reached for {model} with {file_path}")
            return EvaluationResult(
                provider="OpenAI",
                model=model,
                output_text="",
                status="error",
                error_message=f"Maximum fallback attempts exceeded. Last failure: {failure_reason}"
            )
        
        file_extension = file_path.lower()
        is_text_file = file_extension.endswith(('.txt', '.md', '.eml'))
        is_image_file = file_extension.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp'))
        is_document_file = file_extension.endswith(('.pdf', '.docx', '.doc'))
        
        model_info = self._get_model_info(model)
        
        self.logger.info(f"Intelligent fallback for {model} with {file_path} (reason: {failure_reason}, depth: {recursion_depth})")
        
        try:
            # Priority 1: If it's a text file, use inline content (most reliable)
            if is_text_file:
                self.logger.debug("Fallback: Using inline text content")
                return self._process_text_inline(file_path, prompt, model, 'chat_completions')
            
            # Priority 2: For images with vision support, try vision API
            elif is_image_file and model_info.get('supports_vision', False):
                self.logger.debug("Fallback: Using vision API for image")
                return self._process_image_vision(file_path, prompt, model)
            
            # Priority 3: For documents, try Chat Completions file upload (but only for supported formats)
            elif is_document_file:
                # Check if the file format is supported by Chat Completions
                if file_extension.endswith('.pdf'):
                    # PDF files are generally supported by Chat Completions
                    self.logger.debug("Fallback: Using Chat Completions for PDF")
                    return self._chat_completions_file_upload(file_path, prompt, model)
                elif file_extension.endswith(('.docx', '.doc')):
                    # DOCX files are NOT supported by Chat Completions API for many models
                    # Try to convert to text first
                    self.logger.debug("Fallback: DOCX not supported by Chat Completions, attempting text extraction")
                    return self._extract_text_and_process(file_path, prompt, model)
                else:
                    # Try Chat Completions file upload as last resort
                    self.logger.debug("Fallback: Trying Chat Completions file upload for document")
                    return self._chat_completions_file_upload(file_path, prompt, model)
            
            # Priority 4: For older models that support Assistants API, try it
            elif not any(pattern in model.lower() for pattern in ['o3', 'o4', 'o5']):
                self.logger.debug("Fallback: Using Assistants API for older model")
                return self._assistants_api_fallback(file_path, prompt, model)
            
            # Final fallback: Extract text and process inline
            else:
                self.logger.debug("Final fallback: Extracting text and processing inline")
                return self._extract_text_and_process(file_path, prompt, model)
                
        except Exception as e:
            self.logger.error(f"All fallback methods failed: {e}")
            return EvaluationResult(
                provider="OpenAI",
                model=model,
                output_text="",
                status="error",
                error_message=f"File format not supported by {model}. {failure_reason}. Error: {str(e)}"
            )
    
    def _extract_text_and_process(self, file_path: str, prompt: str, model: str) -> EvaluationResult:
        """
        Extract text from various file types and process inline.
        This is the most compatible fallback method.
        """
        try:
            file_extension = file_path.lower()
            
            if file_extension.endswith('.pdf'):
                # Try to extract text from PDF with timeout protection
                try:
                    import PyPDF2
                    import os
                    
                    # Check file size to prevent processing huge files
                    file_size = os.path.getsize(file_path)
                    max_file_size = 50 * 1024 * 1024  # 50MB limit
                    if file_size > max_file_size:
                        raise Exception(f"PDF file too large ({file_size / 1024 / 1024:.1f}MB). Max size: {max_file_size / 1024 / 1024}MB")
                    
                    self.logger.debug(f"Extracting text from PDF ({file_size / 1024:.1f}KB)")
                    
                    # Use timeout wrapper for PDF processing
                    def extract_pdf_text():
                        with open(file_path, 'rb') as file:
                            reader = PyPDF2.PdfReader(file)
                            text_content = ""
                            page_count = len(reader.pages)
                            
                            # Limit pages to prevent infinite processing
                            max_pages = 100
                            if page_count > max_pages:
                                self.logger.warning(f"PDF has {page_count} pages, limiting to {max_pages} pages")
                                pages_to_process = reader.pages[:max_pages]
                            else:
                                pages_to_process = reader.pages
                            
                            for i, page in enumerate(pages_to_process):
                                try:
                                    page_text = page.extract_text()
                                    text_content += page_text + "\n"
                                    
                                    # Progress logging for large PDFs
                                    if i > 0 and i % 10 == 0:
                                        self.logger.debug(f"Processed {i}/{len(pages_to_process)} PDF pages")
                                        
                                except Exception as page_error:
                                    self.logger.warning(f"Failed to extract text from page {i}: {page_error}")
                                    continue
                            
                            return text_content
                    
                    # Extract with timeout using threading
                    import threading
                    import queue
                    result_queue = queue.Queue()
                    exception_queue = queue.Queue()
                    
                    def extraction_thread():
                        try:
                            result = extract_pdf_text()
                            result_queue.put(result)
                        except Exception as e:
                            exception_queue.put(e)
                    
                    thread = threading.Thread(target=extraction_thread)
                    thread.daemon = True
                    thread.start()
                    thread.join(timeout=30)  # 30 second timeout for PDF extraction
                    
                    if thread.is_alive():
                        self.logger.error("PDF extraction timeout - thread still running")
                        raise Exception("PDF text extraction timeout after 30 seconds")
                    
                    if not exception_queue.empty():
                        raise exception_queue.get()
                    
                    if not result_queue.empty():
                        text_content = result_queue.get()
                    else:
                        raise Exception("PDF extraction failed - no result returned")
                    
                    if not text_content.strip():
                        raise Exception("Could not extract text from PDF - document may be image-based or corrupted")
                        
                except ImportError:
                    raise Exception("PyPDF2 not available for PDF text extraction. Install with: pip install PyPDF2")
                    
            elif file_extension.endswith(('.docx', '.doc')):
                # Try to extract text from Word documents
                try:
                    import docx
                    doc = docx.Document(file_path)
                    text_content = ""
                    for paragraph in doc.paragraphs:
                        text_content += paragraph.text + "\n"
                    
                    if not text_content.strip():
                        raise Exception("Could not extract text from Word document")
                        
                except ImportError:
                    raise Exception("python-docx not available for Word document text extraction. Install with: pip install python-docx")
                    
            elif file_extension.endswith(('.txt', '.md', '.eml')):
                # Read text files directly
                with open(file_path, 'r', encoding='utf-8') as file:
                    text_content = file.read()
                    
            else:
                raise Exception(f"Text extraction not supported for file type: {file_extension}")
            
            # Process the extracted text inline
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nFile content:\n{text_content}"
                    }
                ]
            )
            
            message = response.choices[0].message
            output_text = message.content
            
            return EvaluationResult(
                provider="OpenAI",
                model=model,
                output_text=output_text,
                input_tokens=response.usage.prompt_tokens if response.usage else None,
                output_tokens=response.usage.completion_tokens if response.usage else None,
                status="success"
            )
            
        except Exception as e:
            self.logger.error(f"Text extraction and processing failed: {e}")
            return EvaluationResult(
                provider="OpenAI",
                model=model,
                output_text="",
                status="error",
                error_message=f"Could not process file {file_path}: {str(e)}"
            )
    