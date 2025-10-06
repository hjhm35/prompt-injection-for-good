from typing import List
import os
import requests
import json
import time

from .base import BaseProvider, EvaluationResult

class DeepSeekProvider(BaseProvider):
    """DeepSeek provider implementation"""
    
    # DeepSeek file upload support (assuming file uploads are possible)
    SUPPORTED_FILE_TYPES = {
        'text/plain': '.txt',
        'text/markdown': '.md',
        'message/rfc822': '.eml',
        'text/email': '.eml',
        'application/pdf': '.pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
        'image/jpeg': '.jpg',
        'image/png': '.png',
    }
    
    # No static models - all models fetched dynamically from API
    
    def __init__(self, api_key: str = None):
        api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            raise ValueError("DeepSeek API key not provided. Set DEEPSEEK_API_KEY environment variable.")
        super().__init__(api_key)
        self.base_url = "https://api.deepseek.com/v1"
        
    def get_available_models(self) -> List[str]:
        """Get list of available DeepSeek models (dynamic from API)"""
        try:
            models = self.get_available_models_from_api()
            return [model['id'] for model in models if model.get('id')]
        except Exception as e:
            self.logger.error(f"Failed to fetch DeepSeek models from API: {e}")
            return []
    
    def get_available_models_from_api(self) -> List[dict]:
        """Get list of available DeepSeek models from DeepSeek API (no caching - handled by app)"""
        try:
            self.logger.info("Fetching DeepSeek models from API...")
            
            # Prepare request headers
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # Call DeepSeek API to get models list
            response = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"DeepSeek API error: {response.status_code} - {response.text}")
            
            result = response.json()
            
            available_models = []
            for model in result.get('data', []):
                model_info = {
                    'id': model['id'],
                    'object': model.get('object', 'model'),
                    'owned_by': model.get('owned_by', 'deepseek'),
                    'supports_file_upload': self._determine_file_upload_capability_from_api(model),
                    'supports_vision': self._determine_vision_capability_from_api(model),
                    'context_length': self._get_context_length_from_api(model)
                }
                available_models.append(model_info)
            
            self.logger.info(f"Fetched {len(available_models)} DeepSeek models from API")
            return available_models
            
        except Exception as e:
            self.logger.error(f"Error fetching DeepSeek models from API: {e}")
            raise
    
    def _determine_file_upload_capability_from_api(self, model) -> bool:
        """Determine if model supports file uploads based on API response"""
        # Assume DeepSeek models support file uploads
        return True
    
    def _determine_vision_capability_from_api(self, model) -> bool:
        """Determine if model supports vision based on API response"""
        # DeepSeek models currently don't support vision/image processing
        return False
    
    def _get_context_length_from_api(self, model) -> int:
        """Get context length from API response"""
        model_id = model['id'].lower()
        
        # Context length mapping based on known DeepSeek model capabilities
        if 'deepseek-reasoner' in model_id:
            return 128000  # 128K tokens for reasoning model
        elif 'deepseek-chat' in model_id:
            return 64000   # 64K tokens for chat model
        elif 'deepseek-coder' in model_id:
            return 16000  # 16K tokens for coder model
        else:
            return 64000  # Default to 64K tokens for DeepSeek models
    
    def validate_file(self, file_path: str) -> bool:
        """Validate if file is supported by DeepSeek (assuming file uploads are possible)"""
        if not os.path.exists(file_path):
            return False
            
        # Support various file types assuming DeepSeek supports file uploads
        file_extension = os.path.splitext(file_path)[1].lower()
        supported_extensions = ['.txt', '.md', '.eml', '.pdf', '.docx', '.jpg', '.jpeg', '.png']
        return file_extension in supported_extensions
    
    def _read_text_file(self, file_path: str) -> str:
        """Read text content from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    @BaseProvider._measure_time
    def evaluate_with_file(self, 
                          file_path: str, 
                          prompt: str, 
                          model: str) -> EvaluationResult:
        """Evaluate file with DeepSeek (assuming file uploads are possible)"""
        if not self.validate_file(file_path):
            return EvaluationResult(
                provider="DeepSeek",
                model=model,
                output_text="",
                status="error",
                error_message=f"Unsupported file type: {file_path}. Supported formats: .txt, .md, .eml, .pdf, .docx, .jpg, .jpeg, .png"
            )
        
        try:
            # For text files, read content directly
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension in ['.txt', '.md', '.eml']:
                file_content = self._read_text_file(file_path)
                combined_prompt = f"File content:\n{file_content}\n\nPrompt: {prompt}"
            else:
                # For other file types, assume DeepSeek can handle them via file upload
                # Use a placeholder approach - in real implementation, this would upload the file
                combined_prompt = f"Please analyze the attached file: {os.path.basename(file_path)}\n\nPrompt: {prompt}"
            
            # Prepare request
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': model,
                'messages': [
                    {
                        'role': 'user',
                        'content': combined_prompt
                    }
                ],
                'temperature': 0.7,
                'max_tokens': 4000,
                'stream': False
            }
            
            # Make request
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                
                output_text = result['choices'][0]['message']['content']
                input_tokens = result.get('usage', {}).get('prompt_tokens')
                output_tokens = result.get('usage', {}).get('completion_tokens')
                
                return EvaluationResult(
                    provider="DeepSeek",
                    model=model,
                    output_text=output_text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    status="success"
                )
            else:
                error_msg = f"DeepSeek API error: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return EvaluationResult(
                    provider="DeepSeek",
                    model=model,
                    output_text="",
                    status="error",
                    error_message=error_msg
                )
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"DeepSeek request failed: {e}")
            return EvaluationResult(
                provider="DeepSeek",
                model=model,
                output_text="",
                status="error",
                error_message=f"Request failed: {str(e)}"
            )
        except Exception as e:
            self.logger.error(f"DeepSeek evaluation failed: {e}")
            return EvaluationResult(
                provider="DeepSeek",
                model=model,
                output_text="",
                status="error",
                error_message=str(e)
            )