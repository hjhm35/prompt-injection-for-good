import json
import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

class ConfigurationParser:
    """Parse and validate JSON configuration files for evaluations"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def parse_config_file(self, config_file_path: str) -> Dict[str, Any]:
        """
        Parse JSON configuration file
        
        Args:
            config_file_path: Path to the JSON configuration file
            
        Returns:
            Parsed and validated configuration
        """
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Configuration file not found: {config_file_path}")
        
        try:
            with open(config_file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        
        # Validate configuration
        validated_config = self._validate_config(config)
        
        self.logger.info(f"Loaded configuration from {config_file_path}")
        return validated_config
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration structure and values"""
        errors = []
        
        # Required fields
        if 'run_number' not in config:
            errors.append("Missing required field: run_number")
        elif not isinstance(config['run_number'], int):
            errors.append("run_number must be an integer")
        
        # Files configuration
        if 'files' not in config:
            errors.append("Missing required field: files")
        elif not isinstance(config['files'], list):
            errors.append("files must be a list")
        else:
            # Validate each file entry
            for i, file_entry in enumerate(config['files']):
                if isinstance(file_entry, str):
                    # Simple file path - convert to dict format
                    config['files'][i] = {
                        'path': file_entry,
                        'prompt': config.get('default_prompt', ''),
                        'prompt_description': None
                    }
                elif isinstance(file_entry, dict):
                    # Validate dict format
                    if 'path' not in file_entry:
                        errors.append(f"File entry {i}: missing 'path' field")
                    if 'prompt' not in file_entry and 'default_prompt' not in config:
                        errors.append(f"File entry {i}: missing 'prompt' field and no default_prompt")
                    
                    # Set defaults
                    if 'prompt' not in file_entry:
                        file_entry['prompt'] = config.get('default_prompt', '')
                    if 'prompt_description' not in file_entry:
                        file_entry['prompt_description'] = None
                else:
                    errors.append(f"File entry {i}: must be string or object")
        
        # Models configuration (optional)
        if 'models' in config:
            if not isinstance(config['models'], list):
                errors.append("models must be a list")
        
        # Providers configuration (optional)
        if 'providers' in config:
            if not isinstance(config['providers'], list):
                errors.append("providers must be a list")
        
        # Evaluation configuration (optional)
        if 'evaluation' in config:
            self._validate_evaluation_config(config['evaluation'], errors)
        
        if errors:
            raise ValueError(f"Configuration validation errors:\n" + "\n".join(f"  - {error}" for error in errors))
        
        return config
    
    def _validate_evaluation_config(self, eval_config: Dict[str, Any], errors: List[str]):
        """Validate evaluation-specific configuration"""
        if not isinstance(eval_config, dict):
            errors.append("evaluation must be an object")
            return
        
        # Enabled flag
        if 'enabled' in eval_config and not isinstance(eval_config['enabled'], bool):
            errors.append("evaluation.enabled must be a boolean")
        
        # Judge model
        if 'judge_model' in eval_config and not isinstance(eval_config['judge_model'], str):
            errors.append("evaluation.judge_model must be a string")
        
        # Criteria
        if 'criteria' in eval_config:
            if not isinstance(eval_config['criteria'], list):
                errors.append("evaluation.criteria must be a list")
            elif not all(isinstance(c, str) for c in eval_config['criteria']):
                errors.append("evaluation.criteria must be a list of strings")
        
        # Semantic similarity config
        if 'semantic_similarity' in eval_config:
            sem_config = eval_config['semantic_similarity']
            if not isinstance(sem_config, dict):
                errors.append("evaluation.semantic_similarity must be an object")
            else:
                if 'enabled' in sem_config and not isinstance(sem_config['enabled'], bool):
                    errors.append("evaluation.semantic_similarity.enabled must be a boolean")
                if 'threshold' in sem_config:
                    threshold = sem_config['threshold']
                    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                        errors.append("evaluation.semantic_similarity.threshold must be a number between 0 and 1")
        
        # Confidence threshold
        if 'confidence_threshold' in eval_config:
            threshold = eval_config['confidence_threshold']
            if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                errors.append("evaluation.confidence_threshold must be a number between 0 and 1")
    
    def create_example_config(self, output_path: str) -> None:
        """Create an example configuration file"""
        example_config = {
            "run_number": 1,
            "models": ["gpt-4", "claude-3-sonnet"],
            "providers": ["openai", "anthropic"],
            "default_prompt": "Summarize the key points in this document",
            "files": [
                {
                    "path": "document1.pdf",
                    "prompt": "Summarize this document in 3 bullet points",
                    "prompt_description": "Basic summarization test"
                },
                {
                    "path": "document2.docx",
                    "prompt": "Extract key insights and provide analysis",
                    "prompt_description": "Advanced analysis test"
                },
                {
                    "path": "report.txt",
                    "prompt_description": "Uses default_prompt"
                }
            ],
            "evaluation": {
                "enabled": True,
                "judge_model": "gpt-4",
                "criteria": ["accuracy", "relevance", "clarity", "completeness"],
                "semantic_similarity": {
                    "enabled": True,
                    "threshold": 0.8,
                    "model": "sentence-transformers/all-mpnet-base-v2"
                },
                "confidence_threshold": 0.7,
                "auto_flag_for_review": True
            },
            "export": {
                "csv_file": "results.csv",
                "json_file": "results.json"
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(example_config, f, indent=2)
        
        self.logger.info(f"Created example configuration at {output_path}")
    
    def validate_file_paths(self, config: Dict[str, Any]) -> List[str]:
        """Validate that all file paths in config exist"""
        missing_files = []
        
        for file_entry in config.get('files', []):
            file_path = file_entry['path']
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        return missing_files
    
    def resolve_relative_paths(self, config: Dict[str, Any], config_file_dir: str) -> Dict[str, Any]:
        """Resolve relative file paths based on config file location"""
        for file_entry in config.get('files', []):
            file_path = file_entry['path']
            
            # If path is relative, make it relative to config file directory
            if not os.path.isabs(file_path):
                resolved_path = os.path.join(config_file_dir, file_path)
                file_entry['path'] = os.path.abspath(resolved_path)
        
        return config
    
    def extract_evaluation_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters for the evaluation engine"""
        return {
            'files': [
                {
                    'file_path': entry['path'],
                    'description': entry['prompt_description']
                }
                for entry in config['files']
            ],
            'prompts': [entry['prompt'] for entry in config['files']],
            'run_number': config['run_number'],
            'providers': config.get('providers'),
            'models': config.get('models'),
            'evaluation_config': config.get('evaluation', {}),
            'export_config': config.get('export', {})
        }