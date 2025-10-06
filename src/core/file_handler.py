import os
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

class FileHandler:
    """Handle file validation and processing"""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf', '.docx', '.doc', '.txt', '.md', 
        '.jpg', '.jpeg', '.png', '.gif', '.webp', '.eml'
    }
    
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB default
    
    def __init__(self, max_file_size: int = None):
        self.max_file_size = max_file_size or self.MAX_FILE_SIZE
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_file(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a single file
        
        Returns:
            (is_valid, error_message)
        """
        if not os.path.exists(file_path):
            return False, f"File does not exist: {file_path}"
        
        if not os.path.isfile(file_path):
            return False, f"Path is not a file: {file_path}"
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size:
            return False, f"File too large: {file_size} bytes (max: {self.max_file_size})"
        
        if file_size == 0:
            return False, f"File is empty: {file_path}"
        
        # Check file extension
        file_extension = Path(file_path).suffix.lower()
        if file_extension not in self.SUPPORTED_EXTENSIONS:
            return False, f"Unsupported file type: {file_extension}. Supported: {self.SUPPORTED_EXTENSIONS}"
        
        # Check if file is readable
        try:
            with open(file_path, 'rb') as f:
                f.read(1024)  # Try to read first 1KB
        except Exception as e:
            return False, f"Cannot read file: {e}"
        
        return True, None
    
    def validate_files(self, file_paths: List[str]) -> Dict[str, Optional[str]]:
        """
        Validate multiple files
        
        Returns:
            Dict mapping file_path to error_message (None if valid)
        """
        results = {}
        for file_path in file_paths:
            is_valid, error = self.validate_file(file_path)
            results[file_path] = None if is_valid else error
        
        return results
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get detailed file information"""
        if not os.path.exists(file_path):
            return {}
        
        stat = os.stat(file_path)
        mime_type, encoding = mimetypes.guess_type(file_path)
        
        return {
            'path': file_path,
            'name': os.path.basename(file_path),
            'size': stat.st_size,
            'extension': Path(file_path).suffix.lower(),
            'mime_type': mime_type,
            'encoding': encoding,
            'modified_time': stat.st_mtime,
            'is_readable': os.access(file_path, os.R_OK)
        }
    
    def process_file_configs(self, file_configs: List[str]) -> List[Dict[str, str]]:
        """
        Process file configurations in format "file_path:description"
        
        Args:
            file_configs: List of strings like "file1.pdf:Description 1"
        
        Returns:
            List of dicts with 'file_path' and 'description' keys
        """
        processed = []
        for config in file_configs:
            if ':' in config:
                file_path, description = config.split(':', 1)
                processed.append({
                    'file_path': file_path.strip(),
                    'description': description.strip()
                })
            else:
                processed.append({
                    'file_path': config.strip(),
                    'description': None
                })
        
        return processed