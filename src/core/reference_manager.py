import os
import csv
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

from ..database.operations import DatabaseManager
from ..database.models import ReferenceOutput

class ReferenceManager:
    """Manage reference outputs for evaluation scoring"""
    
    def __init__(self, database_manager: DatabaseManager = None):
        self.db = database_manager or DatabaseManager()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize database
        self.db.init_database()
    
    def add_reference_output(self, 
                            file_name: str,
                            prompt_text: str,
                            reference_output: str,
                            prompt_description: Optional[str] = None,
                            created_by: Optional[str] = None,
                            notes: Optional[str] = None) -> int:
        """
        Add a new reference output
        
        Args:
            file_name: Name of the file this reference applies to
            prompt_text: The prompt text
            reference_output: The ideal/expected output
            prompt_description: Description of what the prompt tests
            created_by: Who created this reference
            notes: Additional notes
            
        Returns:
            ID of the created reference output
        """
        reference_data = {
            'file_name': file_name,
            'prompt_text': prompt_text,
            'prompt_description': prompt_description,
            'reference_output': reference_output,
            'created_by': created_by or os.getenv('USER', 'unknown'),
            'notes': notes
        }
        
        reference_id = self.db.insert_reference_output(reference_data)
        self.logger.info(f"Added reference output {reference_id} for file {file_name}")
        return reference_id
    
    def import_from_csv(self, csv_file_path: str) -> Tuple[int, List[str]]:
        """
        Import reference outputs from CSV file
        
        CSV format: file_name,prompt_text,reference_output,prompt_description,notes
        
        Returns:
            (number_imported, list_of_errors)
        """
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
        
        imported_count = 0
        errors = []
        
        try:
            with open(csv_file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                required_fields = ['file_name', 'prompt_text', 'reference_output']
                
                for row_num, row in enumerate(reader, start=2):  # Start at 2 for header
                    try:
                        # Validate required fields
                        missing_fields = [field for field in required_fields if not row.get(field)]
                        if missing_fields:
                            errors.append(f"Row {row_num}: Missing required fields: {missing_fields}")
                            continue
                        
                        # Add reference output
                        self.add_reference_output(
                            file_name=row['file_name'].strip(),
                            prompt_text=row['prompt_text'].strip(),
                            reference_output=row['reference_output'].strip(),
                            prompt_description=row.get('prompt_description', '').strip() or None,
                            notes=row.get('notes', '').strip() or None
                        )
                        
                        imported_count += 1
                        
                    except Exception as e:
                        errors.append(f"Row {row_num}: {str(e)}")
            
            self.logger.info(f"Imported {imported_count} reference outputs from {csv_file_path}")
            
        except Exception as e:
            errors.append(f"Failed to read CSV file: {str(e)}")
        
        return imported_count, errors
    
    def import_from_json(self, json_file_path: str) -> Tuple[int, List[str]]:
        """
        Import reference outputs from JSON file
        
        JSON format: [
            {
                "file_name": "doc.pdf",
                "prompt_text": "Summarize this",
                "reference_output": "Expected summary...",
                "prompt_description": "Basic summarization",
                "notes": "Optional notes"
            }
        ]
        
        Returns:
            (number_imported, list_of_errors)
        """
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"JSON file not found: {json_file_path}")
        
        imported_count = 0
        errors = []
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            if not isinstance(data, list):
                errors.append("JSON file must contain an array of reference outputs")
                return 0, errors
            
            required_fields = ['file_name', 'prompt_text', 'reference_output']
            
            for item_num, item in enumerate(data):
                try:
                    # Validate required fields
                    missing_fields = [field for field in required_fields if not item.get(field)]
                    if missing_fields:
                        errors.append(f"Item {item_num + 1}: Missing required fields: {missing_fields}")
                        continue
                    
                    # Add reference output
                    self.add_reference_output(
                        file_name=item['file_name'],
                        prompt_text=item['prompt_text'],
                        reference_output=item['reference_output'],
                        prompt_description=item.get('prompt_description'),
                        notes=item.get('notes')
                    )
                    
                    imported_count += 1
                    
                except Exception as e:
                    errors.append(f"Item {item_num + 1}: {str(e)}")
            
            self.logger.info(f"Imported {imported_count} reference outputs from {json_file_path}")
            
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            errors.append(f"Failed to read JSON file: {str(e)}")
        
        return imported_count, errors
    
    def list_reference_outputs(self, 
                              file_name: Optional[str] = None,
                              prompt_text: Optional[str] = None,
                              is_active: bool = True) -> List[Dict[str, Any]]:
        """List reference outputs with optional filtering"""
        references = self.db.query_reference_outputs(
            file_name=file_name,
            prompt_text=prompt_text,
            is_active=is_active
        )
        
        return [ref.to_dict() for ref in references]
    
    def get_reference_output(self, reference_id: int) -> Optional[Dict[str, Any]]:
        """Get specific reference output by ID"""
        reference = self.db.get_reference_output(reference_id)
        return reference.to_dict() if reference else None
    
    def update_reference_output(self, 
                               reference_id: int,
                               updates: Dict[str, Any]) -> bool:
        """Update reference output"""
        return self.db.update_reference_output(reference_id, updates)
    
    def deactivate_reference_output(self, reference_id: int) -> bool:
        """Deactivate (soft delete) reference output"""
        return self.update_reference_output(reference_id, {'is_active': False})
    
    def find_matching_references(self, 
                                file_name: str,
                                prompt_text: str) -> List[Dict[str, Any]]:
        """
        Find reference outputs that match the given file and prompt
        
        Uses exact file name match and fuzzy prompt matching
        """
        # First try exact matches
        exact_matches = self.db.query_reference_outputs(
            file_name=file_name,
            prompt_text=prompt_text,
            is_active=True
        )
        
        if exact_matches:
            return [ref.to_dict() for ref in exact_matches]
        
        # Try file name only match
        file_matches = self.db.query_reference_outputs(
            file_name=file_name,
            is_active=True
        )
        
        # Filter by similar prompts (simple keyword matching)
        prompt_keywords = set(prompt_text.lower().split())
        similar_matches = []
        
        for ref in file_matches:
            ref_keywords = set(ref.prompt_text.lower().split())
            # Calculate simple similarity (intersection over union)
            similarity = len(prompt_keywords & ref_keywords) / len(prompt_keywords | ref_keywords)
            
            if similarity > 0.5:  # 50% similarity threshold
                ref_dict = ref.to_dict()
                ref_dict['similarity_score'] = similarity
                similar_matches.append(ref_dict)
        
        # Sort by similarity score
        similar_matches.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        return similar_matches
    
    def export_to_csv(self, output_file: str, is_active: bool = True) -> int:
        """Export reference outputs to CSV file using centralized database export"""
        # Query reference outputs from database
        reference_objects = self.db.query_reference_outputs(is_active=is_active)
        
        if not reference_objects:
            return 0
        
        # Use centralized database export method
        self.db.export_reference_outputs_to_csv_file(reference_objects, output_file)
        
        self.logger.info(f"Exported {len(reference_objects)} reference outputs to {output_file}")
        return len(reference_objects)
    
    def export_to_json(self, output_file: str, is_active: bool = True) -> int:
        """Export reference outputs to JSON file"""
        references = self.list_reference_outputs(is_active=is_active)
        
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(references, file, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Exported {len(references)} reference outputs to {output_file}")
        return len(references)
    
    def validate_reference_output(self, reference_data: Dict[str, Any]) -> List[str]:
        """Validate reference output data and return list of errors"""
        errors = []
        
        required_fields = ['file_name', 'prompt_text', 'reference_output']
        for field in required_fields:
            if not reference_data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Validate file_name
        file_name = reference_data.get('file_name', '')
        if file_name and len(file_name) > 500:
            errors.append("File name too long (max 500 characters)")
        
        # Validate prompt_text
        prompt_text = reference_data.get('prompt_text', '')
        if prompt_text and len(prompt_text) > 10000:
            errors.append("Prompt text too long (max 10000 characters)")
        
        # Validate reference_output
        reference_output = reference_data.get('reference_output', '')
        if reference_output and len(reference_output) > 50000:
            errors.append("Reference output too long (max 50000 characters)")
        
        return errors
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about reference outputs"""
        all_references = self.list_reference_outputs(is_active=None)
        active_references = [ref for ref in all_references if ref['is_active']]
        
        # Group by file name
        by_file = {}
        for ref in active_references:
            file_name = ref['file_name']
            if file_name not in by_file:
                by_file[file_name] = 0
            by_file[file_name] += 1
        
        # Group by creator
        by_creator = {}
        for ref in active_references:
            creator = ref['created_by'] or 'unknown'
            if creator not in by_creator:
                by_creator[creator] = 0
            by_creator[creator] += 1
        
        return {
            'total_references': len(all_references),
            'active_references': len(active_references),
            'inactive_references': len(all_references) - len(active_references),
            'files_with_references': len(by_file),
            'references_by_file': dict(sorted(by_file.items(), key=lambda x: x[1], reverse=True)[:10]),
            'references_by_creator': by_creator,
            'latest_reference_date': max([ref['created_date'] for ref in all_references]) if all_references else None
        }