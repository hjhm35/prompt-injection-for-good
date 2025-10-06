#!/usr/bin/env python3
"""
Prompt Extraction Utility
Extracts prompts from existing generated documents and updates the database
"""

import os
import sys
import json
import hashlib
import re
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from database.operations import DatabaseManager
from database.models import GeneratedDocument

def calculate_prompt_hash(prompt_text: str) -> str:
    """Calculate SHA-256 hash of prompt text for deduplication"""
    if not prompt_text:
        return ""
    return hashlib.sha256(prompt_text.strip().encode('utf-8')).hexdigest()

def extract_prompt_from_docx(file_path: str) -> Optional[str]:
    """Extract prompt from DOCX file using steganographic header"""
    try:
        from docx import Document
        
        doc = Document(file_path)
        
        # Look for steganographic header in document XML
        xml_content = None
        try:
            # Try to extract XML content to look for embedded comments
            from docx.oxml import parse_xml
            # This is a simplified approach - in practice you'd need to examine the document structure
            pass
        except:
            pass
            
        # Look for prompt in document text (if it's visible)
        full_text = ""
        for paragraph in doc.paragraphs:
            full_text += paragraph.text + "\n"
        
        # Look for patterns that might indicate embedded prompts
        # Pattern 1: Look for JSON-like structures
        json_pattern = r'<!-- LLM_EVAL_HEADER:(.*?) -->'
        match = re.search(json_pattern, full_text)
        if match:
            try:
                header_data = json.loads(match.group(1))
                if 'prompt' in header_data:
                    return header_data['prompt']
            except json.JSONDecodeError:
                pass
        
        # Pattern 2: Look for common prompt patterns
        prompt_patterns = [
            r'Analyze this document.*?(?:\n|$)',
            r'Please analyze.*?(?:\n|$)',
            r'Evaluate.*?(?:\n|$)',
            r'Review.*?(?:\n|$)',
        ]
        
        for pattern in prompt_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                return match.group(0).strip()
                
        return None
        
    except Exception as e:
        print(f"Error extracting prompt from {file_path}: {e}")
        return None

def extract_prompt_from_text(file_path: str) -> Optional[str]:
    """Extract prompt from text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for steganographic header
        json_pattern = r'<!-- LLM_EVAL_HEADER:(.*?) -->'
        match = re.search(json_pattern, content)
        if match:
            try:
                header_data = json.loads(match.group(1))
                if 'prompt' in header_data:
                    return header_data['prompt']
            except json.JSONDecodeError:
                pass
        
        # Look for prompt patterns in first few lines
        lines = content.split('\n')[:10]  # Check first 10 lines
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['analyze', 'evaluate', 'review', 'assess']):
                if len(line) > 20 and len(line) < 200:  # Reasonable prompt length
                    return line
                    
        return None
        
    except Exception as e:
        print(f"Error extracting prompt from {file_path}: {e}")
        return None

def extract_prompt_from_file(file_path: str) -> Optional[str]:
    """Extract prompt from any supported file type"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        return None
        
    extension = file_path.suffix.lower()
    
    if extension == '.docx':
        return extract_prompt_from_docx(str(file_path))
    elif extension in ['.txt', '.md','.eml']:
        return extract_prompt_from_text(str(file_path))
    else:
        return None

def update_document_with_prompt(db_manager: DatabaseManager, document: GeneratedDocument, prompt_text: str) -> bool:
    """Update a document record with extracted prompt and hash"""
    try:
        prompt_hash = calculate_prompt_hash(prompt_text)
        
        session = db_manager.get_session()
        
        # Update the document
        doc = session.query(GeneratedDocument).filter(GeneratedDocument.id == document.id).first()
        if doc:
            doc.prompt_text = prompt_text
            doc.prompt_hash = prompt_hash
            session.commit()
            session.close()
            return True
        else:
            session.close()
            return False
            
    except Exception as e:
        print(f"Error updating document {document.id}: {e}")
        return False

def main():
    """Main entry point for prompt extraction utility"""
    print("🔍 Prompt Extraction Utility")
    print("=" * 50)
    
    # Initialize database
    db_manager = DatabaseManager()
    
    # Get all documents without prompt information
    session = db_manager.get_session()
    
    documents_without_prompts = session.query(GeneratedDocument).filter(
        (GeneratedDocument.prompt_text.is_(None)) | 
        (GeneratedDocument.prompt_text == '') |
        (GeneratedDocument.prompt_hash.is_(None))
    ).all()
    
    session.close()
    
    print(f"Found {len(documents_without_prompts)} documents without prompt information")
    
    if len(documents_without_prompts) == 0:
        print("✅ All documents already have prompt information!")
        return
    
    # Ask for confirmation
    response = input(f"\nDo you want to attempt prompt extraction for {len(documents_without_prompts)} documents? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Operation cancelled.")
        return
    
    successful_extractions = 0
    failed_extractions = 0
    
    for i, document in enumerate(documents_without_prompts):
        print(f"\nProcessing {i+1}/{len(documents_without_prompts)}: {document.file_name}")
        
        # Try to extract prompt from file
        extracted_prompt = extract_prompt_from_file(document.file_path)
        
        if extracted_prompt:
            print(f"  ✓ Extracted prompt: {extracted_prompt[:60]}{'...' if len(extracted_prompt) > 60 else ''}")
            
            # Update database
            if update_document_with_prompt(db_manager, document, extracted_prompt):
                print(f"  ✓ Updated database record")
                successful_extractions += 1
            else:
                print(f"  ✗ Failed to update database")
                failed_extractions += 1
        else:
            print(f"  ✗ Could not extract prompt from file")
            failed_extractions += 1
    
    print("\n" + "=" * 50)
    print("📊 Extraction Summary")
    print("=" * 50)
    print(f"Successful extractions: {successful_extractions}")
    print(f"Failed extractions: {failed_extractions}")
    print(f"Success rate: {successful_extractions / len(documents_without_prompts) * 100:.1f}%")
    
    if successful_extractions > 0:
        print(f"\n✅ Successfully extracted and stored prompts for {successful_extractions} documents!")
        
        # Show unique prompts found
        session = db_manager.get_session()
        unique_prompts = session.execute("""
            SELECT COUNT(DISTINCT prompt_hash) as unique_count
            FROM generated_documents 
            WHERE prompt_hash IS NOT NULL
        """).fetchone()
        session.close()
        
        print(f"📈 Total unique prompts in system: {unique_prompts[0]}")

if __name__ == "__main__":
    main()