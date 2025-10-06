#!/usr/bin/env python3
"""
Bulk Prompt Assignment Utility
Allows manual assignment of prompts to documents that don't have them
"""

import os
import sys
import hashlib
from typing import List, Dict, Any

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from database.operations import DatabaseManager
from database.models import GeneratedDocument

def calculate_prompt_hash(prompt_text: str) -> str:
    """Calculate SHA-256 hash of prompt text for deduplication"""
    if not prompt_text:
        return ""
    return hashlib.sha256(prompt_text.strip().encode('utf-8')).hexdigest()

def show_documents_without_prompts(db_manager: DatabaseManager) -> List[GeneratedDocument]:
    """Show all documents without prompt information"""
    session = db_manager.get_session()
    
    documents = session.query(GeneratedDocument).filter(
        (GeneratedDocument.prompt_text.is_(None)) | 
        (GeneratedDocument.prompt_text == '') |
        (GeneratedDocument.prompt_hash.is_(None))
    ).order_by(GeneratedDocument.timestamp.desc()).all()
    
    session.close()
    
    print("\n📋 Documents without prompt information:")
    print("-" * 80)
    print(f"{'ID':<5} {'File Name':<30} {'Format':<8} {'Generated':<20}")
    print("-" * 80)
    
    for doc in documents:
        timestamp = doc.timestamp.strftime("%Y-%m-%d %H:%M") if doc.timestamp else "Unknown"
        print(f"{doc.id:<5} {doc.file_name[:29]:<30} {doc.format_type:<8} {timestamp:<20}")
    
    return documents

def assign_prompt_to_documents(db_manager: DatabaseManager, document_ids: List[int], prompt_text: str) -> int:
    """Assign a prompt to multiple documents"""
    prompt_hash = calculate_prompt_hash(prompt_text)
    
    session = db_manager.get_session()
    updated_count = 0
    
    try:
        for doc_id in document_ids:
            doc = session.query(GeneratedDocument).filter(GeneratedDocument.id == doc_id).first()
            if doc:
                doc.prompt_text = prompt_text
                doc.prompt_hash = prompt_hash
                updated_count += 1
        
        session.commit()
        
    except Exception as e:
        session.rollback()
        print(f"Error updating documents: {e}")
        updated_count = 0
    finally:
        session.close()
    
    return updated_count

def show_existing_prompts(db_manager: DatabaseManager) -> List[Dict[str, Any]]:
    """Show existing prompts for reuse"""
    session = db_manager.get_session()
    
    prompts = session.execute("""
        SELECT 
            prompt_text,
            prompt_hash,
            COUNT(*) as usage_count
        FROM generated_documents 
        WHERE prompt_text IS NOT NULL AND prompt_text != ''
        GROUP BY prompt_hash
        ORDER BY usage_count DESC
    """).fetchall()
    
    session.close()
    
    if prompts:
        print("\n💡 Existing prompts (you can reuse these):")
        print("-" * 80)
        for i, prompt in enumerate(prompts, 1):
            preview = prompt[0][:60] + "..." if len(prompt[0]) > 60 else prompt[0]
            print(f"{i}. [{prompt[2]} uses] {preview}")
        print("-" * 80)
    
    return prompts

def main():
    """Main interactive prompt assignment"""
    print("✏️  Bulk Prompt Assignment Utility")
    print("=" * 50)
    
    db_manager = DatabaseManager()
    
    while True:
        # Show documents without prompts
        documents = show_documents_without_prompts(db_manager)
        
        if not documents:
            print("\n✅ All documents have prompt information!")
            break
        
        print(f"\nFound {len(documents)} documents without prompts")
        
        # Show existing prompts for reference
        existing_prompts = show_existing_prompts(db_manager)
        
        print("\nOptions:")
        print("1. Assign prompt to specific documents")
        print("2. Assign prompt to all documents")
        print("3. Use existing prompt")
        print("4. Exit")
        
        choice = input("\nChoose an option (1-4): ").strip()
        
        if choice == "4":
            break
        elif choice == "1":
            # Assign to specific documents
            doc_ids_input = input("Enter document IDs (comma-separated): ").strip()
            try:
                doc_ids = [int(id.strip()) for id in doc_ids_input.split(",")]
                
                # Validate IDs
                valid_ids = [doc.id for doc in documents]
                invalid_ids = [id for id in doc_ids if id not in valid_ids]
                if invalid_ids:
                    print(f"Invalid document IDs: {invalid_ids}")
                    continue
                
                prompt_text = input("Enter the prompt text: ").strip()
                if not prompt_text:
                    print("Prompt cannot be empty!")
                    continue
                
                updated = assign_prompt_to_documents(db_manager, doc_ids, prompt_text)
                print(f"✅ Updated {updated} documents with the prompt")
                
            except ValueError:
                print("Invalid input! Please enter numeric IDs separated by commas.")
                
        elif choice == "2":
            # Assign to all documents
            prompt_text = input("Enter the prompt text for ALL documents: ").strip()
            if not prompt_text:
                print("Prompt cannot be empty!")
                continue
            
            confirm = input(f"Assign this prompt to ALL {len(documents)} documents? [y/N]: ")
            if confirm.lower() in ['y', 'yes']:
                doc_ids = [doc.id for doc in documents]
                updated = assign_prompt_to_documents(db_manager, doc_ids, prompt_text)
                print(f"✅ Updated {updated} documents with the prompt")
                
        elif choice == "3":
            # Use existing prompt
            if not existing_prompts:
                print("No existing prompts found!")
                continue
                
            try:
                choice_num = int(input("Choose a prompt number: "))
                if 1 <= choice_num <= len(existing_prompts):
                    selected_prompt = existing_prompts[choice_num - 1][0]
                    print(f"\nSelected prompt: {selected_prompt}")
                    
                    doc_ids_input = input("Enter document IDs (comma-separated) or 'all' for all: ").strip()
                    
                    if doc_ids_input.lower() == 'all':
                        doc_ids = [doc.id for doc in documents]
                    else:
                        doc_ids = [int(id.strip()) for id in doc_ids_input.split(",")]
                    
                    updated = assign_prompt_to_documents(db_manager, doc_ids, selected_prompt)
                    print(f"✅ Updated {updated} documents with the existing prompt")
                else:
                    print("Invalid prompt number!")
            except ValueError:
                print("Invalid input!")
        else:
            print("Invalid choice!")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    main()