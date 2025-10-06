#!/usr/bin/env python3
"""
Database migration script to add prompt_hash field to GeneratedDocument table
"""

import hashlib
import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from database.models import Base, GeneratedDocument

def calculate_prompt_hash(prompt_text):
    """Calculate SHA-256 hash of prompt text for deduplication"""
    if not prompt_text:
        return None
    return hashlib.sha256(prompt_text.strip().encode('utf-8')).hexdigest()

def main():
    # Database setup
    db_path = os.path.join(os.path.dirname(__file__), 'llm_evaluations.db')
    engine = create_engine(f'sqlite:///{db_path}')
    
    print("Starting migration to add prompt_hash field...")
    
    try:
        # Check if column already exists
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(generated_documents);"))
            columns = [row[1] for row in result]
            
            if 'prompt_hash' in columns:
                print("prompt_hash column already exists!")
                return
            
            # Add the new column
            print("Adding prompt_hash column to generated_documents table...")
            conn.execute(text("ALTER TABLE generated_documents ADD COLUMN prompt_hash VARCHAR(64);"))
            conn.commit()
            
            # Create index on prompt_hash for performance
            print("Creating index on prompt_hash...")
            conn.execute(text("CREATE INDEX idx_generated_documents_prompt_hash ON generated_documents(prompt_hash);"))
            conn.commit()
            
        # Update existing records with prompt hashes
        Session = sessionmaker(bind=engine)
        session = Session()
        
        print("Calculating prompt hashes for existing documents...")
        documents = session.query(GeneratedDocument).filter(GeneratedDocument.prompt_text.isnot(None)).all()
        
        updated_count = 0
        for doc in documents:
            if doc.prompt_text:
                doc.prompt_hash = calculate_prompt_hash(doc.prompt_text)
                updated_count += 1
                
                if updated_count % 100 == 0:
                    print(f"Processed {updated_count} documents...")
        
        session.commit()
        session.close()
        
        print(f"Migration completed successfully! Updated {updated_count} documents with prompt hashes.")
        
        # Show some statistics
        session = Session()
        total_docs = session.query(GeneratedDocument).count()
        docs_with_hashes = session.query(GeneratedDocument).filter(GeneratedDocument.prompt_hash.isnot(None)).count()
        unique_prompts = session.query(GeneratedDocument.prompt_hash).distinct().filter(GeneratedDocument.prompt_hash.isnot(None)).count()
        
        print(f"\nStatistics:")
        print(f"Total documents: {total_docs}")
        print(f"Documents with prompt hashes: {docs_with_hashes}")
        print(f"Unique prompts: {unique_prompts}")
        
        session.close()
        
    except Exception as e:
        print(f"Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()