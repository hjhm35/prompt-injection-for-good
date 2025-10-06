from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Evaluation(Base):
    __tablename__ = 'evaluations'
    
    id = Column(Integer, primary_key=True)
    evaluation_run_number = Column(Integer, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    file_name = Column(String(500), nullable=False)
    input_prompt_text = Column(Text, nullable=False)
    manual_prompt_description = Column(Text)
    llm_provider = Column(String(100), nullable=False, index=True)
    model_version = Column(String(100), nullable=False)
    output_text = Column(Text)
    input_token_usage = Column(Integer)
    output_token_usage = Column(Integer)
    response_time_seconds = Column(Float)
    status = Column(String(20), nullable=False, index=True)  # 'success' or 'error'
    error_message = Column(Text)
    
    # Relationships for advanced features
    scores = relationship("EvaluationScore", back_populates="evaluation", cascade="all, delete-orphan")
    
    def to_dict(self):
        """Convert to dictionary for easy export"""
        return {
            'id': self.id,
            'run_number': self.evaluation_run_number,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'file_name': self.file_name,
            'prompt': self.input_prompt_text,
            'prompt_description': self.manual_prompt_description,
            'provider': self.llm_provider,
            'model': self.model_version,
            'output': self.output_text,
            'input_tokens': self.input_token_usage,
            'output_tokens': self.output_token_usage,
            'response_time': self.response_time_seconds,
            'status': self.status,
            'error': self.error_message
        }

class ReferenceOutput(Base):
    __tablename__ = 'reference_outputs'
    
    id = Column(Integer, primary_key=True)
    file_name = Column(String(500), nullable=False, index=True)
    prompt_text = Column(Text, nullable=False)
    prompt_description = Column(Text)
    reference_output = Column(Text, nullable=False)
    created_by = Column(String(100))
    created_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    notes = Column(Text)
    
    # Relationships
    scores = relationship("EvaluationScore", back_populates="reference", cascade="all, delete-orphan")
    
    def to_dict(self):
        """Convert to dictionary for easy export"""
        return {
            'id': self.id,
            'file_name': self.file_name,
            'prompt_text': self.prompt_text,
            'prompt_description': self.prompt_description,
            'reference_output': self.reference_output,
            'created_by': self.created_by,
            'created_date': self.created_date.isoformat() if self.created_date else None,
            'is_active': self.is_active,
            'notes': self.notes
        }

class EvaluationScore(Base):
    __tablename__ = 'evaluation_scores'
    
    id = Column(Integer, primary_key=True)
    evaluation_id = Column(Integer, ForeignKey('evaluations.id'), nullable=False, index=True)
    reference_output_id = Column(Integer, ForeignKey('reference_outputs.id'), nullable=True, index=True)
    
    # Automated scoring
    judge_model = Column(String(100))
    automated_score = Column(Float)
    confidence_score = Column(Float)
    semantic_similarity_score = Column(Float)
    
    # Human review
    human_score = Column(Float)
    needs_review = Column(Boolean, default=False, nullable=False, index=True)
    reviewer_name = Column(String(100))
    reviewer_notes = Column(Text)
    review_date = Column(DateTime)
    
    # Scoring details
    scoring_criteria = Column(Text)  # JSON string of criteria scores
    scoring_method = Column(String(50))  # 'llm_judge', 'semantic_similarity', 'human', etc.
    created_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    evaluation = relationship("Evaluation", back_populates="scores")
    reference = relationship("ReferenceOutput", back_populates="scores")
    
    def to_dict(self):
        """Convert to dictionary for easy export"""
        return {
            'id': self.id,
            'evaluation_id': self.evaluation_id,
            'reference_output_id': self.reference_output_id,
            'judge_model': self.judge_model,
            'automated_score': self.automated_score,
            'confidence_score': self.confidence_score,
            'semantic_similarity_score': self.semantic_similarity_score,
            'human_score': self.human_score,
            'needs_review': self.needs_review,
            'reviewer_name': self.reviewer_name,
            'reviewer_notes': self.reviewer_notes,
            'review_date': self.review_date.isoformat() if self.review_date else None,
            'scoring_criteria': self.scoring_criteria,
            'scoring_method': self.scoring_method,
            'created_date': self.created_date.isoformat() if self.created_date else None
        }

class EvaluationCriteria(Base):
    __tablename__ = 'evaluation_criteria'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    scoring_scale = Column(String(100))  # e.g., "1-5", "0-100", "pass/fail"
    weight = Column(Float, default=1.0)
    category = Column(String(50))  # e.g., "accuracy", "relevance", "clarity"
    is_active = Column(Boolean, default=True, nullable=False)
    created_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def to_dict(self):
        """Convert to dictionary for easy export"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'scoring_scale': self.scoring_scale,
            'weight': self.weight,
            'category': self.category,
            'is_active': self.is_active,
            'created_date': self.created_date.isoformat() if self.created_date else None
        }

class GeneratedDocument(Base):
    __tablename__ = 'generated_documents'
    
    id = Column(Integer, primary_key=True)
    generation_run_number = Column(Integer, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    file_name = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=False)
    prompt_text = Column(Text, nullable=False)
    prompt_description = Column(Text)
    prompt_hash = Column(String(64), index=True)  # SHA-256 hash for prompt deduplication
    content_body = Column(Text)
    format_type = Column(String(50), nullable=False)  # 'docx', 'pdf', 'email', 'txt'
    variant_name = Column(String(100))
    is_steganographic = Column(Boolean, default=False, nullable=False)
    validation_passed = Column(Boolean, default=False, nullable=False)
    file_size_bytes = Column(Integer)
    generation_method = Column(String(100))  # 'simple', 'combinatorial', 'multi_injection'
    status = Column(String(20), nullable=False, index=True, default='success')  # 'success' or 'error'
    error_message = Column(Text)
    document_metadata = Column(Text)  # JSON string for additional metadata
    created_by = Column(String(100), default='system')
    
    def to_dict(self):
        """Convert to dictionary for easy export"""
        return {
            'id': self.id,
            'generation_run_number': self.generation_run_number,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'file_name': self.file_name,
            'file_path': self.file_path,
            'prompt_text': self.prompt_text,
            'prompt_description': self.prompt_description,
            'prompt_hash': self.prompt_hash,
            'content_body': self.content_body,
            'format_type': self.format_type,
            'variant_name': self.variant_name,
            'is_steganographic': self.is_steganographic,
            'validation_passed': self.validation_passed,
            'file_size_bytes': self.file_size_bytes,
            'generation_method': self.generation_method,
            'status': self.status,
            'error_message': self.error_message,
            'document_metadata': self.document_metadata,
            'created_by': self.created_by
        }