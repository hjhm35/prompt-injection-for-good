from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Optional, Dict, Any
import os
import csv
import io
from datetime import datetime

from .models import Base, Evaluation, ReferenceOutput, EvaluationScore, EvaluationCriteria, GeneratedDocument

class DatabaseManager:
    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv('DATABASE_URL', 'sqlite:///llm_evaluations.db')
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def init_database(self):
        """Initialize database tables"""
        Base.metadata.create_all(bind=self.engine)
        
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    def insert_evaluation(self, evaluation_data: Dict[str, Any]) -> int:
        """Insert evaluation result into database"""
        session = self.get_session()
        try:
            evaluation = Evaluation(**evaluation_data)
            session.add(evaluation)
            session.commit()
            return evaluation.id
        except SQLAlchemyError as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def query_evaluations(self, 
                         run_number: Optional[int] = None,
                         file_name: Optional[str] = None,
                         provider: Optional[str] = None,
                         status: Optional[str] = None,
                         limit: Optional[int] = None) -> List[Evaluation]:
        """Query evaluations with filters"""
        session = self.get_session()
        try:
            query = session.query(Evaluation)
            
            if run_number:
                query = query.filter(Evaluation.evaluation_run_number == run_number)
            if file_name:
                query = query.filter(Evaluation.file_name.like(f"%{file_name}%"))
            if provider:
                query = query.filter(Evaluation.llm_provider == provider)
            if status:
                query = query.filter(Evaluation.status == status)
                
            query = query.order_by(desc(Evaluation.timestamp))
            
            if limit:
                query = query.limit(limit)
                
            return query.all()
        finally:
            session.close()
    
    def get_run_statistics(self, run_number: int) -> Dict[str, Any]:
        """Get statistics for a specific run"""
        session = self.get_session()
        try:
            evaluations = session.query(Evaluation).filter(
                Evaluation.evaluation_run_number == run_number
            ).all()
            
            if not evaluations:
                return {'run_number': run_number, 'total_evaluations': 0}
            
            total_count = len(evaluations)
            success_count = len([e for e in evaluations if e.status == 'success'])
            error_count = total_count - success_count
            
            response_times = [e.response_time_seconds for e in evaluations if e.response_time_seconds]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            total_input_tokens = sum(e.input_token_usage for e in evaluations if e.input_token_usage)
            total_output_tokens = sum(e.output_token_usage for e in evaluations if e.output_token_usage)
            
            providers = list(set(e.llm_provider for e in evaluations))
            models = list(set(e.model_version for e in evaluations))
            
            return {
                'run_number': run_number,
                'total_evaluations': total_count,
                'successful_evaluations': success_count,
                'failed_evaluations': error_count,
                'success_rate': success_count / total_count if total_count > 0 else 0,
                'average_response_time': avg_response_time,
                'total_input_tokens': total_input_tokens,
                'total_output_tokens': total_output_tokens,
                'providers_tested': providers,
                'models_tested': models,
                'start_time': min(e.timestamp for e in evaluations),
                'end_time': max(e.timestamp for e in evaluations)
            }
        finally:
            session.close()
    
    def export_to_dict(self, evaluations: List[Evaluation]) -> List[Dict[str, Any]]:
        """Convert evaluations to dictionary format for export"""
        return [e.to_dict() for e in evaluations]
    
    # Reference Output Operations
    def insert_reference_output(self, reference_data: Dict[str, Any]) -> int:
        """Insert reference output into database"""
        session = self.get_session()
        try:
            reference = ReferenceOutput(**reference_data)
            session.add(reference)
            session.commit()
            return reference.id
        except SQLAlchemyError as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def query_reference_outputs(self, 
                               file_name: Optional[str] = None,
                               prompt_text: Optional[str] = None,
                               is_active: bool = True) -> List[ReferenceOutput]:
        """Query reference outputs with filters"""
        session = self.get_session()
        try:
            query = session.query(ReferenceOutput)
            
            if file_name:
                query = query.filter(ReferenceOutput.file_name.like(f"%{file_name}%"))
            if prompt_text:
                query = query.filter(ReferenceOutput.prompt_text.like(f"%{prompt_text}%"))
            if is_active is not None:
                query = query.filter(ReferenceOutput.is_active == is_active)
                
            return query.order_by(ReferenceOutput.created_date.desc()).all()
        finally:
            session.close()
    
    def get_reference_output(self, reference_id: int) -> Optional[ReferenceOutput]:
        """Get specific reference output by ID"""
        session = self.get_session()
        try:
            return session.query(ReferenceOutput).filter(
                ReferenceOutput.id == reference_id
            ).first()
        finally:
            session.close()
    
    def update_reference_output(self, reference_id: int, updates: Dict[str, Any]) -> bool:
        """Update reference output"""
        session = self.get_session()
        try:
            reference = session.query(ReferenceOutput).filter(
                ReferenceOutput.id == reference_id
            ).first()
            
            if not reference:
                return False
            
            for key, value in updates.items():
                if hasattr(reference, key):
                    setattr(reference, key, value)
            
            session.commit()
            return True
        except SQLAlchemyError as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    # Evaluation Score Operations
    def insert_evaluation_score(self, score_data: Dict[str, Any]) -> int:
        """Insert evaluation score into database"""
        session = self.get_session()
        try:
            score = EvaluationScore(**score_data)
            session.add(score)
            session.commit()
            return score.id
        except SQLAlchemyError as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def query_evaluation_scores(self,
                               evaluation_id: Optional[int] = None,
                               reference_output_id: Optional[int] = None,
                               needs_review: Optional[bool] = None,
                               scoring_method: Optional[str] = None) -> List[EvaluationScore]:
        """Query evaluation scores with filters"""
        session = self.get_session()
        try:
            query = session.query(EvaluationScore)
            
            if evaluation_id:
                query = query.filter(EvaluationScore.evaluation_id == evaluation_id)
            if reference_output_id:
                query = query.filter(EvaluationScore.reference_output_id == reference_output_id)
            if needs_review is not None:
                query = query.filter(EvaluationScore.needs_review == needs_review)
            if scoring_method:
                query = query.filter(EvaluationScore.scoring_method == scoring_method)
                
            return query.order_by(EvaluationScore.created_date.desc()).all()
        finally:
            session.close()
    
    def update_evaluation_score(self, score_id: int, updates: Dict[str, Any]) -> bool:
        """Update evaluation score"""
        session = self.get_session()
        try:
            score = session.query(EvaluationScore).filter(
                EvaluationScore.id == score_id
            ).first()
            
            if not score:
                return False
            
            for key, value in updates.items():
                if hasattr(score, key):
                    setattr(score, key, value)
            
            session.commit()
            return True
        except SQLAlchemyError as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    # Evaluation Criteria Operations
    def insert_evaluation_criteria(self, criteria_data: Dict[str, Any]) -> int:
        """Insert evaluation criteria into database"""
        session = self.get_session()
        try:
            criteria = EvaluationCriteria(**criteria_data)
            session.add(criteria)
            session.commit()
            return criteria.id
        except SQLAlchemyError as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def query_evaluation_criteria(self, is_active: bool = True) -> List[EvaluationCriteria]:
        """Query evaluation criteria"""
        session = self.get_session()
        try:
            query = session.query(EvaluationCriteria)
            
            if is_active is not None:
                query = query.filter(EvaluationCriteria.is_active == is_active)
                
            return query.order_by(EvaluationCriteria.name).all()
        finally:
            session.close()
    
    def get_scores_for_run(self, run_number: int) -> List[Dict[str, Any]]:
        """Get all scores for evaluations in a specific run"""
        session = self.get_session()
        try:
            query = session.query(EvaluationScore).join(Evaluation).filter(
                Evaluation.evaluation_run_number == run_number
            )
            
            scores = query.all()
            return [score.to_dict() for score in scores]
        finally:
            session.close()
    
    def get_evaluations_needing_review(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get evaluations that need human review"""
        session = self.get_session()
        try:
            query = session.query(Evaluation).join(EvaluationScore).filter(
                EvaluationScore.needs_review == True
            )
            
            if limit:
                query = query.limit(limit)
                
            evaluations = query.all()
            return [eval.to_dict() for eval in evaluations]
        finally:
            session.close()
    
    def export_to_csv(self, evaluations: List[Evaluation]) -> str:
        """Export evaluations to CSV format as string"""
        if not evaluations:
            return ""
        
        # Convert evaluations to dictionary format
        data = [eval.to_dict() for eval in evaluations]
        
        # Create CSV string in memory
        output = io.StringIO()
        
        if data:
            fieldnames = data[0].keys()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in data:
                writer.writerow(row)
        
        csv_content = output.getvalue()
        output.close()
        
        return csv_content
    
    def export_evaluations_to_csv_file(self, evaluations: List[Evaluation], filename: str) -> str:
        """Export evaluations to CSV file"""
        csv_content = self.export_to_csv(evaluations)
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            csvfile.write(csv_content)
        
        return filename
    
    def export_reference_outputs_to_csv(self, reference_outputs: List[ReferenceOutput]) -> str:
        """Export reference outputs to CSV format as string"""
        if not reference_outputs:
            return ""
        
        # Convert to dictionary format
        data = [ref.to_dict() for ref in reference_outputs]
        
        # Create CSV string in memory
        output = io.StringIO()
        
        if data:
            # Define field order for reference outputs
            fieldnames = [
                'id', 'file_name', 'prompt_text', 'reference_output',
                'prompt_description', 'created_by', 'created_date', 'notes', 'is_active'
            ]
            # Only use fields that exist in the data
            fieldnames = [f for f in fieldnames if f in data[0]]
            
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in data:
                writer.writerow(row)
        
        csv_content = output.getvalue()
        output.close()
        
        return csv_content
    
    def export_reference_outputs_to_csv_file(self, reference_outputs: List[ReferenceOutput], filename: str) -> str:
        """Export reference outputs to CSV file"""
        csv_content = self.export_reference_outputs_to_csv(reference_outputs)
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            csvfile.write(csv_content)
        
        return filename
    
    # Generated Document Operations
    def insert_generated_document(self, document_data: Dict[str, Any]) -> int:
        """Insert generated document record into database"""
        session = self.get_session()
        try:
            document = GeneratedDocument(**document_data)
            session.add(document)
            session.commit()
            return document.id
        except SQLAlchemyError as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def query_generated_documents(self,
                                 generation_run_number: Optional[int] = None,
                                 format_type: Optional[str] = None,
                                 status: Optional[str] = None,
                                 created_by: Optional[str] = None,
                                 limit: Optional[int] = None) -> List[GeneratedDocument]:
        """Query generated documents with filters"""
        session = self.get_session()
        try:
            query = session.query(GeneratedDocument)
            
            if generation_run_number:
                query = query.filter(GeneratedDocument.generation_run_number == generation_run_number)
            if format_type:
                query = query.filter(GeneratedDocument.format_type == format_type)
            if status:
                query = query.filter(GeneratedDocument.status == status)
            if created_by:
                query = query.filter(GeneratedDocument.created_by == created_by)
                
            query = query.order_by(desc(GeneratedDocument.timestamp))
            
            if limit:
                query = query.limit(limit)
                
            return query.all()
        finally:
            session.close()
    
    def get_generation_statistics(self, generation_run_number: int) -> Dict[str, Any]:
        """Get statistics for a specific document generation run"""
        session = self.get_session()
        try:
            documents = session.query(GeneratedDocument).filter(
                GeneratedDocument.generation_run_number == generation_run_number
            ).all()
            
            if not documents:
                return {'generation_run_number': generation_run_number, 'total_documents': 0}
            
            total_count = len(documents)
            success_count = len([d for d in documents if d.status == 'success'])
            error_count = total_count - success_count
            
            total_file_size = sum(d.file_size_bytes for d in documents if d.file_size_bytes)
            
            format_types = list(set(d.format_type for d in documents))
            variants = list(set(d.variant_name for d in documents if d.variant_name))
            steganographic_count = len([d for d in documents if d.is_steganographic])
            
            return {
                'generation_run_number': generation_run_number,
                'total_documents': total_count,
                'successful_generations': success_count,
                'failed_generations': error_count,
                'success_rate': success_count / total_count if total_count > 0 else 0,
                'total_file_size_bytes': total_file_size,
                'format_types': format_types,
                'variants': variants,
                'steganographic_documents': steganographic_count,
                'start_time': min(d.timestamp for d in documents),
                'end_time': max(d.timestamp for d in documents)
            }
        finally:
            session.close()
    
    def export_generated_documents_to_csv(self, documents: List[GeneratedDocument]) -> str:
        """Export generated documents to CSV format as string"""
        if not documents:
            return ""
        
        # Convert to dictionary format
        data = [doc.to_dict() for doc in documents]
        
        # Create CSV string in memory
        output = io.StringIO()
        
        if data:
            fieldnames = data[0].keys()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in data:
                writer.writerow(row)
        
        csv_content = output.getvalue()
        output.close()
        
        return csv_content
    
    def export_generated_documents_to_csv_file(self, documents: List[GeneratedDocument], filename: str) -> str:
        """Export generated documents to CSV file"""
        csv_content = self.export_generated_documents_to_csv(documents)
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            csvfile.write(csv_content)
        
        return filename