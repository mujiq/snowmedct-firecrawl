"""
SNOMED-CT RF2 (Release Format 2) Parser

This module provides functionality to parse SNOMED-CT RF2 files including
concepts, descriptions, and relationships.
"""

import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple
import pandas as pd
from dataclasses import dataclass
from datetime import datetime

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConceptRecord:
    """Represents a SNOMED-CT concept record."""
    id: int
    effective_time: str
    active: bool
    module_id: int
    definition_status_id: int


@dataclass
class DescriptionRecord:
    """Represents a SNOMED-CT description record."""
    id: int
    effective_time: str
    active: bool
    module_id: int
    concept_id: int
    language_code: str
    type_id: int
    term: str
    case_significance_id: int


@dataclass
class RelationshipRecord:
    """Represents a SNOMED-CT relationship record."""
    id: int
    effective_time: str
    active: bool
    module_id: int
    source_id: int
    destination_id: int
    relationship_group: int
    type_id: int
    characteristic_type_id: int
    modifier_id: int


class RF2Parser:
    """Parser for SNOMED-CT RF2 files."""
    
    # Standard RF2 file patterns
    CONCEPT_FILE_PATTERN = "*Concept_Snapshot_INT*.txt"
    DESCRIPTION_FILE_PATTERN = "*Description_Snapshot-en_INT*.txt"
    RELATIONSHIP_FILE_PATTERN = "*Relationship_Snapshot_INT*.txt"
    
    def __init__(self, data_dir: Path):
        """
        Initialize RF2 parser.
        
        Args:
            data_dir: Directory containing RF2 files or zip archives
        """
        self.data_dir = Path(data_dir)
        self.extracted_dir: Optional[Path] = None
        
    def extract_rf2_archive(self, archive_path: Path) -> Path:
        """
        Extract RF2 zip archive.
        
        Args:
            archive_path: Path to the RF2 zip file
            
        Returns:
            Path to extracted directory
        """
        logger.info(f"Extracting RF2 archive: {archive_path}")
        
        extract_dir = self.data_dir / "extracted" / archive_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        self.extracted_dir = extract_dir
        logger.info(f"Extracted to: {extract_dir}")
        
        return extract_dir
    
    def find_rf2_files(self, search_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Find RF2 files in the specified directory.
        
        Args:
            search_dir: Directory to search in (defaults to data_dir)
            
        Returns:
            Dictionary mapping file types to file paths
        """
        if search_dir is None:
            search_dir = self.extracted_dir or self.data_dir
        
        files = {}
        
        # Find concept file
        concept_files = list(search_dir.rglob(self.CONCEPT_FILE_PATTERN))
        if concept_files:
            files['concepts'] = concept_files[0]
            logger.info(f"Found concept file: {files['concepts']}")
        
        # Find description file
        description_files = list(search_dir.rglob(self.DESCRIPTION_FILE_PATTERN))
        if description_files:
            files['descriptions'] = description_files[0]
            logger.info(f"Found description file: {files['descriptions']}")
        
        # Find relationship file
        relationship_files = list(search_dir.rglob(self.RELATIONSHIP_FILE_PATTERN))
        if relationship_files:
            files['relationships'] = relationship_files[0]
            logger.info(f"Found relationship file: {files['relationships']}")
        
        return files
    
    def parse_concepts(self, file_path: Path, chunk_size: int = 10000) -> Iterator[List[ConceptRecord]]:
        """
        Parse concepts from RF2 concept file.
        
        Args:
            file_path: Path to concept file
            chunk_size: Number of records to process at once
            
        Yields:
            Lists of ConceptRecord objects
        """
        logger.info(f"Parsing concepts from: {file_path}")
        
        # Define column names for concept file
        columns = [
            'id', 'effectiveTime', 'active', 'moduleId', 'definitionStatusId'
        ]
        
        # Read file in chunks
        for chunk_df in pd.read_csv(
            file_path,
            sep='\t',
            dtype={
                'id': 'int64',
                'effectiveTime': 'str',
                'active': 'str',
                'moduleId': 'int64',
                'definitionStatusId': 'int64'
            },
            chunksize=chunk_size
        ):
            records = []
            for _, row in chunk_df.iterrows():
                record = ConceptRecord(
                    id=row['id'],
                    effective_time=row['effectiveTime'],
                    active=row['active'] == '1',
                    module_id=row['moduleId'],
                    definition_status_id=row['definitionStatusId']
                )
                records.append(record)
            
            logger.debug(f"Parsed {len(records)} concept records")
            yield records
    
    def parse_descriptions(self, file_path: Path, chunk_size: int = 10000) -> Iterator[List[DescriptionRecord]]:
        """
        Parse descriptions from RF2 description file.
        
        Args:
            file_path: Path to description file
            chunk_size: Number of records to process at once
            
        Yields:
            Lists of DescriptionRecord objects
        """
        logger.info(f"Parsing descriptions from: {file_path}")
        
        # Define column names for description file
        columns = [
            'id', 'effectiveTime', 'active', 'moduleId', 'conceptId',
            'languageCode', 'typeId', 'term', 'caseSignificanceId'
        ]
        
        # Read file in chunks
        for chunk_df in pd.read_csv(
            file_path,
            sep='\t',
            dtype={
                'id': 'int64',
                'effectiveTime': 'str',
                'active': 'str',
                'moduleId': 'int64',
                'conceptId': 'int64',
                'languageCode': 'str',
                'typeId': 'int64',
                'term': 'str',
                'caseSignificanceId': 'int64'
            },
            chunksize=chunk_size
        ):
            records = []
            for _, row in chunk_df.iterrows():
                record = DescriptionRecord(
                    id=row['id'],
                    effective_time=row['effectiveTime'],
                    active=row['active'] == '1',
                    module_id=row['moduleId'],
                    concept_id=row['conceptId'],
                    language_code=row['languageCode'],
                    type_id=row['typeId'],
                    term=row['term'],
                    case_significance_id=row['caseSignificanceId']
                )
                records.append(record)
            
            logger.debug(f"Parsed {len(records)} description records")
            yield records
    
    def parse_relationships(self, file_path: Path, chunk_size: int = 10000) -> Iterator[List[RelationshipRecord]]:
        """
        Parse relationships from RF2 relationship file.
        
        Args:
            file_path: Path to relationship file
            chunk_size: Number of records to process at once
            
        Yields:
            Lists of RelationshipRecord objects
        """
        logger.info(f"Parsing relationships from: {file_path}")
        
        # Read file in chunks
        for chunk_df in pd.read_csv(
            file_path,
            sep='\t',
            dtype={
                'id': 'int64',
                'effectiveTime': 'str',
                'active': 'str',
                'moduleId': 'int64',
                'sourceId': 'int64',
                'destinationId': 'int64',
                'relationshipGroup': 'int32',
                'typeId': 'int64',
                'characteristicTypeId': 'int64',
                'modifierId': 'int64'
            },
            chunksize=chunk_size
        ):
            records = []
            for _, row in chunk_df.iterrows():
                record = RelationshipRecord(
                    id=row['id'],
                    effective_time=row['effectiveTime'],
                    active=row['active'] == '1',
                    module_id=row['moduleId'],
                    source_id=row['sourceId'],
                    destination_id=row['destinationId'],
                    relationship_group=row['relationshipGroup'],
                    type_id=row['typeId'],
                    characteristic_type_id=row['characteristicTypeId'],
                    modifier_id=row['modifierId']
                )
                records.append(record)
            
            logger.debug(f"Parsed {len(records)} relationship records")
            yield records
    
    def get_file_stats(self, file_path: Path) -> Dict[str, int]:
        """
        Get basic statistics about an RF2 file.
        
        Args:
            file_path: Path to RF2 file
            
        Returns:
            Dictionary with file statistics
        """
        logger.info(f"Getting stats for: {file_path}")
        
        # Count total lines (excluding header)
        with open(file_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f) - 1  # Subtract header line
        
        # Read first chunk to get column info
        df_sample = pd.read_csv(file_path, sep='\t', nrows=1000)
        
        stats = {
            'total_records': total_lines,
            'columns': len(df_sample.columns),
            'column_names': list(df_sample.columns),
        }
        
        # Count active records if 'active' column exists
        if 'active' in df_sample.columns:
            active_count = 0
            for chunk_df in pd.read_csv(file_path, sep='\t', chunksize=10000):
                active_count += (chunk_df['active'] == '1').sum()
            stats['active_records'] = active_count
        
        logger.info(f"File stats: {stats}")
        return stats 