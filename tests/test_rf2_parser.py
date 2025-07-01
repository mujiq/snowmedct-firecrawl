"""
Unit tests for RF2 parser module.
"""

import pytest
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd

from src.snomed_ct_platform.parsers.rf2_parser import (
    RF2Parser, ConceptRecord, DescriptionRecord, RelationshipRecord
)


class TestConceptRecord:
    """Test cases for ConceptRecord dataclass."""
    
    def test_concept_record_creation(self):
        """Test ConceptRecord creation with valid data."""
        record = ConceptRecord(
            id=12345,
            effective_time="20220131",
            active=True,
            module_id=900000000000207008,
            definition_status_id=900000000000074008
        )
        
        assert record.id == 12345
        assert record.effective_time == "20220131"
        assert record.active == True
        assert record.module_id == 900000000000207008
        assert record.definition_status_id == 900000000000074008
    
    def test_concept_record_string_representation(self):
        """Test ConceptRecord string representation."""
        record = ConceptRecord(
            id=12345,
            effective_time="20220131",
            active=True,
            module_id=900000000000207008,
            definition_status_id=900000000000074008
        )
        
        str_repr = str(record)
        assert "12345" in str_repr
        assert "20220131" in str_repr
        assert "True" in str_repr


class TestDescriptionRecord:
    """Test cases for DescriptionRecord dataclass."""
    
    def test_description_record_creation(self):
        """Test DescriptionRecord creation with valid data."""
        record = DescriptionRecord(
            id=12345001,
            effective_time="20220131",
            active=True,
            module_id=900000000000207008,
            concept_id=12345,
            language_code="en",
            type_id=900000000000013009,
            term="Heart disease",
            case_significance_id=900000000000448009
        )
        
        assert record.id == 12345001
        assert record.concept_id == 12345
        assert record.language_code == "en"
        assert record.term == "Heart disease"
        assert record.type_id == 900000000000013009
    
    def test_description_record_unicode_term(self):
        """Test DescriptionRecord with Unicode characters in term."""
        record = DescriptionRecord(
            id=12345001,
            effective_time="20220131",
            active=True,
            module_id=900000000000207008,
            concept_id=12345,
            language_code="en",
            type_id=900000000000013009,
            term="Corazón disease",  # Unicode character
            case_significance_id=900000000000448009
        )
        
        assert record.term == "Corazón disease"


class TestRelationshipRecord:
    """Test cases for RelationshipRecord dataclass."""
    
    def test_relationship_record_creation(self):
        """Test RelationshipRecord creation with valid data."""
        record = RelationshipRecord(
            id=12345001,
            effective_time="20220131",
            active=True,
            module_id=900000000000207008,
            source_id=12345,
            destination_id=404684003,
            relationship_group=0,
            type_id=116680003,
            characteristic_type_id=900000000000011006,
            modifier_id=900000000000451002
        )
        
        assert record.id == 12345001
        assert record.source_id == 12345
        assert record.destination_id == 404684003
        assert record.relationship_group == 0
        assert record.type_id == 116680003


class TestRF2Parser:
    """Test cases for RF2Parser class."""
    
    def test_parser_initialization(self, temp_dir):
        """Test RF2Parser initialization."""
        parser = RF2Parser(temp_dir)
        
        assert parser.data_dir == temp_dir
        assert parser.extracted_dir is None
        assert parser.CONCEPT_FILE_PATTERN == "*Concept_Snapshot_INT*.txt"
        assert parser.DESCRIPTION_FILE_PATTERN == "*Description_Snapshot-en_INT*.txt"
        assert parser.RELATIONSHIP_FILE_PATTERN == "*Relationship_Snapshot_INT*.txt"
    
    def test_extract_rf2_archive(self, temp_dir):
        """Test RF2 archive extraction."""
        # Create a test zip file
        zip_path = temp_dir / "test_rf2.zip"
        test_file_content = "test content"
        
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            zip_file.writestr("test_file.txt", test_file_content)
        
        parser = RF2Parser(temp_dir)
        
        # Extract the archive
        extracted_dir = parser.extract_rf2_archive(zip_path)
        
        # Verify extraction
        assert extracted_dir.exists()
        assert extracted_dir.is_dir()
        assert (extracted_dir / "test_file.txt").exists()
        assert (extracted_dir / "test_file.txt").read_text() == test_file_content
        assert parser.extracted_dir == extracted_dir
    
    def test_extract_rf2_archive_creates_directory(self, temp_dir):
        """Test that archive extraction creates necessary directories."""
        zip_path = temp_dir / "nested" / "test_rf2.zip"
        zip_path.parent.mkdir(parents=True)
        
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            zip_file.writestr("dummy.txt", "content")
        
        parser = RF2Parser(temp_dir)
        extracted_dir = parser.extract_rf2_archive(zip_path)
        
        # Should create extract directory structure
        expected_dir = temp_dir / "extracted" / "test_rf2"
        assert extracted_dir == expected_dir
        assert extracted_dir.exists()
    
    def test_find_rf2_files(self, sample_rf2_files, temp_dir):
        """Test finding RF2 files in directory."""
        parser = RF2Parser(temp_dir)
        found_files = parser.find_rf2_files(temp_dir)
        
        assert 'concepts' in found_files
        assert 'descriptions' in found_files
        assert 'relationships' in found_files
        
        assert found_files['concepts'] == sample_rf2_files['concepts']
        assert found_files['descriptions'] == sample_rf2_files['descriptions']
        assert found_files['relationships'] == sample_rf2_files['relationships']
    
    def test_find_rf2_files_missing_files(self, temp_dir):
        """Test finding RF2 files when some are missing."""
        # Create only concept file
        concept_file = temp_dir / "sct2_Concept_Snapshot_INT_20220131.txt"
        concept_file.write_text("dummy content")
        
        parser = RF2Parser(temp_dir)
        found_files = parser.find_rf2_files(temp_dir)
        
        assert 'concepts' in found_files
        assert 'descriptions' not in found_files
        assert 'relationships' not in found_files
    
    def test_find_rf2_files_recursive_search(self, temp_dir):
        """Test recursive search for RF2 files."""
        # Create files in nested directory
        nested_dir = temp_dir / "nested" / "deeper"
        nested_dir.mkdir(parents=True)
        
        concept_file = nested_dir / "sct2_Concept_Snapshot_INT_20220131.txt"
        concept_file.write_text("dummy content")
        
        parser = RF2Parser(temp_dir)
        found_files = parser.find_rf2_files(temp_dir)
        
        assert 'concepts' in found_files
        assert found_files['concepts'] == concept_file
    
    def test_parse_concepts(self, sample_rf2_files):
        """Test parsing concepts from RF2 file."""
        parser = RF2Parser(sample_rf2_files['concepts'].parent)
        
        concept_records = []
        for chunk in parser.parse_concepts(sample_rf2_files['concepts']):
            concept_records.extend(chunk)
        
        assert len(concept_records) == 3
        
        # Check first record
        first_record = concept_records[0]
        assert isinstance(first_record, ConceptRecord)
        assert first_record.id == 12345
        assert first_record.effective_time == "20220131"
        assert first_record.active == True
        assert first_record.module_id == 900000000000207008
        
        # Check inactive record
        inactive_record = concept_records[2]
        assert inactive_record.id == 11111
        assert inactive_record.active == False
    
    def test_parse_concepts_chunked(self, sample_rf2_files):
        """Test parsing concepts with small chunk size."""
        parser = RF2Parser(sample_rf2_files['concepts'].parent)
        
        chunks = list(parser.parse_concepts(sample_rf2_files['concepts'], chunk_size=1))
        
        # Should have 3 chunks with 1 record each (plus header handling)
        assert len(chunks) >= 3
        
        total_records = sum(len(chunk) for chunk in chunks)
        assert total_records == 3
    
    def test_parse_descriptions(self, sample_rf2_files):
        """Test parsing descriptions from RF2 file."""
        parser = RF2Parser(sample_rf2_files['descriptions'].parent)
        
        description_records = []
        for chunk in parser.parse_descriptions(sample_rf2_files['descriptions']):
            description_records.extend(chunk)
        
        assert len(description_records) == 3
        
        # Check first record
        first_record = description_records[0]
        assert isinstance(first_record, DescriptionRecord)
        assert first_record.id == 12345001
        assert first_record.concept_id == 12345
        assert first_record.language_code == "en"
        assert first_record.term == "Heart disease"
        assert first_record.type_id == 900000000000013009
    
    def test_parse_descriptions_multiple_terms_per_concept(self, sample_rf2_files):
        """Test parsing descriptions with multiple terms per concept."""
        parser = RF2Parser(sample_rf2_files['descriptions'].parent)
        
        description_records = []
        for chunk in parser.parse_descriptions(sample_rf2_files['descriptions']):
            description_records.extend(chunk)
        
        # Should have multiple descriptions for concept 12345
        concept_12345_descriptions = [
            record for record in description_records 
            if record.concept_id == 12345
        ]
        
        assert len(concept_12345_descriptions) == 2
        terms = [desc.term for desc in concept_12345_descriptions]
        assert "Heart disease" in terms
        assert "Cardiac disorder" in terms
    
    def test_parse_relationships(self, sample_rf2_files):
        """Test parsing relationships from RF2 file."""
        parser = RF2Parser(sample_rf2_files['relationships'].parent)
        
        relationship_records = []
        for chunk in parser.parse_relationships(sample_rf2_files['relationships']):
            relationship_records.extend(chunk)
        
        assert len(relationship_records) == 3
        
        # Check first record
        first_record = relationship_records[0]
        assert isinstance(first_record, RelationshipRecord)
        assert first_record.id == 12345001
        assert first_record.source_id == 12345
        assert first_record.destination_id == 404684003
        assert first_record.relationship_group == 0
        assert first_record.type_id == 116680003
    
    def test_get_file_stats(self, sample_rf2_files):
        """Test getting file statistics."""
        parser = RF2Parser(sample_rf2_files['concepts'].parent)
        
        stats = parser.get_file_stats(sample_rf2_files['concepts'])
        
        assert 'total_lines' in stats
        assert 'active_records' in stats
        assert 'inactive_records' in stats
        
        # Should have 4 total lines (3 data + 1 header)
        assert stats['total_lines'] == 4
        # Should have 2 active and 1 inactive record
        assert stats['active_records'] == 2
        assert stats['inactive_records'] == 1
    
    def test_get_file_stats_empty_file(self, temp_dir):
        """Test getting statistics for empty file."""
        empty_file = temp_dir / "empty.txt"
        empty_file.write_text("")
        
        parser = RF2Parser(temp_dir)
        stats = parser.get_file_stats(empty_file)
        
        assert stats['total_lines'] == 1  # Just header or empty
        assert stats['active_records'] == 0
        assert stats['inactive_records'] == 0


class TestRF2ParserIntegration:
    """Integration tests for RF2Parser."""
    
    def test_full_parsing_workflow(self, sample_rf2_files, temp_dir):
        """Test complete parsing workflow."""
        parser = RF2Parser(temp_dir)
        
        # Find files
        found_files = parser.find_rf2_files(temp_dir)
        assert len(found_files) == 3
        
        # Parse each file type
        concepts = []
        for chunk in parser.parse_concepts(found_files['concepts']):
            concepts.extend(chunk)
        
        descriptions = []
        for chunk in parser.parse_descriptions(found_files['descriptions']):
            descriptions.extend(chunk)
        
        relationships = []
        for chunk in parser.parse_relationships(found_files['relationships']):
            relationships.extend(chunk)
        
        # Verify data integrity
        assert len(concepts) == 3
        assert len(descriptions) == 3
        assert len(relationships) == 3
        
        # Verify concept-description relationships
        concept_ids = {concept.id for concept in concepts}
        description_concept_ids = {desc.concept_id for desc in descriptions}
        
        # All descriptions should reference existing concepts
        assert description_concept_ids.issubset(concept_ids)
    
    def test_parsing_with_malformed_data(self, temp_dir):
        """Test parsing behavior with malformed RF2 data."""
        # Create malformed concept file
        malformed_file = temp_dir / "sct2_Concept_Snapshot_INT_20220131.txt"
        malformed_content = """id	effectiveTime	active	moduleId	definitionStatusId
invalid_id	20220131	1	900000000000207008	900000000000074008
12345	invalid_date	1	900000000000207008	900000000000074008"""
        malformed_file.write_text(malformed_content)
        
        parser = RF2Parser(temp_dir)
        
        # Should handle parsing errors gracefully
        with pytest.raises((ValueError, TypeError)):
            concepts = []
            for chunk in parser.parse_concepts(malformed_file):
                concepts.extend(chunk)
    
    def test_large_file_chunking(self, temp_dir):
        """Test parsing behavior with large files using chunking."""
        # Create a larger concept file
        large_file = temp_dir / "sct2_Concept_Snapshot_INT_20220131.txt"
        
        header = "id\teffectiveTime\tactive\tmoduleId\tdefinitionStatusId\n"
        records = []
        for i in range(100):  # Create 100 records
            records.append(f"{i}\t20220131\t1\t900000000000207008\t900000000000074008\n")
        
        large_file.write_text(header + "".join(records))
        
        parser = RF2Parser(temp_dir)
        
        # Parse with small chunk size
        all_concepts = []
        chunk_count = 0
        for chunk in parser.parse_concepts(large_file, chunk_size=10):
            all_concepts.extend(chunk)
            chunk_count += 1
        
        # Should have processed all 100 records
        assert len(all_concepts) == 100
        # Should have used multiple chunks
        assert chunk_count >= 10
    
    def test_archive_extraction_and_parsing(self, temp_dir):
        """Test complete workflow from archive extraction to parsing."""
        # Create RF2 files
        concept_content = """id	effectiveTime	active	moduleId	definitionStatusId
12345	20220131	1	900000000000207008	900000000000074008"""
        
        desc_content = """id	effectiveTime	active	moduleId	conceptId	languageCode	typeId	term	caseSignificanceId
12345001	20220131	1	900000000000207008	12345	en	900000000000013009	Heart disease	900000000000448009"""
        
        # Create zip archive
        zip_path = temp_dir / "rf2_archive.zip"
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            zip_file.writestr("sct2_Concept_Snapshot_INT_20220131.txt", concept_content)
            zip_file.writestr("sct2_Description_Snapshot-en_INT_20220131.txt", desc_content)
        
        parser = RF2Parser(temp_dir)
        
        # Extract archive
        extracted_dir = parser.extract_rf2_archive(zip_path)
        
        # Find and parse files
        found_files = parser.find_rf2_files(extracted_dir)
        
        concepts = []
        for chunk in parser.parse_concepts(found_files['concepts']):
            concepts.extend(chunk)
        
        descriptions = []
        for chunk in parser.parse_descriptions(found_files['descriptions']):
            descriptions.extend(chunk)
        
        # Verify results
        assert len(concepts) == 1
        assert len(descriptions) == 1
        assert concepts[0].id == 12345
        assert descriptions[0].concept_id == 12345 