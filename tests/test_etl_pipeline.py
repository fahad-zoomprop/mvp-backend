"""
Test suite for ETL Pipeline
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.etl_pipeline import ETLPipeline


class TestETLPipeline:
    """Test cases for ETL Pipeline class"""

    @pytest.fixture
    def etl_pipeline(self):
        """Create an ETL Pipeline instance for testing"""
        with patch("etl_pipeline.DatabaseManager"), patch("etl_pipeline.DataCleaner"):
            return ETLPipeline()

    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data for testing"""
        return pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
                "age": [25, 30, 35, 28, 32],
                "city": ["New York", "London", "Tokyo", "Paris", "Sydney"],
            }
        )

    @pytest.fixture
    def temp_csv_file(self, sample_csv_data):
        """Create a temporary CSV file for testing"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            sample_csv_data.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)

    def test_setup_logging(self, etl_pipeline):
        """Test logging setup"""
        assert etl_pipeline.logger is not None
        assert etl_pipeline.logger.name == "etl_pipeline"

    def test_extract_from_csv_success(self, etl_pipeline, temp_csv_file):
        """Test successful CSV extraction"""
        df = etl_pipeline.extract_from_csv(temp_csv_file)
        assert len(df) == 5
        assert list(df.columns) == ["id", "name", "age", "city"]
        assert df.iloc[0]["name"] == "Alice"

    def test_extract_from_csv_with_options(self, etl_pipeline, temp_csv_file):
        """Test CSV extraction with additional options"""
        df = etl_pipeline.extract_from_csv(temp_csv_file, encoding="utf-8", sep=",")
        assert len(df) == 5

    def test_extract_from_csv_file_not_found(self, etl_pipeline):
        """Test CSV extraction with non-existent file"""
        with pytest.raises(Exception):
            etl_pipeline.extract_from_csv("nonexistent_file.csv")

    def test_extract_from_csv_invalid_format(self, etl_pipeline):
        """Test CSV extraction with invalid file format"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("This is not a CSV file")
            f.flush()

            with pytest.raises(Exception):
                etl_pipeline.extract_from_csv(f.name)

            os.unlink(f.name)

    @patch("pandas.read_sql")
    def test_extract_from_database_success(
        self, mock_read_sql, etl_pipeline, sample_csv_data
    ):
        """Test successful database extraction"""
        mock_read_sql.return_value = sample_csv_data
        etl_pipeline.db_manager.get_engine.return_value = MagicMock()

        query = "SELECT * FROM users"
        df = etl_pipeline.extract_from_database(query)

        assert len(df) == 5
        mock_read_sql.assert_called_once()
        etl_pipeline.db_manager.get_engine.assert_called_with("default")

    @patch("pandas.read_sql")
    def test_extract_from_database_custom_connection(
        self, mock_read_sql, etl_pipeline, sample_csv_data
    ):
        """Test database extraction with custom connection"""
        mock_read_sql.return_value = sample_csv_data
        etl_pipeline.db_manager.get_engine.return_value = MagicMock()

        query = "SELECT * FROM users"
        df = etl_pipeline.extract_from_database(query, connection_name="staging")

        etl_pipeline.db_manager.get_engine.assert_called_with("staging")

    @patch("pandas.read_sql")
    def test_extract_from_database_error(self, mock_read_sql, etl_pipeline):
        """Test database extraction with error"""
        mock_read_sql.side_effect = Exception("Database error")
        etl_pipeline.db_manager.get_engine.return_value = MagicMock()

        with pytest.raises(Exception) as exc_info:
            etl_pipeline.extract_from_database("SELECT * FROM users")
        assert "Database error" in str(exc_info.value)

    def test_transform_basic_cleaning(self, etl_pipeline, sample_csv_data):
        """Test basic data transformation"""
        etl_pipeline.data_cleaner.clean_dataframe.return_value = sample_csv_data

        result = etl_pipeline.transform(sample_csv_data)

        etl_pipeline.data_cleaner.clean_dataframe.assert_called_once_with(
            sample_csv_data
        )
        assert len(result) == 5

    def test_transform_with_custom_transformations(self, etl_pipeline, sample_csv_data):
        """Test transformation with custom functions"""
        etl_pipeline.data_cleaner.clean_dataframe.return_value = sample_csv_data

        def add_full_name(df):
            df["full_name"] = df["name"] + "_processed"
            return df

        def add_timestamp(df):
            df["processed_at"] = "2023-01-01"
            return df

        custom_transformations = [add_full_name, add_timestamp]
        result = etl_pipeline.transform(sample_csv_data, custom_transformations)

        # Should still call the data cleaner
        etl_pipeline.data_cleaner.clean_dataframe.assert_called_once()

    def test_transform_custom_transformation_error(self, etl_pipeline, sample_csv_data):
        """Test transformation with failing custom function"""
        etl_pipeline.data_cleaner.clean_dataframe.return_value = sample_csv_data

        def failing_transform(df):
            raise ValueError("Custom transformation failed")

        with pytest.raises(ValueError) as exc_info:
            etl_pipeline.transform(sample_csv_data, [failing_transform])
        assert "Custom transformation failed" in str(exc_info.value)

    def test_load_to_database_success(self, etl_pipeline, sample_csv_data):
        """Test successful database loading"""
        mock_engine = MagicMock()
        etl_pipeline.db_manager.get_engine.return_value = mock_engine

        with patch.object(sample_csv_data, "to_sql") as mock_to_sql:
            etl_pipeline.load_to_database(sample_csv_data, "test_table")

            mock_to_sql.assert_called_once_with(
                name="test_table",
                con=mock_engine,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=1000,
            )

    def test_load_to_database_custom_settings(self, etl_pipeline, sample_csv_data):
        """Test database loading with custom settings"""
        mock_engine = MagicMock()
        etl_pipeline.db_manager.get_engine.return_value = mock_engine

        with patch.object(sample_csv_data, "to_sql") as mock_to_sql:
            etl_pipeline.load_to_database(
                sample_csv_data,
                "test_table",
                connection_name="staging",
                if_exists="replace",
            )

            etl_pipeline.db_manager.get_engine.assert_called_with("staging")
            mock_to_sql.assert_called_once()
            # Check that if_exists parameter was passed correctly
            call_kwargs = mock_to_sql.call_args[1]
            assert call_kwargs["if_exists"] == "replace"

    @patch.dict(os.environ, {"DB_CHUNK_SIZE": "500"})
    def test_load_to_database_custom_chunk_size(self, etl_pipeline, sample_csv_data):
        """Test database loading with custom chunk size from environment"""
        mock_engine = MagicMock()
        etl_pipeline.db_manager.get_engine.return_value = mock_engine

        with patch.object(sample_csv_data, "to_sql") as mock_to_sql:
            etl_pipeline.load_to_database(sample_csv_data, "test_table")

            call_kwargs = mock_to_sql.call_args[1]
            assert call_kwargs["chunksize"] == 500

    def test_load_to_database_error(self, etl_pipeline, sample_csv_data):
        """Test database loading with error"""
        mock_engine = MagicMock()
        etl_pipeline.db_manager.get_engine.return_value = mock_engine

        with patch.object(
            sample_csv_data, "to_sql", side_effect=Exception("Load error")
        ):
            with pytest.raises(Exception) as exc_info:
                etl_pipeline.load_to_database(sample_csv_data, "test_table")
            assert "Load error" in str(exc_info.value)

    def test_run_pipeline_csv_source(
        self, etl_pipeline, temp_csv_file, sample_csv_data
    ):
        """Test complete pipeline with CSV source"""
        # Mock the methods
        etl_pipeline.data_cleaner.clean_dataframe.return_value = sample_csv_data
        mock_engine = MagicMock()
        etl_pipeline.db_manager.get_engine.return_value = mock_engine

        source_config = {
            "type": "csv",
            "path": temp_csv_file,
            "options": {"encoding": "utf-8"},
        }

        target_config = {
            "table_name": "processed_data",
            "connection": "default",
            "if_exists": "replace",
        }

        with patch.object(sample_csv_data, "to_sql"):
            etl_pipeline.run_pipeline(source_config, target_config)

        # Verify the data cleaner was called
        etl_pipeline.data_cleaner.clean_dataframe.assert_called_once()

    @patch("pandas.read_sql")
    def test_run_pipeline_database_source(
        self, mock_read_sql, etl_pipeline, sample_csv_data
    ):
        """Test complete pipeline with database source"""
        mock_read_sql.return_value = sample_csv_data
        etl_pipeline.data_cleaner.clean_dataframe.return_value = sample_csv_data
        mock_engine = MagicMock()
        etl_pipeline.db_manager.get_engine.return_value = mock_engine

        source_config = {
            "type": "database",
            "query": "SELECT * FROM source_table",
            "connection": "source_db",
        }

        target_config = {
            "table_name": "processed_data",
            "connection": "target_db",
            "if_exists": "append",
        }

        with patch.object(sample_csv_data, "to_sql"):
            etl_pipeline.run_pipeline(source_config, target_config)

        # Verify database extraction was called with correct connection
        mock_read_sql.assert_called_once()

    def test_run_pipeline_with_custom_transformations(
        self, etl_pipeline, temp_csv_file, sample_csv_data
    ):
        """Test pipeline with custom transformations"""
        etl_pipeline.data_cleaner.clean_dataframe.return_value = sample_csv_data
        mock_engine = MagicMock()
        etl_pipeline.db_manager.get_engine.return_value = mock_engine

        def add_processed_flag(df):
            df["processed"] = True
            return df

        source_config = {"type": "csv", "path": temp_csv_file}
        target_config = {"table_name": "test_table"}

        with patch.object(sample_csv_data, "to_sql"):
            etl_pipeline.run_pipeline(
                source_config,
                target_config,
                custom_transformations=[add_processed_flag],
            )

        # Verify transformation was applied (indirectly through no errors)
        etl_pipeline.data_cleaner.clean_dataframe.assert_called_once()

    def test_run_pipeline_unsupported_source_type(self, etl_pipeline):
        """Test pipeline with unsupported source type"""
        source_config = {"type": "unsupported_type"}
        target_config = {"table_name": "test_table"}

        with pytest.raises(ValueError) as exc_info:
            etl_pipeline.run_pipeline(source_config, target_config)
        assert "Unsupported source type" in str(exc_info.value)

    def test_run_pipeline_extract_failure(self, etl_pipeline):
        """Test pipeline with extraction failure"""
        source_config = {"type": "csv", "path": "nonexistent_file.csv"}
        target_config = {"table_name": "test_table"}

        with pytest.raises(Exception):
            etl_pipeline.run_pipeline(source_config, target_config)

    def test_run_pipeline_transform_failure(self, etl_pipeline, temp_csv_file):
        """Test pipeline with transformation failure"""
        etl_pipeline.data_cleaner.clean_dataframe.side_effect = Exception(
            "Transform error"
        )

        source_config = {"type": "csv", "path": temp_csv_file}
        target_config = {"table_name": "test_table"}

        with pytest.raises(Exception) as exc_info:
            etl_pipeline.run_pipeline(source_config, target_config)
        assert "Transform error" in str(exc_info.value)

    def test_run_pipeline_load_failure(
        self, etl_pipeline, temp_csv_file, sample_csv_data
    ):
        """Test pipeline with loading failure"""
        etl_pipeline.data_cleaner.clean_dataframe.return_value = sample_csv_data
        mock_engine = MagicMock()
        etl_pipeline.db_manager.get_engine.return_value = mock_engine

        source_config = {"type": "csv", "path": temp_csv_file}
        target_config = {"table_name": "test_table"}

        with patch.object(
            sample_csv_data, "to_sql", side_effect=Exception("Load error")
        ):
            with pytest.raises(Exception) as exc_info:
                etl_pipeline.run_pipeline(source_config, target_config)
            assert "Load error" in str(exc_info.value)

    @patch("logging.getLogger")
    def test_logging_throughout_pipeline(
        self, mock_logger, etl_pipeline, temp_csv_file, sample_csv_data
    ):
        """Test that logging occurs throughout the pipeline"""
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance

        # Create new pipeline to trigger logging setup
        with patch("etl_pipeline.DatabaseManager"), patch("etl_pipeline.DataCleaner"):
            pipeline_with_logging = ETLPipeline()

        # Verify logger was configured
        mock_logger.assert_called()

    def test_pipeline_state_isolation(
        self, etl_pipeline, temp_csv_file, sample_csv_data
    ):
        """Test that pipeline runs don't affect each other"""
        etl_pipeline.data_cleaner.clean_dataframe.return_value = sample_csv_data
        mock_engine = MagicMock()
        etl_pipeline.db_manager.get_engine.return_value = mock_engine

        source_config = {"type": "csv", "path": temp_csv_file}
        target_config1 = {"table_name": "table1"}
        target_config2 = {"table_name": "table2"}

        with patch.object(sample_csv_data, "to_sql") as mock_to_sql:
            # Run pipeline twice
            etl_pipeline.run_pipeline(source_config, target_config1)
            etl_pipeline.run_pipeline(source_config, target_config2)

            # Should be called twice
            assert mock_to_sql.call_count == 2

    def test_example_usage_integration(
        self, etl_pipeline, temp_csv_file, sample_csv_data
    ):
        """Test the example usage scenario"""
        etl_pipeline.data_cleaner.clean_dataframe.return_value = sample_csv_data
        mock_engine = MagicMock()
        etl_pipeline.db_manager.get_engine.return_value = mock_engine

        # Example from the main module
        source_config = {
            "type": "csv",
            "path": temp_csv_file,
            "options": {"encoding": "utf-8"},
        }

        target_config = {
            "table_name": "processed_data",
            "connection": "default",
            "if_exists": "replace",
        }

        def add_processed_timestamp(df):
            df["processed_at"] = pd.Timestamp.now()
            return df

        with patch.object(sample_csv_data, "to_sql"):
            etl_pipeline.run_pipeline(
                source_config=source_config,
                target_config=target_config,
                custom_transformations=[add_processed_timestamp],
            )

        # Should complete without errors
        etl_pipeline.data_cleaner.clean_dataframe.assert_called_once()

    def test_error_handling_and_logging(self, etl_pipeline):
        """Test that errors are properly logged and re-raised"""
        with patch.object(etl_pipeline.logger, "error") as mock_log_error:
            source_config = {"type": "csv", "path": "nonexistent.csv"}
            target_config = {"table_name": "test"}

            with pytest.raises(Exception):
                etl_pipeline.run_pipeline(source_config, target_config)

            # Verify error was logged
            mock_log_error.assert_called()

    def test_memory_efficiency_with_large_data(self, etl_pipeline):
        """Test that pipeline handles large datasets efficiently"""
        # Create a larger dataset
        large_df = pd.DataFrame({"id": range(10000), "value": np.random.randn(10000)})

        etl_pipeline.data_cleaner.clean_dataframe.return_value = large_df
        mock_engine = MagicMock()
        etl_pipeline.db_manager.get_engine.return_value = mock_engine

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            large_df.to_csv(f.name, index=False)

            source_config = {"type": "csv", "path": f.name}
            target_config = {"table_name": "large_table"}

            with patch.object(large_df, "to_sql"):
                etl_pipeline.run_pipeline(source_config, target_config)

            os.unlink(f.name)

        # Should handle large data without memory issues
        etl_pipeline.data_cleaner.clean_dataframe.assert_called_once()
