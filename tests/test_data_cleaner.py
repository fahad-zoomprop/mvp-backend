"""
Test suite for DataCleaner utility
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add the src directory to the path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.utils.data_cleaner import DataCleaner


class TestDataCleaner:
    """Test cases for DataCleaner class"""

    @pytest.fixture
    def cleaner(self):
        """Create a DataCleaner instance for testing"""
        return DataCleaner()

    @pytest.fixture
    def sample_dirty_df(self):
        """Create a sample dirty dataframe for testing"""
        return pd.DataFrame(
            {
                "Name": ["John", "Jane", "", "null", "Bob Smith", "  Alice  "],
                "Age": ["25", "NaN", "30", "none", "35", ""],
                "City": ["New York", "NULL", "Chicago", "N/A", "Boston", "n/a"],
                "Salary": ["50000", "60000", "nan", "70000", "None", "80000"],
                "Email": [
                    "john@email.com",
                    "",
                    "jane@test.com",
                    "N/A",
                    "bob@company.com",
                    "alice@example.com",
                ],
            }
        )

    @pytest.fixture
    def messy_columns_df(self):
        """Create a dataframe with messy column names"""
        return pd.DataFrame(
            {
                "First Name": [1, 2, 3],
                "Last-Name": [4, 5, 6],
                "Email Address (Primary)": [7, 8, 9],
                "Age/Years": [10, 11, 12],
                "Salary[$]": [13, 14, 15],
            }
        )

    def test_clean_null_values_empty_strings(self, cleaner, sample_dirty_df):
        """Test cleaning of empty strings"""
        result = cleaner.clean_null_values(sample_dirty_df)
        assert pd.isna(result.loc[2, "Name"])  # Empty string should become NaN
        assert pd.isna(result.loc[5, "Age"])  # Empty string should become NaN

    def test_clean_null_values_null_strings(self, cleaner, sample_dirty_df):
        """Test cleaning of null string representations"""
        result = cleaner.clean_null_values(sample_dirty_df)
        assert pd.isna(result.loc[3, "Name"])  # 'null' should become NaN
        assert pd.isna(result.loc[1, "Age"])  # 'NaN' should become NaN
        assert pd.isna(result.loc[1, "City"])  # 'NULL' should become NaN
        assert pd.isna(result.loc[3, "City"])  # 'N/A' should become NaN
        assert pd.isna(result.loc[5, "City"])  # 'n/a' should become NaN

    def test_clean_null_values_case_insensitive(self, cleaner):
        """Test case-insensitive null cleaning"""
        df = pd.DataFrame(
            {
                "col1": ["NONE", "none", "None", "NULL", "null", "Null"],
                "col2": ["NAN", "nan", "NaN", "NA", "na", "Na"],
            }
        )
        result = cleaner.clean_null_values(df)
        assert result["col1"].isna().all()
        assert result["col2"].isna().all()

    def test_clean_whitespace(self, cleaner, sample_dirty_df):
        """Test whitespace cleaning"""
        result = cleaner.clean_whitespace(sample_dirty_df)
        assert result.loc[5, "Name"] == "Alice"  # '  Alice  ' should become 'Alice'
        # Check that non-string columns are not affected
        assert not pd.isna(result.loc[0, "Age"])  # Should preserve '25'

    def test_standardize_column_names(self, cleaner, messy_columns_df):
        """Test column name standardization"""
        result = cleaner.standardize_column_names(messy_columns_df)
        expected_columns = [
            "first_name",
            "last_name",
            "email_address_primary",
            "age_years",
            "salary_",
        ]
        assert list(result.columns) == expected_columns

    def test_remove_duplicate_rows(self, cleaner):
        """Test duplicate row removal"""
        df = pd.DataFrame({"A": [1, 2, 2, 3], "B": ["a", "b", "b", "c"]})
        result = cleaner.remove_duplicate_rows(df)
        assert len(result) == 3
        assert result.iloc[1]["A"] == 2  # First occurrence should be kept

    def test_remove_duplicate_rows_subset(self, cleaner):
        """Test duplicate removal with subset of columns"""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "name": ["John", "Jane", "John", "Bob"],
                "age": [25, 30, 35, 40],
            }
        )
        result = cleaner.remove_duplicate_rows(df, subset=["name"])
        assert len(result) == 3  # Should remove duplicate 'John'
        assert (
            result[result["name"] == "John"].iloc[0]["age"] == 25
        )  # First John should be kept

    def test_convert_data_types_manual_mapping(self, cleaner):
        """Test data type conversion with manual mapping"""
        df = pd.DataFrame(
            {
                "numbers": ["1", "2", "3"],
                "floats": ["1.5", "2.5", "3.5"],
                "text": ["a", "b", "c"],
            }
        )
        type_mapping = {"numbers": "int64", "floats": "float64"}
        result = cleaner.convert_data_types(df, type_mapping)
        assert result["numbers"].dtype == "int64"
        assert result["floats"].dtype == "float64"
        assert result["text"].dtype == "object"

    def test_convert_data_types_automatic(self, cleaner):
        """Test automatic data type conversion"""
        df = pd.DataFrame(
            {
                "mostly_numeric": ["1", "2", "3", "4", "not_a_number"],
                "all_numeric": ["10", "20", "30", "40", "50"],
            }
        )
        result = cleaner.convert_data_types(df)
        # all_numeric should be converted (100% convertible)
        assert pd.api.types.is_numeric_dtype(result["all_numeric"])
        # mostly_numeric should remain object (only 80% convertible, below 90% threshold)
        assert result["mostly_numeric"].dtype == "object"

    def test_handle_outliers_iqr_method(self, cleaner):
        """Test outlier detection using IQR method"""
        df = pd.DataFrame({"values": [1, 2, 3, 4, 5, 100]})  # 100 is an outlier
        # This method doesn't modify data, just detects outliers
        result = cleaner.handle_outliers(df, columns=["values"], method="iqr")
        assert len(result) == len(df)  # No rows should be removed

    def test_clean_dataframe_full_pipeline(self, cleaner, sample_dirty_df):
        """Test the complete cleaning pipeline"""
        result = cleaner.clean_dataframe(
            sample_dirty_df,
            clean_nulls=True,
            clean_whitespace=True,
            standardize_columns=True,
            remove_duplicates=False,
            convert_types=True,
        )

        # Check that various cleaning operations were applied
        assert "name" in result.columns  # Column standardization
        assert pd.isna(result.loc[2, "name"])  # Empty string cleaned
        assert pd.isna(result.loc[3, "name"])  # 'null' cleaned
        assert result.loc[5, "name"] == "Alice"  # Whitespace cleaned

    def test_clean_dataframe_selective_operations(self, cleaner, sample_dirty_df):
        """Test selective cleaning operations"""
        result = cleaner.clean_dataframe(
            sample_dirty_df,
            clean_nulls=True,
            clean_whitespace=False,
            standardize_columns=False,
            remove_duplicates=False,
            convert_types=False,
        )

        # Original column names should be preserved
        assert "Name" in result.columns
        # Null cleaning should be applied
        assert pd.isna(result.loc[3, "Name"])
        # Whitespace should NOT be cleaned
        assert result.loc[5, "Name"] == "  Alice  "

    def test_null_patterns_comprehensive(self, cleaner):
        """Test all null patterns are properly handled"""
        null_values = [
            "",
            " ",
            "null",
            "NULL",
            "Null",
            "none",
            "None",
            "NONE",
            "na",
            "NA",
            "Na",
            "nan",
            "NaN",
            "NAN",
            "n/a",
            "N/A",
            "nil",
            "NIL",
            "Nil",
            "undefined",
            "UNDEFINED",
            "#N/A",
            "#NA",
            "#NULL!",
            "#DIV/0!",
            "missing",
            "MISSING",
        ]

        df = pd.DataFrame({"test_col": null_values})
        result = cleaner.clean_null_values(df)

        # All values should be converted to NaN
        assert result["test_col"].isna().all()

    @patch("logging.getLogger")
    def test_logging_calls(self, mock_logger, cleaner, sample_dirty_df):
        """Test that appropriate logging calls are made"""
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance

        cleaner_with_logging = DataCleaner()
        cleaner_with_logging.clean_dataframe(sample_dirty_df)

        # Check that info logging was called
        assert mock_logger_instance.info.called

    def test_edge_cases_empty_dataframe(self, cleaner):
        """Test handling of empty dataframes"""
        empty_df = pd.DataFrame()
        result = cleaner.clean_dataframe(empty_df)
        assert len(result) == 0
        assert len(result.columns) == 0

    def test_edge_cases_single_column(self, cleaner):
        """Test handling of single column dataframes"""
        df = pd.DataFrame({"col": ["", "null", "valid", "NaN"]})
        result = cleaner.clean_dataframe(df)
        assert len(result) == 4
        assert result.loc[2, "col"] == "valid"
        assert pd.isna(result.loc[0, "col"])
        assert pd.isna(result.loc[1, "col"])
        assert pd.isna(result.loc[3, "col"])

    def test_numeric_data_preservation(self, cleaner):
        """Test that numeric data is properly preserved during cleaning"""
        df = pd.DataFrame(
            {
                "integers": [1, 2, 3, 4, 5],
                "floats": [1.1, 2.2, 3.3, 4.4, 5.5],
                "mixed": [1, 2.5, 3, 4.7, 5],
            }
        )
        result = cleaner.clean_dataframe(df)

        # Numeric data should be preserved
        assert all(result["integers"] == df["integers"])
        assert all(result["floats"] == df["floats"])
        assert all(result["mixed"] == df["mixed"])
