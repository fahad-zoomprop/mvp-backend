"""
Data Cleaner Utility Module
Handles common data cleaning operations for ETL processes
"""

import logging
from typing import Any, List, Union

import numpy as np
import pandas as pd


class DataCleaner:
    """
    Utility class for cleaning and standardizing dataframe data
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Define patterns that should be converted to NULL/NaN
        self.null_patterns = {
            "empty_string": ["", " ", "  "],
            "null_strings": [
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
                "n/A",
                "N/a",
                "nil",
                "NIL",
                "Nil",
                "undefined",
                "UNDEFINED",
                "Undefined",
                "#N/A",
                "#NA",
                "#NULL!",
                "#DIV/0!",
                "missing",
                "MISSING",
                "Missing",
            ],
        }

    def clean_null_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean various representations of null values and convert them to proper NaN

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        df_cleaned = df.copy()

        # Convert empty strings and whitespace-only strings to NaN
        for pattern in self.null_patterns["empty_string"]:
            df_cleaned = df_cleaned.replace(pattern, np.nan)

        # Convert null string representations to NaN
        for pattern in self.null_patterns["null_strings"]:
            df_cleaned = df_cleaned.replace(pattern, np.nan)

        # Handle case-insensitive replacements for string columns
        for col in df_cleaned.select_dtypes(include=["object"]).columns:
            # Create a mask for string columns
            string_mask = (
                df_cleaned[col]
                .astype(str)
                .str.lower()
                .isin([p.lower() for p in self.null_patterns["null_strings"]])
            )
            df_cleaned.loc[string_mask, col] = np.nan

        return df_cleaned

    def clean_whitespace(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove leading and trailing whitespace from string columns

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        df_cleaned = df.copy()

        # Clean whitespace in string/object columns
        string_columns = df_cleaned.select_dtypes(include=["object"]).columns
        for col in string_columns:
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
            # Convert 'nan' strings back to actual NaN (from previous str conversion)
            df_cleaned[col] = df_cleaned[col].replace("nan", np.nan)

        return df_cleaned

    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names (lowercase, replace spaces with underscores)

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Dataframe with standardized column names
        """
        df_cleaned = df.copy()

        # Clean column names
        df_cleaned.columns = (
            df_cleaned.columns.str.lower()
            .str.replace(" ", "_")
            .str.replace("-", "_")
            .str.replace(".", "_")
            .str.replace("(", "")
            .str.replace(")", "")
            .str.replace("[", "")
            .str.replace("]", "")
            .str.replace("/", "_")
            .str.replace("\\", "_")
        )

        return df_cleaned

    def remove_duplicate_rows(
        self, df: pd.DataFrame, subset: Union[List[str], None] = None
    ) -> pd.DataFrame:
        """
        Remove duplicate rows from dataframe

        Args:
            df (pd.DataFrame): Input dataframe
            subset (List[str], optional): Column names to consider for duplicates

        Returns:
            pd.DataFrame: Dataframe without duplicates
        """
        df_cleaned = df.copy()
        initial_count = len(df_cleaned)

        df_cleaned = df_cleaned.drop_duplicates(subset=subset, keep="first")

        removed_count = initial_count - len(df_cleaned)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} duplicate rows")

        return df_cleaned

    def convert_data_types(
        self, df: pd.DataFrame, type_mapping: dict = None
    ) -> pd.DataFrame:
        """
        Convert data types based on mapping or automatic inference

        Args:
            df (pd.DataFrame): Input dataframe
            type_mapping (dict, optional): Manual type mapping {column: dtype}

        Returns:
            pd.DataFrame: Dataframe with converted types
        """
        df_cleaned = df.copy()

        if type_mapping:
            for col, dtype in type_mapping.items():
                if col in df_cleaned.columns:
                    try:
                        df_cleaned[col] = df_cleaned[col].astype(dtype)
                    except Exception as e:
                        self.logger.warning(
                            f"Could not convert {col} to {dtype}: {str(e)}"
                        )

        # Automatic type inference for numeric columns
        for col in df_cleaned.select_dtypes(include=["object"]).columns:
            # Try to convert to numeric
            numeric_series = pd.to_numeric(df_cleaned[col], errors="coerce")
            # If more than 90% of non-null values can be converted, convert the column
            non_null_ratio = (
                numeric_series.notna().sum() / df_cleaned[col].notna().sum()
            )
            if non_null_ratio > 0.9:
                df_cleaned[col] = numeric_series

        return df_cleaned

    def handle_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        method: str = "iqr",
        factor: float = 1.5,
    ) -> pd.DataFrame:
        """
        Handle outliers in numeric columns

        Args:
            df (pd.DataFrame): Input dataframe
            columns (List[str], optional): Columns to check for outliers
            method (str): Method to detect outliers ('iqr' or 'zscore')
            factor (float): Factor for outlier detection

        Returns:
            pd.DataFrame: Dataframe with outliers handled
        """
        df_cleaned = df.copy()

        if columns is None:
            columns = df_cleaned.select_dtypes(include=[np.number]).columns

        for col in columns:
            if col not in df_cleaned.columns:
                continue

            if method == "iqr":
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR

                # Flag outliers but don't remove them (just log)
                outliers = df_cleaned[
                    (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
                ]
                if len(outliers) > 0:
                    self.logger.info(f"Found {len(outliers)} outliers in column {col}")

        return df_cleaned

    def clean_dataframe(
        self,
        df: pd.DataFrame,
        clean_nulls: bool = True,
        clean_whitespace: bool = True,
        standardize_columns: bool = True,
        remove_duplicates: bool = False,
        convert_types: bool = True,
        type_mapping: dict = None,
    ) -> pd.DataFrame:
        """
        Apply all cleaning operations to the dataframe

        Args:
            df (pd.DataFrame): Input dataframe
            clean_nulls (bool): Whether to clean null values
            clean_whitespace (bool): Whether to clean whitespace
            standardize_columns (bool): Whether to standardize column names
            remove_duplicates (bool): Whether to remove duplicate rows
            convert_types (bool): Whether to convert data types
            type_mapping (dict): Manual type mapping

        Returns:
            pd.DataFrame: Fully cleaned dataframe
        """
        self.logger.info(
            f"Starting data cleaning for dataframe with {len(df)} rows and {len(df.columns)} columns"
        )

        df_cleaned = df.copy()

        if standardize_columns:
            df_cleaned = self.standardize_column_names(df_cleaned)

        if clean_nulls:
            df_cleaned = self.clean_null_values(df_cleaned)

        if clean_whitespace:
            df_cleaned = self.clean_whitespace(df_cleaned)

        if remove_duplicates:
            df_cleaned = self.remove_duplicate_rows(df_cleaned)

        if convert_types:
            df_cleaned = self.convert_data_types(df_cleaned, type_mapping)

        self.logger.info(f"Data cleaning completed. Final shape: {df_cleaned.shape}")
        return df_cleaned
