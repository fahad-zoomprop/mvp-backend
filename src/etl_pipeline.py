"""
Zoomprop ETL Template - Main ETL Module
A standardized ETL framework using Pandas and SQLAlchemy
"""

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from .utils.data_cleaner import DataCleaner
from .utils.db_manager import DatabaseManager

# Load environment variables
load_dotenv()


class ETLPipeline:
    """
    Main ETL Pipeline class that orchestrates the extract, transform, and load operations.
    """

    def __init__(self):
        self.logger = self._setup_logging()
        self.db_manager = DatabaseManager()
        self.data_cleaner = DataCleaner()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        return logging.getLogger(__name__)

    def extract_from_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Extract data from CSV file

        Args:
            file_path (str): Path to the CSV file
            **kwargs: Additional arguments for pandas.read_csv()

        Returns:
            pd.DataFrame: Extracted data
        """
        try:
            self.logger.info(f"Extracting data from CSV: {file_path}")
            df = pd.read_csv(file_path, **kwargs)
            self.logger.info(f"Successfully extracted {len(df)} rows")
            return df
        except Exception as e:
            self.logger.error(f"Error extracting from CSV: {str(e)}")
            raise

    def extract_from_database(
        self, query: str, connection_name: str = "default"
    ) -> pd.DataFrame:
        """
        Extract data from database using SQL query

        Args:
            query (str): SQL query to execute
            connection_name (str): Database connection name from environment

        Returns:
            pd.DataFrame: Extracted data
        """
        try:
            self.logger.info(f"Extracting data from database using query")
            engine = self.db_manager.get_engine(connection_name)
            df = pd.read_sql(query, engine)
            self.logger.info(f"Successfully extracted {len(df)} rows from database")
            return df
        except Exception as e:
            self.logger.error(f"Error extracting from database: {str(e)}")
            raise

    def transform(
        self, df: pd.DataFrame, custom_transformations: Optional[List] = None
    ) -> pd.DataFrame:
        """
        Transform the data using standard cleaning and optional custom transformations

        Args:
            df (pd.DataFrame): Input dataframe
            custom_transformations (List, optional): List of custom transformation functions

        Returns:
            pd.DataFrame: Transformed data
        """
        try:
            self.logger.info("Starting data transformation")

            # Apply standard cleaning
            df_cleaned = self.data_cleaner.clean_dataframe(df)

            # Apply custom transformations if provided
            if custom_transformations:
                for transform_func in custom_transformations:
                    df_cleaned = transform_func(df_cleaned)

            self.logger.info("Data transformation completed successfully")
            return df_cleaned
        except Exception as e:
            self.logger.error(f"Error during transformation: {str(e)}")
            raise

    def load_to_database(
        self,
        df: pd.DataFrame,
        table_name: str,
        connection_name: str = "default",
        if_exists: str = "append",
    ) -> None:
        """
        Load data to database table

        Args:
            df (pd.DataFrame): Data to load
            table_name (str): Target table name
            connection_name (str): Database connection name
            if_exists (str): How to behave if table exists ('fail', 'replace', 'append')
        """
        try:
            self.logger.info(f"Loading {len(df)} rows to table: {table_name}")
            engine = self.db_manager.get_engine(connection_name)

            df.to_sql(
                name=table_name,
                con=engine,
                if_exists=if_exists,
                index=False,
                method="multi",
                chunksize=int(os.getenv("DB_CHUNK_SIZE", 1000)),
            )

            self.logger.info(f"Successfully loaded data to {table_name}")
        except Exception as e:
            self.logger.error(f"Error loading to database: {str(e)}")
            raise

    def run_pipeline(
        self,
        source_config: Dict[str, Any],
        target_config: Dict[str, Any],
        custom_transformations: Optional[List] = None,
    ) -> None:
        """
        Run the complete ETL pipeline

        Args:
            source_config (Dict): Source configuration
            target_config (Dict): Target configuration
            custom_transformations (List, optional): Custom transformation functions
        """
        try:
            self.logger.info("Starting ETL pipeline")

            # Extract
            if source_config["type"] == "csv":
                df = self.extract_from_csv(
                    source_config["path"], **source_config.get("options", {})
                )
            elif source_config["type"] == "database":
                df = self.extract_from_database(
                    source_config["query"], source_config.get("connection", "default")
                )
            else:
                raise ValueError(f"Unsupported source type: {source_config['type']}")

            # Transform
            df_transformed = self.transform(df, custom_transformations)

            # Load
            self.load_to_database(
                df_transformed,
                target_config["table_name"],
                target_config.get("connection", "default"),
                target_config.get("if_exists", "append"),
            )

            self.logger.info("ETL pipeline completed successfully")

        except Exception as e:
            self.logger.error(f"ETL pipeline failed: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Example configuration
    source_config = {
        "type": "csv",
        "path": "data/input.csv",
        "options": {"encoding": "utf-8"},
    }

    target_config = {
        "table_name": "processed_data",
        "connection": "default",
        "if_exists": "replace",
    }

    # Custom transformation example
    def add_processed_timestamp(df):
        df["processed_at"] = pd.Timestamp.now()
        return df

    # Run pipeline
    etl = ETLPipeline()
    etl.run_pipeline(
        source_config=source_config,
        target_config=target_config,
        custom_transformations=[add_processed_timestamp],
    )
