"""
Database Manager Utility
Handles database connections and operations using SQLAlchemy
"""

import logging
import os
from typing import Dict, Optional
from urllib.parse import quote_plus

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError


class DatabaseManager:
    """
    Manages database connections and provides utility methods for database operations
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._engines: Dict[str, Engine] = {}

    def _build_connection_string(self, connection_name: str = "default") -> str:
        """
        Build database connection string from environment variables

        Args:
            connection_name (str): Name of the connection configuration

        Returns:
            str: Database connection string
        """
        # Support multiple database connections by using prefixes
        prefix = f"{connection_name.upper()}_" if connection_name != "default" else ""

        # Get database configuration from environment
        db_type = os.getenv(f"{prefix}DB_TYPE", "postgresql")
        db_host = os.getenv(f"{prefix}DB_HOST", "localhost")
        db_port = os.getenv(f"{prefix}DB_PORT")
        db_name = os.getenv(f"{prefix}DB_NAME")
        db_user = os.getenv(f"{prefix}DB_USER")
        db_password = os.getenv(f"{prefix}DB_PASSWORD")

        if not all([db_name, db_user, db_password]):
            raise ValueError(
                f"Missing required database configuration for connection '{connection_name}'. "
                f"Please set {prefix}DB_NAME, {prefix}DB_USER, and {prefix}DB_PASSWORD"
            )

        # URL encode password to handle special characters
        encoded_password = quote_plus(db_password)

        # Build connection string based on database type
        if db_type.lower() == "postgresql":
            port = db_port or "5432"
            return (
                f"postgresql://{db_user}:{encoded_password}@{db_host}:{port}/{db_name}"
            )
        elif db_type.lower() == "mysql":
            port = db_port or "3306"
            return f"mysql+pymysql://{db_user}:{encoded_password}@{db_host}:{port}/{db_name}"
        elif db_type.lower() == "sqlite":
            return f"sqlite:///{db_name}"
        elif db_type.lower() == "mssql":
            port = db_port or "1433"
            return f"mssql+pyodbc://{db_user}:{encoded_password}@{db_host}:{port}/{db_name}?driver=ODBC+Driver+17+for+SQL+Server"
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    def get_engine(self, connection_name: str = "default") -> Engine:
        """
        Get or create a database engine for the specified connection

        Args:
            connection_name (str): Name of the connection

        Returns:
            Engine: SQLAlchemy engine instance
        """
        if connection_name not in self._engines:
            try:
                connection_string = self._build_connection_string(connection_name)

                # Create engine with common configurations
                engine = create_engine(
                    connection_string,
                    pool_size=int(
                        os.getenv(
                            (
                                f"{connection_name.upper()}_POOL_SIZE"
                                if connection_name != "default"
                                else "DB_POOL_SIZE"
                            ),
                            5,
                        )
                    ),
                    max_overflow=int(
                        os.getenv(
                            (
                                f"{connection_name.upper()}_MAX_OVERFLOW"
                                if connection_name != "default"
                                else "DB_MAX_OVERFLOW"
                            ),
                            10,
                        )
                    ),
                    pool_timeout=int(
                        os.getenv(
                            (
                                f"{connection_name.upper()}_POOL_TIMEOUT"
                                if connection_name != "default"
                                else "DB_POOL_TIMEOUT"
                            ),
                            30,
                        )
                    ),
                    pool_recycle=int(
                        os.getenv(
                            (
                                f"{connection_name.upper()}_POOL_RECYCLE"
                                if connection_name != "default"
                                else "DB_POOL_RECYCLE"
                            ),
                            3600,
                        )
                    ),
                    echo=os.getenv(
                        (
                            f"{connection_name.upper()}_DB_ECHO"
                            if connection_name != "default"
                            else "DB_ECHO"
                        ),
                        "false",
                    ).lower()
                    == "true",
                )

                # Test the connection
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))

                self._engines[connection_name] = engine
                self.logger.info(
                    f"Successfully created database engine for connection '{connection_name}'"
                )

            except Exception as e:
                self.logger.error(
                    f"Failed to create database engine for connection '{connection_name}': {str(e)}"
                )
                raise

        return self._engines[connection_name]

    def test_connection(self, connection_name: str = "default") -> bool:
        """
        Test database connection

        Args:
            connection_name (str): Name of the connection to test

        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            engine = self.get_engine(connection_name)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self.logger.info(f"Connection test successful for '{connection_name}'")
            return True
        except Exception as e:
            self.logger.error(
                f"Connection test failed for '{connection_name}': {str(e)}"
            )
            return False

    def execute_query(
        self,
        query: str,
        connection_name: str = "default",
        params: Optional[dict] = None,
    ) -> None:
        """
        Execute a SQL query (non-select)

        Args:
            query (str): SQL query to execute
            connection_name (str): Database connection name
            params (dict, optional): Query parameters
        """
        try:
            engine = self.get_engine(connection_name)
            with engine.connect() as conn:
                if params:
                    conn.execute(text(query), params)
                else:
                    conn.execute(text(query))
                conn.commit()
            self.logger.info("Query executed successfully")
        except SQLAlchemyError as e:
            self.logger.error(f"Database error executing query: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            raise

    def get_table_info(self, table_name: str, connection_name: str = "default") -> dict:
        """
        Get information about a table (columns, types, etc.)

        Args:
            table_name (str): Name of the table
            connection_name (str): Database connection name

        Returns:
            dict: Table information
        """
        try:
            engine = self.get_engine(connection_name)

            # Query to get column information (works for most SQL databases)
            query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_name = :table_name
            ORDER BY ordinal_position
            """

            import pandas as pd

            df = pd.read_sql(query, engine, params={"table_name": table_name})

            return {"table_name": table_name, "columns": df.to_dict("records")}
        except Exception as e:
            self.logger.error(f"Error getting table info for {table_name}: {str(e)}")
            raise

    def close_all_connections(self) -> None:
        """
        Close all database connections
        """
        for name, engine in self._engines.items():
            try:
                engine.dispose()
                self.logger.info(f"Closed database connection '{name}'")
            except Exception as e:
                self.logger.error(f"Error closing connection '{name}': {str(e)}")

        self._engines.clear()

    def __del__(self):
        """Cleanup database connections on object destruction"""
        self.close_all_connections()
