"""
Test suite for DatabaseManager utility
"""

import os
import sys
from unittest.mock import MagicMock, mock_open, patch

import pytest
from sqlalchemy.exc import SQLAlchemyError

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.utils.db_manager import DatabaseManager


class TestDatabaseManager:
    """Test cases for DatabaseManager class"""

    @pytest.fixture
    def db_manager(self):
        """Create a DatabaseManager instance for testing"""
        return DatabaseManager()

    @pytest.fixture
    def mock_env_vars(self):
        """Mock environment variables for testing"""
        return {
            "DB_TYPE": "postgresql",
            "DB_HOST": "localhost",
            "DB_PORT": "5432",
            "DB_NAME": "test_db",
            "DB_USER": "test_user",
            "DB_PASSWORD": "test_password",
            "DB_POOL_SIZE": "5",
            "DB_MAX_OVERFLOW": "10",
            "DB_POOL_TIMEOUT": "30",
            "DB_POOL_RECYCLE": "3600",
            "DB_ECHO": "false",
        }

    @patch.dict(os.environ, {})
    def test_build_connection_string_missing_required_vars(self, db_manager):
        """Test connection string building with missing required variables"""
        with pytest.raises(ValueError) as exc_info:
            db_manager._build_connection_string()
        assert "Missing required database configuration" in str(exc_info.value)

    @patch.dict(os.environ)
    def test_build_connection_string_postgresql(self, db_manager, mock_env_vars):
        """Test PostgreSQL connection string building"""
        os.environ.update(mock_env_vars)
        connection_string = db_manager._build_connection_string()
        expected = "postgresql://test_user:test_password@localhost:5432/test_db"
        assert connection_string == expected

    @patch.dict(os.environ)
    def test_build_connection_string_mysql(self, db_manager, mock_env_vars):
        """Test MySQL connection string building"""
        mock_env_vars["DB_TYPE"] = "mysql"
        mock_env_vars["DB_PORT"] = "3306"
        os.environ.update(mock_env_vars)
        connection_string = db_manager._build_connection_string()
        expected = "mysql+pymysql://test_user:test_password@localhost:3306/test_db"
        assert connection_string == expected

    @patch.dict(os.environ)
    def test_build_connection_string_sqlite(self, db_manager, mock_env_vars):
        """Test SQLite connection string building"""
        mock_env_vars["DB_TYPE"] = "sqlite"
        os.environ.update(mock_env_vars)
        connection_string = db_manager._build_connection_string()
        expected = "sqlite:///test_db"
        assert connection_string == expected

    @patch.dict(os.environ)
    def test_build_connection_string_mssql(self, db_manager, mock_env_vars):
        """Test SQL Server connection string building"""
        mock_env_vars["DB_TYPE"] = "mssql"
        mock_env_vars["DB_PORT"] = "1433"
        os.environ.update(mock_env_vars)
        connection_string = db_manager._build_connection_string()
        expected = "mssql+pyodbc://test_user:test_password@localhost:1433/test_db?driver=ODBC+Driver+17+for+SQL+Server"
        assert connection_string == expected

    @patch.dict(os.environ)
    def test_build_connection_string_unsupported_type(self, db_manager, mock_env_vars):
        """Test unsupported database type"""
        mock_env_vars["DB_TYPE"] = "unsupported_db"
        os.environ.update(mock_env_vars)
        with pytest.raises(ValueError) as exc_info:
            db_manager._build_connection_string()
        assert "Unsupported database type" in str(exc_info.value)

    @patch.dict(os.environ)
    def test_build_connection_string_special_password(self, db_manager, mock_env_vars):
        """Test connection string with special characters in password"""
        mock_env_vars["DB_PASSWORD"] = "p@ssw0rd!#$%"
        os.environ.update(mock_env_vars)
        connection_string = db_manager._build_connection_string()
        # Password should be URL encoded
        assert "p%40ssw0rd%21%23%24%25" in connection_string

    @patch.dict(os.environ)
    def test_secondary_connection_config(self, db_manager, mock_env_vars):
        """Test secondary database connection configuration"""
        # Set up staging environment variables
        staging_vars = {f"STAGING_{k}": v for k, v in mock_env_vars.items()}
        os.environ.update(staging_vars)

        connection_string = db_manager._build_connection_string("staging")
        expected = "postgresql://test_user:test_password@localhost:5432/test_db"
        assert connection_string == expected

    @patch("sqlalchemy.create_engine")
    @patch.dict(os.environ)
    def test_get_engine_creation(self, mock_create_engine, db_manager, mock_env_vars):
        """Test engine creation and caching"""
        # Setup mocks
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine

        os.environ.update(mock_env_vars)

        # First call should create engine
        engine1 = db_manager.get_engine()
        assert mock_create_engine.called
        assert engine1 == mock_engine

        # Second call should return cached engine
        mock_create_engine.reset_mock()
        engine2 = db_manager.get_engine()
        assert not mock_create_engine.called
        assert engine2 == mock_engine
        assert engine1 is engine2

    @patch("sqlalchemy.create_engine")
    @patch.dict(os.environ)
    def test_get_engine_connection_test_failure(
        self, mock_create_engine, db_manager, mock_env_vars
    ):
        """Test engine creation with connection test failure"""
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("Connection failed")
        mock_create_engine.return_value = mock_engine

        os.environ.update(mock_env_vars)

        with pytest.raises(Exception) as exc_info:
            db_manager.get_engine()
        assert "Connection failed" in str(exc_info.value)

    @patch("sqlalchemy.create_engine")
    @patch.dict(os.environ)
    def test_test_connection_success(
        self, mock_create_engine, db_manager, mock_env_vars
    ):
        """Test successful connection test"""
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine

        os.environ.update(mock_env_vars)

        result = db_manager.test_connection()
        assert result is True
        mock_connection.execute.assert_called()

    @patch("sqlalchemy.create_engine")
    @patch.dict(os.environ)
    def test_test_connection_failure(
        self, mock_create_engine, db_manager, mock_env_vars
    ):
        """Test connection test failure"""
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("Connection failed")
        mock_create_engine.return_value = mock_engine

        os.environ.update(mock_env_vars)

        result = db_manager.test_connection()
        assert result is False

    @patch("sqlalchemy.create_engine")
    @patch.dict(os.environ)
    def test_execute_query_success(self, mock_create_engine, db_manager, mock_env_vars):
        """Test successful query execution"""
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine

        os.environ.update(mock_env_vars)

        db_manager.execute_query("CREATE TABLE test (id INT)")
        mock_connection.execute.assert_called()
        mock_connection.commit.assert_called()

    @patch("sqlalchemy.create_engine")
    @patch.dict(os.environ)
    def test_execute_query_with_params(
        self, mock_create_engine, db_manager, mock_env_vars
    ):
        """Test query execution with parameters"""
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine

        os.environ.update(mock_env_vars)

        params = {"name": "John", "age": 30}
        db_manager.execute_query(
            "INSERT INTO users (name, age) VALUES (:name, :age)", params=params
        )

        mock_connection.execute.assert_called()
        mock_connection.commit.assert_called()

    @patch("sqlalchemy.create_engine")
    @patch.dict(os.environ)
    def test_execute_query_sqlalchemy_error(
        self, mock_create_engine, db_manager, mock_env_vars
    ):
        """Test query execution with SQLAlchemy error"""
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_connection.execute.side_effect = SQLAlchemyError("SQL Error")
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine

        os.environ.update(mock_env_vars)

        with pytest.raises(SQLAlchemyError):
            db_manager.execute_query("INVALID SQL")

    @patch("pandas.read_sql")
    @patch("sqlalchemy.create_engine")
    @patch.dict(os.environ)
    def test_get_table_info_success(
        self, mock_create_engine, mock_read_sql, db_manager, mock_env_vars
    ):
        """Test successful table info retrieval"""
        # Setup mocks
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        mock_df = MagicMock()
        mock_df.to_dict.return_value = [
            {"column_name": "id", "data_type": "integer", "is_nullable": "NO"},
            {"column_name": "name", "data_type": "varchar", "is_nullable": "YES"},
        ]
        mock_read_sql.return_value = mock_df

        os.environ.update(mock_env_vars)

        result = db_manager.get_table_info("test_table")

        assert result["table_name"] == "test_table"
        assert len(result["columns"]) == 2
        mock_read_sql.assert_called_once()

    @patch("sqlalchemy.create_engine")
    @patch.dict(os.environ)
    def test_get_table_info_error(self, mock_create_engine, db_manager, mock_env_vars):
        """Test table info retrieval error"""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        os.environ.update(mock_env_vars)

        with patch("pandas.read_sql", side_effect=Exception("Table not found")):
            with pytest.raises(Exception) as exc_info:
                db_manager.get_table_info("nonexistent_table")
            assert "Table not found" in str(exc_info.value)

    def test_close_all_connections(self, db_manager):
        """Test closing all database connections"""
        # Create mock engines
        mock_engine1 = MagicMock()
        mock_engine2 = MagicMock()

        db_manager._engines = {"default": mock_engine1, "staging": mock_engine2}

        db_manager.close_all_connections()

        mock_engine1.dispose.assert_called_once()
        mock_engine2.dispose.assert_called_once()
        assert len(db_manager._engines) == 0

    def test_close_connections_with_error(self, db_manager):
        """Test closing connections when dispose raises error"""
        mock_engine = MagicMock()
        mock_engine.dispose.side_effect = Exception("Dispose error")

        db_manager._engines = {"default": mock_engine}

        # Should not raise exception
        db_manager.close_all_connections()
        assert len(db_manager._engines) == 0

    @patch.dict(os.environ)
    def test_multiple_connection_names(self, db_manager, mock_env_vars):
        """Test handling multiple named connections"""
        # Setup default connection
        os.environ.update(mock_env_vars)

        # Setup staging connection
        staging_vars = {f"STAGING_{k}": v for k, v in mock_env_vars.items()}
        staging_vars["STAGING_DB_NAME"] = "staging_db"
        os.environ.update(staging_vars)

        default_conn_str = db_manager._build_connection_string("default")
        staging_conn_str = db_manager._build_connection_string("staging")

        assert "test_db" in default_conn_str
        assert "staging_db" in staging_conn_str
        assert default_conn_str != staging_conn_str

    def test_destructor_calls_close(self, db_manager):
        """Test that destructor calls close_all_connections"""
        mock_engine = MagicMock()
        db_manager._engines = {"default": mock_engine}

        # Manually call destructor
        db_manager.__del__()

        mock_engine.dispose.assert_called_once()

    @patch("logging.getLogger")
    def test_logging_calls(self, mock_logger, db_manager):
        """Test that appropriate logging calls are made"""
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance

        db_manager_with_logging = DatabaseManager()

        # Logger should be initialized
        mock_logger.assert_called_with("utils.database")

    @patch.dict(os.environ)
    def test_default_port_assignment(self, db_manager, mock_env_vars):
        """Test that default ports are assigned correctly"""
        # Remove port from env vars
        del mock_env_vars["DB_PORT"]
        os.environ.update(mock_env_vars)

        # Test PostgreSQL default port
        connection_string = db_manager._build_connection_string()
        assert ":5432/" in connection_string

        # Test MySQL default port
        mock_env_vars["DB_TYPE"] = "mysql"
        os.environ.update(mock_env_vars)
        connection_string = db_manager._build_connection_string()
        assert ":3306/" in connection_string
