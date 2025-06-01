# Zoomprop ETL Template

A lightweight ETL template designed for rapid MVP development with clean data processing and focused unit testing.

## 🚀 Quick Start

```bash
# Clone the template
git clone <repository-url> my-etl-project
cd my-etl-project

# Install dependencies
pipenv install --dev

# Configure environment
cp .env.template .env
# Edit .env with your database credentials

# Run tests
pipenv run pytest

# Ready to build your ETL!
```

## 📁 Project Structure

```raw
zoomprop-etl-template/
├── README.md
├── Pipfile                      # Python dependencies
├── .env.template                # Environment variables template
├── .gitignore
├── .pre-commit-config.yaml      # Code quality hooks
├── LICENSE
├── src/
│   ├── __init__.py
│   ├── etl_pipeline.py         # Main ETL orchestration
│   └── utils/
│       ├── __init__.py
│       ├── data_cleaner.py     # Data cleaning utilities
│       └── db_manager.py       # Database management
└── tests/
    ├── __init__.py
    ├── test_data_cleaner.py    # Data cleaning tests
    ├── test_db_manager.py      # Database operation tests
    └── test_etl_pipeline.py    # Pipeline workflow tests
```

## ✨ Key Features

- **Smart Data Cleaning**: Handles 20+ null patterns (`""`, `"null"`, `"NaN"`, `"N/A"`, etc.)
- **Multi-Database Support**: PostgreSQL, MySQL, SQLite, SQL Server
- **Environment Configuration**: Multiple database connections via `.env`
- **Comprehensive Testing**: 100+ unit tests for reliability
- **Code Quality**: Pre-commit hooks with ruff formatting

## 🎯 Usage Example

```python
from src.etl_pipeline import ETLPipeline

# Initialize pipeline
etl = ETLPipeline()

# Configure source and target
source_config = {
    'type': 'csv',
    'path': 'data/sales_data.csv'
}

target_config = {
    'table_name': 'cleaned_sales',
    'if_exists': 'replace'
}

# Add custom transformation
def add_metadata(df):
    import pandas as pd
    df['processed_at'] = pd.Timestamp.now()
    return df

# Run pipeline
etl.run_pipeline(
    source_config=source_config,
    target_config=target_config,
    custom_transformations=[add_metadata]
)
```

## ⚙️ Configuration

Create `.env` from `.env.template`:

```bash
# Database Configuration
DB_TYPE=postgresql
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database
DB_USER=your_username
DB_PASSWORD=your_password

# Performance Settings
DB_CHUNK_SIZE=1000
LOG_LEVEL=INFO
```

## 🧪 Testing

```bash
# Run all tests
pipenv run pytest

# With coverage
pipenv run pytest --cov=src

# Specific test file
pipenv run pytest tests/test_data_cleaner.py
```

## 🔧 Extending

### Custom Transformations

```python
def your_business_logic(df):
    # Your transformation here
    df['new_column'] = df['old_column'] * 2
    return df

etl.run_pipeline(source_config, target_config, [your_business_logic])
```

### New Data Sources

Extend `etl_pipeline.py`:

```python
def extract_from_api(self, endpoint):
    import requests
    response = requests.get(endpoint)
    return pd.DataFrame(response.json())
```

Built with ❤️ by the Zoomprop Engineering Team.
