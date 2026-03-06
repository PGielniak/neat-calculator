# database.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, inspect
from sqlalchemy.orm import sessionmaker, Session
import sqlite3
from datetime import datetime
import os
from dotenv import load_dotenv



# Abstract Base Class (Interface)
class DatabaseEngine(ABC):
    """Abstract base class for database operations."""
    
    @abstractmethod
    def save_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append') -> None:
        """Save DataFrame to database table."""
        pass
    
    @abstractmethod
    def save_record(self, record: Dict[str, Any], table_name: str) -> None:
        """Save a single record to database table."""
        pass
    
    @abstractmethod
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists."""
        pass
    
    @abstractmethod
    def get_records(self, table_name: str, filters: Optional[Dict[str, Any]] = None, columns: Optional[list] = None) -> pd.DataFrame:
        """Retrieve records from table."""
        pass
    
    @abstractmethod
    def update_record(self, table_name: str, primary_key: str, key_value: Any, updates: dict) -> None:
        """Update a record in the database."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close database connection."""
        pass


# SQLite Implementation
class SQLiteEngine(DatabaseEngine):
    """SQLite database engine implementation."""
    
    def __init__(self, db_path: str = "sensor_features.db"):
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
        self.Session = sessionmaker(bind=self.engine)
    
    def save_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append') -> None:
        """Save DataFrame to SQLite table."""
        df.to_sql(
            name=table_name,
            con=self.engine,
            if_exists=if_exists,
            index=False
        )
    
    def save_record(self, record: Dict[str, Any], table_name: str) -> None:
        """Save single record to SQLite table."""
        df = pd.DataFrame([record])
        self.save_dataframe(df, table_name, if_exists='append')
    
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists in SQLite."""
        inspector = inspect(self.engine)
        return table_name in inspector.get_table_names()
    
    def get_records(self, table_name: str, filters: Optional[Dict[str, Any]] = None, columns: Optional[list] = None) -> pd.DataFrame:
        """Retrieve records from SQLite table."""
        if not columns:
            query = f"SELECT * FROM {table_name}"
        else:
            cols = ", ".join(columns)
            query = f"SELECT {cols} FROM {table_name}"
        
        if filters:
            conditions = " AND ".join([f"{k} = :{k}" for k in filters.keys()])
            query += f" WHERE {conditions}"
        
        return pd.read_sql(query, self.engine, params=filters or {})
    
    def update_record(self, table_name: str, primary_key: str, key_value: Any, updates: dict) -> None:
        """Update a record in the database."""        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Build UPDATE query
            set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
            query = f"UPDATE {table_name} SET {set_clause} WHERE {primary_key} = ?"
            
            values = list(updates.values()) + [key_value]
            cursor.execute(query, values)
            conn.commit()
            
            if cursor.rowcount == 0:
                raise ValueError(f"No record found with {primary_key}={key_value}")
                
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()
    
    def close(self) -> None:
        """Close SQLite connection."""
        self.engine.dispose()


# PostgreSQL Implementation
class PostgreSQLEngine(DatabaseEngine):
    """PostgreSQL database engine implementation."""
    
    def __init__(self, host: str, port: int, database: str, user: str, password: str, ssl_mode: str = 'require'):
        connection_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}?sslmode={ssl_mode}"
        self.engine = create_engine(connection_string,
                                    pool_size=10,
                                    max_overflow=20,
                                    pool_timeout=30,
                                    pool_recycle=1800)
        self.Session = sessionmaker(bind=self.engine)
    
    def save_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append') -> None:
        """Save DataFrame to PostgreSQL table."""
        df.to_sql(
            name=table_name,
            con=self.engine,
            if_exists=if_exists,
            index=False
        )
    
    def save_record(self, record: Dict[str, Any], table_name: str) -> None:
        """Save single record to PostgreSQL table."""
        df = pd.DataFrame([record])
        self.save_dataframe(df, table_name, if_exists='append')
    
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists in PostgreSQL."""
        inspector = inspect(self.engine)
        return table_name in inspector.get_table_names()
    
    def get_records(self, table_name: str, filters: Optional[Dict[str, Any]] = None, columns: Optional[list] = None) -> pd.DataFrame:
        """Retrieve records from PostgreSQL table."""
        
        if not columns:
            query = f"SELECT * FROM {table_name}"
        else:
            # Quote column names to handle special characters like hyphens and parentheses
            quoted_cols = ", ".join([f'"{col}"' for col in columns])
            query = f"SELECT {quoted_cols} FROM {table_name}"
        
        if filters:
            # Quote filter column names as well
            conditions = " AND ".join([f'"{k}" = %({k})s' for k in filters.keys()])
            query += f" WHERE {conditions}"
        records = pd.read_sql(query, self.engine, params=filters or {})
        return records
    
    def update_record(self, table_name: str, primary_key: str, key_value: Any, updates: dict) -> None:
        """Update a record in the database."""
        from sqlalchemy import text
        
        with self.engine.connect() as conn:
            try:
                # Build UPDATE query
                set_clause = ", ".join([f"{k} = :{k}" for k in updates.keys()])
                query = f"UPDATE {table_name} SET {set_clause} WHERE {primary_key} = :key_value"
                
                # Combine updates dict with key_value
                params = {**updates, 'key_value': key_value}
                
                result = conn.execute(text(query), params)
                conn.commit()
                
                if result.rowcount == 0:
                    raise ValueError(f"No record found with {primary_key}={key_value}")
                    
            except Exception as e:
                conn.rollback()
                raise e
    
    def close(self) -> None:
        """Close PostgreSQL connection."""
        self.engine.dispose()


# Factory Pattern for creating database engines
class DatabaseFactory:
    """Factory for creating database engine instances."""
    
    @staticmethod
    def create_engine(db_type: str, **config) -> DatabaseEngine:
        """
        Create database engine based on type.
        
        Args:
            db_type: Type of database ('sqlite' or 'postgresql')
            **config: Configuration parameters for the database
        
        Returns:
            DatabaseEngine instance
        """
        if db_type == 'sqlite':
            db_path = config.get('db_path', 'sensor_features.db')
            return SQLiteEngine(db_path=db_path)
        
        elif db_type == 'postgresql':
            return PostgreSQLEngine(
                host=config.get('db_host', 'localhost'),
                port=config.get('db_port', 5432),
                database=config.get('db_name', 'sensor_data'),
                user=config.get('db_user', 'postgres'),
                password=config.get('db_password', ''),
                ssl_mode=config.get('ssl_mode', 'require')
            )
        
        else:
            raise ValueError(f"Unsupported database type: {db_type}")


# Generic Database Repository
class DatabaseRepository:
    """Generic repository for database operations with dependency injection."""
    
    def __init__(self, engine: DatabaseEngine):
        """
        Initialize repository with database engine.
        
        Args:
            engine: Database engine implementation (SQLite, PostgreSQL, etc.)
        """
        self.engine = engine
    
    def save_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append') -> None:
        """Save DataFrame to database."""
        self.engine.save_dataframe(df, table_name, if_exists)
    
    def save_record(self, record: Dict[str, Any], table_name: str) -> None:
        """Save single record to database."""
        self.engine.save_record(record, table_name)
    
    def save_records(self, records: List[Dict[str, Any]], table_name: str) -> None:
        """Save multiple records to database."""
        df = pd.DataFrame(records)
        self.save_dataframe(df, table_name)
    
    def get_records(self, table_name: str, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Get records from database."""
        return self.engine.get_records(table_name, filters)
    
    def update_record(self, table_name: str, primary_key: str, key_value: Any, updates: dict) -> None:
        """Update a record in the database."""
        self.engine.update_record(table_name, primary_key, key_value, updates)
    
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists."""
        return self.engine.table_exists(table_name)
    
    def save_orm_object(self, obj) -> None:
        """Save a SQLAlchemy ORM model instance using a session."""
        with self.engine.Session() as session:
            session.add(obj)
            session.commit()

    def get_orm_object(self, model, **filters):
        """Retrieve a single SQLAlchemy ORM model instance matching the given filters."""
        with self.engine.Session() as session:
            return session.query(model).filter_by(**filters).first()

    def close(self) -> None:
        """Close database connection."""
        self.engine.close()
        
        
def get_postgres_db_engine() -> DatabaseEngine:
    load_dotenv()
    db_type = 'postgresql'
    db_user = os.getenv('DATABASE_USER')
    db_password = os.getenv('DATABASE_PASSWORD')
    db_host = os.getenv('DATABASE_URL', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DATABASE_NAME', 'training_data_labeled')
    ssl_mode = os.getenv('DB_SSL_MODE', 'require')
    database_engine = DatabaseFactory.create_engine(
        db_type=db_type,
        db_user=db_user,
        db_password=db_password,
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
        ssl_mode=ssl_mode
    )
    return database_engine