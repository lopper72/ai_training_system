"""
Database Extractor Module
Module for extracting data from PostgreSQL database
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os

logger = logging.getLogger(__name__)


class DatabaseExtractor:
    """Class for extracting data from database"""
    
    def __init__(self, config_path: str = "config/database.json"):
        """
        Initialize DatabaseExtractor
        
        Args:
            config_path: Path to database configuration file
        """
        # Handle relative path from ai_training_system directory
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if not os.path.isabs(config_path):
            config_path = os.path.join(base_dir, config_path)
        
        self.config = self._load_config(config_path)
        self.engine = None
        self.session = None
        self._connect()
    
    def _load_config(self, config_path: str) -> Dict:
        """Read configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Error reading JSON file: {config_path}")
            raise
    
    def _connect(self):
        """Connect to database"""
        try:
            # Support both 'database' and 'source_database'
            db_config = self.config.get('source_database') or self.config.get('database')
            
            # Create connection string
            connection_string = (
                f"postgresql://{db_config['username']}:{db_config['password']}"
                f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )
            
            # Create engine
            self.engine = create_engine(
                connection_string,
                pool_size=db_config.get('pool_size', 10),
                max_overflow=db_config.get('max_overflow', 20),
                pool_timeout=db_config.get('pool_timeout', 30),
                echo=db_config.get('echo', False)
            )
            
            # Create session
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            
            logger.info("Database connection successful")
            
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            raise
    
    def extract_data(
        self,
        query: str,
        params: Optional[Dict] = None,
        chunk_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extract data using SQL query
        
        Args:
            query: SQL query
            params: Query parameters
            chunk_size: Chunk size for reading large data
            
        Returns:
            DataFrame containing data
        """
        try:
            if chunk_size:
                # Read in chunks for large data
                chunks = []
                for chunk in pd.read_sql(
                    text(query),
                    self.engine,
                    params=params,
                    chunksize=chunk_size
                ):
                    chunks.append(chunk)
                    logger.info(f"Read {len(chunk)} records")
                
                df = pd.concat(chunks, ignore_index=True)
            else:
                # Read all
                df = pd.read_sql(text(query), self.engine, params=params)
            
            logger.info(f"Successfully extracted {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Data extraction error: {str(e)}")
            raise
    
    def extract_table(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extract data from table
        
        Args:
            table_name: Table name
            columns: List of columns to retrieve
            filters: Filter conditions
            limit: Record limit
            
        Returns:
            DataFrame containing data
        """
        try:
            # Build query
            if columns:
                cols = ", ".join(columns)
            else:
                cols = "*"
            
            query = f"SELECT {cols} FROM {table_name}"
            
            # Add filter conditions
            if filters:
                conditions = []
                params = {}
                for key, value in filters.items():
                    if isinstance(value, list):
                        placeholders = ", ".join([f":{key}_{i}" for i in range(len(value))])
                        conditions.append(f"{key} IN ({placeholders})")
                        for i, v in enumerate(value):
                            params[f"{key}_{i}"] = v
                    else:
                        conditions.append(f"{key} = :{key}")
                        params[key] = value
                
                query += " WHERE " + " AND ".join(conditions)
            else:
                params = {}
            
            # Add limit
            if limit:
                query += f" LIMIT {limit}"
            
            logger.info(f"Query: {query}")
            
            return self.extract_data(query, params)
            
        except Exception as e:
            logger.error(f"Error extracting table {table_name}: {str(e)}")
            raise
    
    def extract_with_join(
        self,
        main_table: str,
        join_table: str,
        join_condition: str,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extract data with JOIN
        
        Args:
            main_table: Main table
            join_table: Join table
            join_condition: Join condition
            columns: List of columns
            filters: Filter conditions
            limit: Record limit
            
        Returns:
            DataFrame containing data
        """
        try:
            if columns:
                cols = ", ".join(columns)
            else:
                cols = "*"
            
            query = f"""
                SELECT {cols}
                FROM {main_table}
                INNER JOIN {join_table} ON {join_condition}
            """
            
            # Add filter conditions
            if filters:
                conditions = []
                params = {}
                for key, value in filters.items():
                    if isinstance(value, list):
                        placeholders = ", ".join([f":{key}_{i}" for i in range(len(value))])
                        conditions.append(f"{key} IN ({placeholders})")
                        for i, v in enumerate(value):
                            params[f"{key}_{i}"] = v
                    else:
                        conditions.append(f"{key} = :{key}")
                        params[key] = value
                
                query += " WHERE " + " AND ".join(conditions)
            else:
                params = {}
            
            if limit:
                query += f" LIMIT {limit}"
            
            return self.extract_data(query, params)
            
        except Exception as e:
            logger.error(f"Error extracting with JOIN: {str(e)}")
            raise
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> Any:
        """
        Execute any query
        
        Args:
            query: SQL query
            params: Parameters
            
        Returns:
            Query result
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                return result.fetchall()
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise
    
    def get_table_info(self, table_name: str) -> Dict:
        """
        Get table information
        
        Args:
            table_name: Table name
            
        Returns:
            Dict containing table information
        """
        try:
            query = """
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default
                FROM information_schema.columns
                WHERE table_name = :table_name
                ORDER BY ordinal_position
            """
            
            result = self.execute_query(query, {"table_name": table_name})
            
            columns = []
            for row in result:
                columns.append({
                    "name": row[0],
                    "type": row[1],
                    "nullable": row[2] == "YES",
                    "default": row[3]
                })
            
            return {
                "table_name": table_name,
                "columns": columns,
                "column_count": len(columns)
            }
            
        except Exception as e:
            logger.error(f"Error getting table info {table_name}: {str(e)}")
            raise
    
    def close(self):
        """Close connection"""
        if self.session:
            self.session.close()
        if self.engine:
            self.engine.dispose()
        logger.info("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()