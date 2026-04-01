"""
Database Extractor Module
Module trích xuất dữ liệu từ database PostgreSQL
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
    """Class trích xuất dữ liệu từ database"""
    
    def __init__(self, config_path: str = "config/database.json"):
        """
        Khởi tạo DatabaseExtractor
        
        Args:
            config_path: Đường dẫn file cấu hình database
        """
        # Xử lý đường dẫn tương đối từ thư mục ai_training_system
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if not os.path.isabs(config_path):
            config_path = os.path.join(base_dir, config_path)
        
        self.config = self._load_config(config_path)
        self.engine = None
        self.session = None
        self._connect()
    
    def _load_config(self, config_path: str) -> Dict:
        """Đọc file cấu hình"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Không tìm thấy file cấu hình: {config_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Lỗi đọc file JSON: {config_path}")
            raise
    
    def _connect(self):
        """Kết nối đến database"""
        try:
            # Hỗ trợ cả 'database' và 'source_database'
            db_config = self.config.get('source_database') or self.config.get('database')
            
            # Tạo connection string
            connection_string = (
                f"postgresql://{db_config['username']}:{db_config['password']}"
                f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )
            
            # Tạo engine
            self.engine = create_engine(
                connection_string,
                pool_size=db_config.get('pool_size', 10),
                max_overflow=db_config.get('max_overflow', 20),
                pool_timeout=db_config.get('pool_timeout', 30),
                echo=db_config.get('echo', False)
            )
            
            # Tạo session
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            
            logger.info("Kết nối database thành công")
            
        except Exception as e:
            logger.error(f"Lỗi kết nối database: {str(e)}")
            raise
    
    def extract_data(
        self,
        query: str,
        params: Optional[Dict] = None,
        chunk_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Trích xuất dữ liệu bằng SQL query
        
        Args:
            query: SQL query
            params: Tham số cho query
            chunk_size: Kích thước chunk để đọc dữ liệu lớn
            
        Returns:
            DataFrame chứa dữ liệu
        """
        try:
            if chunk_size:
                # Đọc theo chunks cho dữ liệu lớn
                chunks = []
                for chunk in pd.read_sql(
                    text(query),
                    self.engine,
                    params=params,
                    chunksize=chunk_size
                ):
                    chunks.append(chunk)
                    logger.info(f"Đã đọc {len(chunk)} records")
                
                df = pd.concat(chunks, ignore_index=True)
            else:
                # Đọc toàn bộ
                df = pd.read_sql(text(query), self.engine, params=params)
            
            logger.info(f"Trích xuất thành công {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Lỗi trích xuất dữ liệu: {str(e)}")
            raise
    
    def extract_table(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Trích xuất dữ liệu từ bảng
        
        Args:
            table_name: Tên bảng
            columns: Danh sách cột cần lấy
            filters: Điều kiện lọc
            limit: Giới hạn số lượng records
            
        Returns:
            DataFrame chứa dữ liệu
        """
        try:
            # Xây dựng query
            if columns:
                cols = ", ".join(columns)
            else:
                cols = "*"
            
            query = f"SELECT {cols} FROM {table_name}"
            
            # Thêm điều kiện lọc
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
            
            # Thêm limit
            if limit:
                query += f" LIMIT {limit}"
            
            logger.info(f"Query: {query}")
            
            return self.extract_data(query, params)
            
        except Exception as e:
            logger.error(f"Lỗi trích xuất bảng {table_name}: {str(e)}")
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
        Trích xuất dữ liệu với JOIN
        
        Args:
            main_table: Bảng chính
            join_table: Bảng join
            join_condition: Điều kiện join
            columns: Danh sách cột
            filters: Điều kiện lọc
            limit: Giới hạn số lượng
            
        Returns:
            DataFrame chứa dữ liệu
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
            
            # Thêm điều kiện lọc
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
            logger.error(f"Lỗi trích xuất với JOIN: {str(e)}")
            raise
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> Any:
        """
        Thực thi query bất kỳ
        
        Args:
            query: SQL query
            params: Tham số
            
        Returns:
            Kết quả query
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                return result.fetchall()
        except Exception as e:
            logger.error(f"Lỗi thực thi query: {str(e)}")
            raise
    
    def get_table_info(self, table_name: str) -> Dict:
        """
        Lấy thông tin về bảng
        
        Args:
            table_name: Tên bảng
            
        Returns:
            Dict chứa thông tin bảng
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
            logger.error(f"Lỗi lấy thông tin bảng {table_name}: {str(e)}")
            raise
    
    def close(self):
        """Đóng kết nối"""
        if self.session:
            self.session.close()
        if self.engine:
            self.engine.dispose()
        logger.info("Đã đóng kết nối database")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()