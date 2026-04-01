"""
Data Transformer Module
Module chuyển đổi và làm sạch dữ liệu cho AI training
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class DataTransformer:
    """Class chuyển đổi và làm sạch dữ liệu"""
    
    def __init__(self, mapping_path: str = "config/mapping.json"):
        """
        Khởi tạo DataTransformer
        
        Args:
            mapping_path: Đường dẫn file mapping
        """
        self.mapping = self._load_mapping(mapping_path)
        self.transaction_type_mapping = self._get_transaction_type_mapping()
    
    def _load_mapping(self, mapping_path: str) -> Dict:
        """Đọc file mapping"""
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Không tìm thấy file mapping: {mapping_path}")
            return {}
    
    def _get_transaction_type_mapping(self) -> Dict:
        """Lấy mapping loại giao dịch"""
        try:
            return self.mapping.get('database_tables', {}).get('scm_sal_main', {}).get('columns', {}).get('tag_table_usage', {}).get('mapping_values', {})
        except:
            return {
                'sal_soe': 'Sales Order Entry',
                'sal_soc': 'Sales Order Confirmation',
                'sal_inv': 'Sales Invoice',
                'sal_quo': 'Sales Quotation',
                'sal_cn': 'Sales Credit Note',
                'stk_do': 'Delivery Order',
                'stk_doc': 'Delivery Order Confirmation'
            }
    
    def clean_sales_main(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Làm sạch dữ liệu scm_sal_main
        
        Args:
            df: DataFrame dữ liệu gốc
            
        Returns:
            DataFrame đã làm sạch
        """
        try:
            logger.info(f"Bắt đầu làm sạch scm_sal_main: {len(df)} records")
            
            df_clean = df.copy()
            
            # 1. Xử lý null values
            df_clean['party_desc'] = df_clean['party_desc'].fillna('Unknown')
            df_clean['party_code'] = df_clean['party_code'].fillna('N/A')
            df_clean['deptunit_desc'] = df_clean['deptunit_desc'].fillna('N/A')
            df_clean['staff_code'] = df_clean['staff_code'].fillna('N/A')
            df_clean['notes_memo'] = df_clean['notes_memo'].fillna('')
            
            # 2. Chuyển đổi kiểu dữ liệu
            df_clean['date_trans'] = pd.to_datetime(df_clean['date_trans'], errors='coerce')
            df_clean['date_due'] = pd.to_datetime(df_clean['date_due'], errors='coerce')
            
            df_clean['amount_local'] = pd.to_numeric(df_clean['amount_local'], errors='coerce').fillna(0)
            df_clean['amount_forex'] = pd.to_numeric(df_clean['amount_forex'], errors='coerce').fillna(0)
            df_clean['salestaxpct'] = pd.to_numeric(df_clean['salestaxpct'], errors='coerce').fillna(0)
            df_clean['curr_rate_forex_f_calc'] = pd.to_numeric(df_clean['curr_rate_forex_f_calc'], errors='coerce').fillna(1)
            
            # 3. Thêm cột transaction_type_name
            df_clean['transaction_type_name'] = df_clean['tag_table_usage'].map(self.transaction_type_mapping).fillna('Unknown')
            
            # 4. Thêm cột temporal features
            df_clean['year'] = df_clean['date_trans'].dt.year
            df_clean['month'] = df_clean['date_trans'].dt.month
            df_clean['quarter'] = df_clean['date_trans'].dt.quarter
            df_clean['day_of_week'] = df_clean['date_trans'].dt.dayofweek
            df_clean['day_of_month'] = df_clean['date_trans'].dt.day
            df_clean['week_of_year'] = df_clean['date_trans'].dt.isocalendar().week
            df_clean['is_weekend'] = df_clean['day_of_week'].isin([5, 6]).astype(int)
            df_clean['is_month_end'] = (df_clean['day_of_month'] >= 25).astype(int)
            
            # 5. Tính toán các chỉ số
            df_clean['amount_local_log'] = np.log1p(df_clean['amount_local'].abs())
            
            # 6. Xử lý outliers (loại bỏ giao dịch có giá trị quá lớn)
            q99 = df_clean['amount_local'].quantile(0.99)
            df_clean['is_outlier'] = (df_clean['amount_local'].abs() > q99).astype(int)
            
            logger.info(f"Hoàn thành làm sạch scm_sal_main: {len(df_clean)} records")
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Lỗi làm sạch scm_sal_main: {str(e)}")
            raise
    
    def clean_sales_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Làm sạch dữ liệu scm_sal_data
        
        Args:
            df: DataFrame dữ liệu gốc
            
        Returns:
            DataFrame đã làm sạch
        """
        try:
            logger.info(f"Bắt đầu làm sạch scm_sal_data: {len(df)} records")
            
            df_clean = df.copy()
            
            # 1. Xử lý null values
            df_clean['stkcode_desc'] = df_clean['stkcode_desc'].fillna('Unknown Product')
            df_clean['stkcode_code'] = df_clean['stkcode_code'].fillna('N/A')
            df_clean['brand_desc'] = df_clean['brand_desc'].fillna('No Brand')
            df_clean['stkcate_desc'] = df_clean['stkcate_desc'].fillna('No Category')
            df_clean['stkvendor_desc'] = df_clean['stkvendor_desc'].fillna('No Vendor')
            df_clean['notes_memo'] = df_clean['notes_memo'].fillna('')
            
            # 2. Chuyển đổi kiểu dữ liệu
            df_clean['date_trans'] = pd.to_datetime(df_clean['date_trans'], errors='coerce')
            
            numeric_cols = [
                'qnty_total', 'qnty_uomstk', 'bal_qnty_total', 'bal_qnty_uomstk',
                'price_unitlist_forex', 'price_unitlist_local',
                'price_unitrate_forex', 'price_unitrate_local',
                'discount_pct', 'amount_forex', 'amount_local',
                'amount_tax_forex', 'amount_tax_local'
            ]
            
            for col in numeric_cols:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
            
            # 3. Tính toán các chỉ số
            df_clean['line_total'] = df_clean['qnty_total'] * df_clean['price_unitrate_local']
            df_clean['discount_amount'] = df_clean['line_total'] * df_clean['discount_pct'] / 100
            df_clean['net_amount'] = df_clean['line_total'] - df_clean['discount_amount']
            
            # 4. Thêm cột temporal features
            df_clean['year'] = df_clean['date_trans'].dt.year
            df_clean['month'] = df_clean['date_trans'].dt.month
            df_clean['quarter'] = df_clean['date_trans'].dt.quarter
            
            # 5. Tạo product identifier
            df_clean['product_id'] = df_clean['stkcode_code'] + '_' + df_clean['stkcode_unique'].astype(str)
            
            logger.info(f"Hoàn thành làm sạch scm_sal_data: {len(df_clean)} records")
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Lỗi làm sạch scm_sal_data: {str(e)}")
            raise
    
    def transform_customer_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuyển đổi dữ liệu cho phân tích khách hàng
        """
        try:
            logger.info("Chuyển đổi dữ liệu phân tích khách hàng")
            
            df_clean = df.copy()
            
            df_clean['customer_name'] = df_clean['customer_name'].fillna('Unknown')
            df_clean['product_name'] = df_clean['product_name'].fillna('Unknown')
            df_clean['category_name'] = df_clean['category_name'].fillna('No Category')
            df_clean['brand_desc'] = df_clean['brand_desc'].fillna('No Brand')
            
            df_clean['transaction_date'] = pd.to_datetime(df_clean['transaction_date'], errors='coerce')
            df_clean['quantity'] = pd.to_numeric(df_clean['quantity'], errors='coerce').fillna(0)
            df_clean['unit_price'] = pd.to_numeric(df_clean['unit_price'], errors='coerce').fillna(0)
            df_clean['line_amount'] = pd.to_numeric(df_clean['line_amount'], errors='coerce').fillna(0)
            
            df_clean['year'] = df_clean['transaction_date'].dt.year
            df_clean['month'] = df_clean['transaction_date'].dt.month
            df_clean['quarter'] = df_clean['transaction_date'].dt.quarter
            df_clean['day_of_week'] = df_clean['transaction_date'].dt.dayofweek
            df_clean['is_weekend'] = df_clean['day_of_week'].isin([5, 6]).astype(int)
            
            df_clean['transaction_value_segment'] = pd.qcut(
                df_clean['line_amount'].abs(),
                q=5,
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                duplicates='drop'
            )
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Lỗi chuyển đổi dữ liệu phân tích khách hàng: {str(e)}")
            raise
    
    def transform_product_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuyển đổi dữ liệu cho phân tích sản phẩm
        """
        try:
            logger.info("Chuyển đổi dữ liệu phân tích sản phẩm")
            
            df_clean = df.copy()
            
            df_clean['product_name'] = df_clean['product_name'].fillna('Unknown')
            df_clean['category_name'] = df_clean['category_name'].fillna('No Category')
            df_clean['brand_desc'] = df_clean['brand_desc'].fillna('No Brand')
            df_clean['vendor_name'] = df_clean['vendor_name'].fillna('No Vendor')
            
            df_clean['transaction_date'] = pd.to_datetime(df_clean['transaction_date'], errors='coerce')
            df_clean['quantity_sold'] = pd.to_numeric(df_clean['quantity_sold'], errors='coerce').fillna(0)
            df_clean['revenue'] = pd.to_numeric(df_clean['revenue'], errors='coerce').fillna(0)
            df_clean['avg_price'] = pd.to_numeric(df_clean['avg_price'], errors='coerce').fillna(0)
            
            df_clean['year'] = df_clean['transaction_date'].dt.year
            df_clean['month'] = df_clean['transaction_date'].dt.month
            df_clean['quarter'] = df_clean['transaction_date'].dt.quarter
            
            df_clean['sales_velocity'] = df_clean['revenue'] / df_clean['quantity_sold'].replace(0, 1)
            
            df_clean['revenue_segment'] = pd.qcut(
                df_clean['revenue'].abs(),
                q=5,
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                duplicates='drop'
            )
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Lỗi chuyển đổi dữ liệu phân tích sản phẩm: {str(e)}")
            raise
    
    def transform_customer_retention(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuyển đổi dữ liệu cho phân tích giữ chân khách hàng
        """
        try:
            logger.info("Chuyển đổi dữ liệu phân tích giữ chân khách hàng")
            
            df_clean = df.copy()
            
            df_clean['customer_name'] = df_clean['customer_name'].fillna('Unknown')
            
            date_cols = ['first_purchase_date', 'last_purchase_date']
            for col in date_cols:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            
            numeric_cols = ['days_since_last_purchase', 'total_purchases', 'total_spent', 
                          'avg_purchase_value', 'purchase_frequency', 'customer_lifetime_days']
            for col in numeric_cols:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
            
            df_clean['recency_score'] = pd.qcut(
                df_clean['days_since_last_purchase'],
                q=5,
                labels=[5, 4, 3, 2, 1],
                duplicates='drop'
            )
            
            df_clean['frequency_score'] = pd.qcut(
                df_clean['total_purchases'].rank(method='first'),
                q=5,
                labels=[1, 2, 3, 4, 5],
                duplicates='drop'
            )
            
            df_clean['monetary_score'] = pd.qcut(
                df_clean['total_spent'].rank(method='first'),
                q=5,
                labels=[1, 2, 3, 4, 5],
                duplicates='drop'
            )
            
            df_clean['recency_score'] = pd.to_numeric(df_clean['recency_score'], errors='coerce')
            df_clean['frequency_score'] = pd.to_numeric(df_clean['frequency_score'], errors='coerce')
            df_clean['monetary_score'] = pd.to_numeric(df_clean['monetary_score'], errors='coerce')
            
            df_clean['rfm_score'] = (
                df_clean['recency_score'] + 
                df_clean['frequency_score'] + 
                df_clean['monetary_score']
            ) / 3
            
            df_clean['customer_segment'] = pd.cut(
                df_clean['rfm_score'],
                bins=[0, 2, 3, 4, 5],
                labels=['At Risk', 'Needs Attention', 'Potential Loyalist', 'Champion']
            )
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Lỗi chuyển đổi dữ liệu retention: {str(e)}")
            raise
    
    def transform_sales_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuyển đổi dữ liệu cho phân tích xu hướng bán hàng
        """
        try:
            logger.info("Chuyển đổi dữ liệu xu hướng bán hàng")
            
            df_clean = df.copy()
            
            # Xử lý null values
            df_clean['business_unit_name'] = df_clean['business_unit_name'].fillna('Unknown')
            df_clean['salesperson'] = df_clean['salesperson'].fillna('Unknown')
            df_clean['transaction_type'] = df_clean['transaction_type'].fillna('Unknown')
            
            # Chuyển đổi kiểu dữ liệu
            df_clean['transaction_date'] = pd.to_datetime(df_clean['transaction_date'], errors='coerce')
            
            numeric_cols = ['total_transactions', 'total_revenue', 'avg_transaction_value', 'unique_customers']
            for col in numeric_cols:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
            
            # Thêm temporal features
            df_clean['year'] = df_clean['transaction_date'].dt.year
            df_clean['month'] = df_clean['transaction_date'].dt.month
            df_clean['quarter'] = df_clean['transaction_date'].dt.quarter
            df_clean['day_of_week'] = df_clean['transaction_date'].dt.dayofweek
            df_clean['day_of_month'] = df_clean['transaction_date'].dt.day
            df_clean['week_of_year'] = df_clean['transaction_date'].dt.isocalendar().week
            df_clean['is_weekend'] = df_clean['day_of_week'].isin([5, 6]).astype(int)
            
            # Cyclical features
            df_clean['month_sin'] = np.sin(2 * np.pi * df_clean['month'] / 12)
            df_clean['month_cos'] = np.cos(2 * np.pi * df_clean['month'] / 12)
            df_clean['day_of_week_sin'] = np.sin(2 * np.pi * df_clean['day_of_week'] / 7)
            df_clean['day_of_week_cos'] = np.cos(2 * np.pi * df_clean['day_of_week'] / 7)
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Lỗi chuyển đổi dữ liệu xu hướng: {str(e)}")
            raise
    
    def save_transformed_data(self, df: pd.DataFrame, output_path: str, format: str = 'parquet'):
        """Lưu dữ liệu đã chuyển đổi"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'parquet':
                df.to_parquet(output_path, index=False, engine='pyarrow')
            elif format == 'csv':
                df.to_csv(output_path, index=False, encoding='utf-8')
            elif format == 'json':
                df.to_json(output_path, orient='records', force_ascii=False, indent=2)
            else:
                raise ValueError(f"Định dạng không hỗ trợ: {format}")
            
            logger.info(f"Đã lưu dữ liệu: {output_path} ({len(df)} records)")
            
        except Exception as e:
            logger.error(f"Lỗi lưu dữ liệu: {str(e)}")
            raise
    
    def load_transformed_data(self, input_path: str, format: Optional[str] = None) -> pd.DataFrame:
        """Đọc dữ liệu đã chuyển đổi"""
        try:
            if format is None:
                format = Path(input_path).suffix.lstrip('.')
            
            if format == 'parquet':
                df = pd.read_parquet(input_path)
            elif format == 'csv':
                df = pd.read_csv(input_path, encoding='utf-8')
            elif format == 'json':
                df = pd.read_json(input_path, orient='records')
            else:
                raise ValueError(f"Định dạng không hỗ trợ: {format}")
            
            logger.info(f"Đã đọc dữ liệu: {input_path} ({len(df)} records)")
            
            return df
            
        except Exception as e:
            logger.error(f"Lỗi đọc dữ liệu: {str(e)}")
            raise
