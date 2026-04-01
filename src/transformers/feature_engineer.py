"""
Feature Engineer Module
Module tạo features cho ML models
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Class tạo features cho ML"""
    
    def __init__(self):
        """Khởi tạo FeatureEngineer"""
        pass
    
    def create_rfm_features(self, df: pd.DataFrame, customer_col: str = 'customer_id') -> pd.DataFrame:
        """
        Tạo features RFM (Recency, Frequency, Monetary)
        
        Args:
            df: DataFrame dữ liệu giao dịch
            customer_col: Tên cột customer ID
            
        Returns:
            DataFrame với RFM features
        """
        try:
            logger.info("Tạo RFM features")
            
            # Tính ngày hiện tại
            current_date = df['transaction_date'].max()
            
            # Group by customer
            rfm = df.groupby(customer_col).agg({
                'transaction_date': lambda x: (current_date - x.max()).days,  # Recency
                'uniquenum_pri': 'nunique',  # Frequency
                'line_amount': 'sum'  # Monetary
            }).reset_index()
            
            rfm.columns = [customer_col, 'recency', 'frequency', 'monetary']
            
            # Tính RFM scores
            rfm['recency_score'] = pd.qcut(rfm['recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
            rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            rfm['monetary_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            
            # Chuyển sang numeric
            rfm['recency_score'] = pd.to_numeric(rfm['recency_score'], errors='coerce')
            rfm['frequency_score'] = pd.to_numeric(rfm['frequency_score'], errors='coerce')
            rfm['monetary_score'] = pd.to_numeric(rfm['monetary_score'], errors='coerce')
            
            # Tính RFM tổng hợp
            rfm['rfm_score'] = (rfm['recency_score'] + rfm['frequency_score'] + rfm['monetary_score']) / 3
            
            # Phân khúc khách hàng
            rfm['customer_segment'] = pd.cut(
                rfm['rfm_score'],
                bins=[0, 2, 3, 4, 5],
                labels=['At Risk', 'Needs Attention', 'Potential Loyalist', 'Champion']
            )
            
            return rfm
            
        except Exception as e:
            logger.error(f"Lỗi tạo RFM features: {str(e)}")
            raise
    
    def create_temporal_features(self, df: pd.DataFrame, date_col: str = 'transaction_date') -> pd.DataFrame:
        """
        Tạo temporal features từ ngày giao dịch
        
        Args:
            df: DataFrame dữ liệu
            date_col: Tên cột ngày
            
        Returns:
            DataFrame với temporal features
        """
        try:
            logger.info("Tạo temporal features")
            
            df_features = df.copy()
            
            # Đảm bảo cột ngày là datetime
            df_features[date_col] = pd.to_datetime(df_features[date_col], errors='coerce')
            
            # Tạo các features
            df_features['year'] = df_features[date_col].dt.year
            df_features['month'] = df_features[date_col].dt.month
            df_features['quarter'] = df_features[date_col].dt.quarter
            df_features['day_of_week'] = df_features[date_col].dt.dayofweek
            df_features['day_of_month'] = df_features[date_col].dt.day
            df_features['day_of_year'] = df_features[date_col].dt.dayofyear
            df_features['week_of_year'] = df_features[date_col].dt.isocalendar().week
            df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
            df_features['is_month_start'] = (df_features[date_col].dt.is_month_start).astype(int)
            df_features['is_month_end'] = (df_features[date_col].dt.is_month_end).astype(int)
            df_features['is_quarter_start'] = (df_features[date_col].dt.is_quarter_start).astype(int)
            df_features['is_quarter_end'] = (df_features[date_col].dt.is_quarter_end).astype(int)
            df_features['is_year_start'] = (df_features[date_col].dt.is_year_start).astype(int)
            df_features['is_year_end'] = (df_features[date_col].dt.is_year_end).astype(int)
            
            # Tạo cyclical features (cho ML models)
            df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
            df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
            df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
            df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
            
            return df_features
            
        except Exception as e:
            logger.error(f"Lỗi tạo temporal features: {str(e)}")
            raise
    
    def create_product_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo features cho sản phẩm
        
        Args:
            df: DataFrame dữ liệu sản phẩm
            
        Returns:
            DataFrame với product features
        """
        try:
            logger.info("Tạo product features")
            
            # Group by product
            product_features = df.groupby('product_code').agg({
                'quantity_sold': ['sum', 'mean', 'std'],
                'revenue': ['sum', 'mean', 'std'],
                'avg_price': 'mean',
                'unique_customers': 'sum',
                'num_transactions': 'sum'
            }).reset_index()
            
            # Flatten column names
            product_features.columns = [
                'product_code',
                'total_quantity_sold',
                'avg_quantity_per_transaction',
                'std_quantity',
                'total_revenue',
                'avg_revenue_per_transaction',
                'std_revenue',
                'avg_selling_price',
                'total_unique_customers',
                'total_transactions'
            ]
            
            # Tính sales velocity (doanh số/ngày)
            date_range = (df['transaction_date'].max() - df['transaction_date'].min()).days
            product_features['sales_velocity'] = product_features['total_revenue'] / max(date_range, 1)
            
            # Tính customer reach
            product_features['customer_reach'] = product_features['total_unique_customers'] / product_features['total_transactions']
            
            return product_features
            
        except Exception as e:
            logger.error(f"Lỗi tạo product features: {str(e)}")
            raise
    
    def create_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo features cho khách hàng
        
        Args:
            df: DataFrame dữ liệu khách hàng
            
        Returns:
            DataFrame với customer features
        """
        try:
            logger.info("Tạo customer features")
            
            # Group by customer
            customer_features = df.groupby('customer_id').agg({
                'transaction_date': ['min', 'max', 'nunique'],
                'line_amount': ['sum', 'mean', 'std'],
                'quantity': ['sum', 'mean'],
                'product_code': 'nunique',
                'category_code': 'nunique',
                'brand_code': 'nunique'
            }).reset_index()
            
            # Flatten column names
            customer_features.columns = [
                'customer_id',
                'first_purchase_date',
                'last_purchase_date',
                'total_purchase_days',
                'total_spent',
                'avg_order_value',
                'std_order_value',
                'total_quantity',
                'avg_quantity_per_order',
                'unique_products',
                'unique_categories',
                'unique_brands'
            ]
            
            # Tính customer lifetime
            customer_features['customer_lifetime_days'] = (
                customer_features['last_purchase_date'] - customer_features['first_purchase_date']
            ).dt.days
            
            # Tính purchase frequency
            customer_features['purchase_frequency'] = (
                customer_features['total_purchase_days'] / 
                customer_features['customer_lifetime_days'].replace(0, 1)
            )
            
            # Tính days since last purchase
            current_date = df['transaction_date'].max()
            customer_features['days_since_last_purchase'] = (
                current_date - customer_features['last_purchase_date']
            ).dt.days
            
            # Tính churn risk
            customer_features['churn_risk'] = pd.cut(
                customer_features['days_since_last_purchase'],
                bins=[0, 30, 60, 90, float('inf')],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
            
            return customer_features
            
        except Exception as e:
            logger.error(f"Lỗi tạo customer features: {str(e)}")
            raise
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        value_col: str,
        date_col: str = 'transaction_date',
        group_col: Optional[str] = None,
        lags: List[int] = [1, 7, 30]
    ) -> pd.DataFrame:
        """
        Tạo lag features cho time series
        
        Args:
            df: DataFrame dữ liệu
            value_col: Tên cột giá trị
            date_col: Tên cột ngày
            group_col: Tên cột group (nếu có)
            lags: Danh sách các lag periods
            
        Returns:
            DataFrame với lag features
        """
        try:
            logger.info(f"Tạo lag features cho {value_col}")
            
            df_features = df.copy()
            df_features = df_features.sort_values(date_col)
            
            if group_col:
                for lag in lags:
                    df_features[f'{value_col}_lag_{lag}'] = df_features.groupby(group_col)[value_col].shift(lag)
            else:
                for lag in lags:
                    df_features[f'{value_col}_lag_{lag}'] = df_features[value_col].shift(lag)
            
            return df_features
            
        except Exception as e:
            logger.error(f"Lỗi tạo lag features: {str(e)}")
            raise
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        value_col: str,
        date_col: str = 'transaction_date',
        group_col: Optional[str] = None,
        windows: List[int] = [7, 30, 90]
    ) -> pd.DataFrame:
        """
        Tạo rolling features cho time series
        
        Args:
            df: DataFrame dữ liệu
            value_col: Tên cột giá trị
            date_col: Tên cột ngày
            group_col: Tên cột group (nếu có)
            windows: Danh sách các window sizes
            
        Returns:
            DataFrame với rolling features
        """
        try:
            logger.info(f"Tạo rolling features cho {value_col}")
            
            df_features = df.copy()
            df_features = df_features.sort_values(date_col)
            
            for window in windows:
                if group_col:
                    df_features[f'{value_col}_rolling_mean_{window}'] = df_features.groupby(group_col)[value_col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                    df_features[f'{value_col}_rolling_std_{window}'] = df_features.groupby(group_col)[value_col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std()
                    )
                else:
                    df_features[f'{value_col}_rolling_mean_{window}'] = df_features[value_col].rolling(window=window, min_periods=1).mean()
                    df_features[f'{value_col}_rolling_std_{window}'] = df_features[value_col].rolling(window=window, min_periods=1).std()
            
            return df_features
            
        except Exception as e:
            logger.error(f"Lỗi tạo rolling features: {str(e)}")
            raise
    
    def create_interaction_features(
        self,
        df: pd.DataFrame,
        feature_pairs: List[tuple]
    ) -> pd.DataFrame:
        """
        Tạo interaction features giữa các features
        
        Args:
            df: DataFrame dữ liệu
            feature_pairs: Danh sách các cặp features [(feat1, feat2), ...]
            
        Returns:
            DataFrame với interaction features
        """
        try:
            logger.info("Tạo interaction features")
            
            df_features = df.copy()
            
            for feat1, feat2 in feature_pairs:
                if feat1 in df_features.columns and feat2 in df_features.columns:
                    # Multiplication interaction
                    df_features[f'{feat1}_x_{feat2}'] = df_features[feat1] * df_features[feat2]
                    
                    # Ratio interaction (tránh chia cho 0)
                    df_features[f'{feat1}_div_{feat2}'] = df_features[feat1] / df_features[feat2].replace(0, np.nan)
            
            return df_features
            
        except Exception as e:
            logger.error(f"Lỗi tạo interaction features: {str(e)}")
            raise