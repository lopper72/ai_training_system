"""
Sales Forecaster Module
Module dự đoán doanh số bán hàng
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from .model_trainer import ModelTrainer

logger = logging.getLogger(__name__)


class SalesForecaster:
    """Class dự đoán doanh số bán hàng"""
    
    def __init__(self, model_dir: str = "data/models"):
        """
        Khởi tạo SalesForecaster
        
        Args:
            model_dir: Thư mục lưu models
        """
        self.model_trainer = ModelTrainer(model_dir)
        self.feature_columns = [
            'year', 'month', 'quarter', 'day_of_week',
            'day_of_month', 'week_of_year', 'is_weekend',
            'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos'
        ]
        self.target_column = 'total_revenue'
    
    def prepare_forecast_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuẩn bị dữ liệu cho forecast
        
        Args:
            df: DataFrame dữ liệu sales trend
            
        Returns:
            DataFrame đã chuẩn bị
        """
        try:
            logger.info("Chuẩn bị dữ liệu forecast")
            
            df_prep = df.copy()
            
            # Đảm bảo có cột ngày
            if 'transaction_date' in df_prep.columns:
                df_prep['transaction_date'] = pd.to_datetime(df_prep['transaction_date'])
                
                # Tạo temporal features
                df_prep['year'] = df_prep['transaction_date'].dt.year
                df_prep['month'] = df_prep['transaction_date'].dt.month
                df_prep['quarter'] = df_prep['transaction_date'].dt.quarter
                df_prep['day_of_week'] = df_prep['transaction_date'].dt.dayofweek
                df_prep['day_of_month'] = df_prep['transaction_date'].dt.day
                df_prep['week_of_year'] = df_prep['transaction_date'].dt.isocalendar().week
                df_prep['is_weekend'] = df_prep['day_of_week'].isin([5, 6]).astype(int)
                
                # Cyclical features
                df_prep['month_sin'] = np.sin(2 * np.pi * df_prep['month'] / 12)
                df_prep['month_cos'] = np.cos(2 * np.pi * df_prep['month'] / 12)
                df_prep['day_of_week_sin'] = np.sin(2 * np.pi * df_prep['day_of_week'] / 7)
                df_prep['day_of_week_cos'] = np.cos(2 * np.pi * df_prep['day_of_week'] / 7)
            
            # Xử lý missing values
            for col in self.feature_columns:
                if col in df_prep.columns:
                    df_prep[col] = df_prep[col].fillna(0)
            
            if self.target_column in df_prep.columns:
                df_prep[self.target_column] = df_prep[self.target_column].fillna(0)
            
            logger.info(f"Chuẩn bị xong {len(df_prep)} samples")
            
            return df_prep
            
        except Exception as e:
            logger.error(f"Lỗi chuẩn bị dữ liệu forecast: {str(e)}")
            raise
    
    def train(
        self,
        df: pd.DataFrame,
        algorithm: str = "random_forest"
    ) -> Dict:
        """
        Training sales forecast model
        
        Args:
            df: DataFrame dữ liệu
            algorithm: Thuật toán
            
        Returns:
            Dict kết quả training
        """
        try:
            logger.info("Training sales forecast model")
            
            # Chuẩn bị features
            available_features = [col for col in self.feature_columns if col in df.columns]
            
            X, y = self.model_trainer.prepare_features(
                df,
                feature_columns=available_features,
                target_column=self.target_column,
                categorical_columns=[]
            )
            
            if y is None:
                raise ValueError(f"Target column '{self.target_column}' không tồn tại")
            
            # Training
            result = self.model_trainer.train_regressor(
                X, y,
                model_name="sales_forecaster",
                algorithm=algorithm
            )
            
            # Thêm thông tin features
            result['feature_columns'] = available_features
            result['avg_daily_sales'] = y.mean()
            
            logger.info(f"Hoàn thành training sales forecaster: R2={result['metrics']['r2']:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Lỗi training sales forecaster: {str(e)}")
            raise
    
    def forecast(
        self,
        df: pd.DataFrame,
        forecast_days: int = 30
    ) -> pd.DataFrame:
        """
        Dự đoán doanh số cho tương lai
        
        Args:
            df: DataFrame dữ liệu lịch sử
            forecast_days: Số ngày cần dự đoán
            
        Returns:
            DataFrame với forecasts
        """
        try:
            logger.info(f"Dự đoán doanh số cho {forecast_days} ngày")
            
            # Tạo future dates
            last_date = df['transaction_date'].max()
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            # Tạo DataFrame cho future
            df_future = pd.DataFrame({'transaction_date': future_dates})
            
            # Tạo temporal features
            df_future['year'] = df_future['transaction_date'].dt.year
            df_future['month'] = df_future['transaction_date'].dt.month
            df_future['quarter'] = df_future['transaction_date'].dt.quarter
            df_future['day_of_week'] = df_future['transaction_date'].dt.dayofweek
            df_future['day_of_month'] = df_future['transaction_date'].dt.day
            df_future['week_of_year'] = df_future['transaction_date'].dt.isocalendar().week
            df_future['is_weekend'] = df_future['day_of_week'].isin([5, 6]).astype(int)
            df_future['month_sin'] = np.sin(2 * np.pi * df_future['month'] / 12)
            df_future['month_cos'] = np.cos(2 * np.pi * df_future['month'] / 12)
            df_future['day_of_week_sin'] = np.sin(2 * np.pi * df_future['day_of_week'] / 7)
            df_future['day_of_week_cos'] = np.cos(2 * np.pi * df_future['day_of_week'] / 7)
            
            # Chuẩn bị features
            available_features = [col for col in self.feature_columns if col in df_future.columns]
            
            X, _ = self.model_trainer.prepare_features(
                df_future,
                feature_columns=available_features,
                categorical_columns=[]
            )
            
            # Predict
            predictions = self.model_trainer.predict("sales_forecaster", X)
            
            df_future['forecasted_revenue'] = predictions
            df_future['forecast_type'] = 'daily'
            
            logger.info(f"Dự đoán xong {len(df_future)} ngày")
            
            return df_future
            
        except Exception as e:
            logger.error(f"Lỗi dự đoán sales: {str(e)}")
            raise
    
    def get_forecast_insights(
        self,
        historical_df: pd.DataFrame,
        forecast_df: pd.DataFrame
    ) -> Dict:
        """
        Lấy insights từ forecast
        
        Args:
            historical_df: DataFrame dữ liệu lịch sử
            forecast_df: DataFrame forecasts
            
        Returns:
            Dict insights
        """
        try:
            insights = {
                'historical_avg_daily_sales': historical_df[self.target_column].mean() if self.target_column in historical_df.columns else 0,
                'forecasted_avg_daily_sales': forecast_df['forecasted_revenue'].mean(),
                'forecast_period_days': len(forecast_df),
                'total_forecasted_revenue': forecast_df['forecasted_revenue'].sum(),
                'forecast_start_date': forecast_df['transaction_date'].min().isoformat() if len(forecast_df) > 0 else None,
                'forecast_end_date': forecast_df['transaction_date'].max().isoformat() if len(forecast_df) > 0 else None
            }
            
            # Tính growth rate
            if insights['historical_avg_daily_sales'] > 0:
                insights['expected_growth_rate'] = (
                    (insights['forecasted_avg_daily_sales'] - insights['historical_avg_daily_sales']) /
                    insights['historical_avg_daily_sales'] * 100
                )
            else:
                insights['expected_growth_rate'] = 0
            
            # Weekly breakdown
            if 'transaction_date' in forecast_df.columns:
                forecast_df['week'] = forecast_df['transaction_date'].dt.isocalendar().week
                weekly_forecast = forecast_df.groupby('week')['forecasted_revenue'].sum()
                insights['weekly_forecast'] = weekly_forecast.to_dict()
            
            return insights
            
        except Exception as e:
            logger.error(f"Lỗi lấy forecast insights: {str(e)}")
            raise