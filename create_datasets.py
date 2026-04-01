#!/usr/bin/env python
"""
Script tạo các dataset cần thiết cho AI Query Interface
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sales_data():
    """Đọc dữ liệu sales"""
    try:
        df_main = pd.read_parquet('data/processed/sales_main.parquet')
        df_data = pd.read_parquet('data/processed/sales_data.parquet')
        logger.info(f"Loaded sales_main: {len(df_main)} records")
        logger.info(f"Loaded sales_data: {len(df_data)} records")
        return df_main, df_data
    except Exception as e:
        logger.error(f"Lỗi đọc dữ liệu: {str(e)}")
        return None, None

def create_customer_analysis(df_main, df_data):
    """Tạo dataset customer_analysis"""
    try:
        # Merge dữ liệu
        df = df_data.merge(df_main, on='uniquenum_pri', how='left')
        
        # Tính toán metrics cho khách hàng (sử dụng cột từ sales_data)
        customer_metrics = df.groupby('party_code_x').agg({
            'amount_local_x': ['sum', 'mean', 'count'],
            'date_trans_x': 'nunique',
            'stkcode_code': 'nunique'
        }).reset_index()
        
        # Đặt tên cột
        customer_metrics.columns = [
            'customer_id',
            'total_revenue',
            'avg_order_value',
            'total_transactions',
            'unique_transaction_days',
            'unique_products'
        ]
        
        # Thêm thông tin khác
        customer_metrics['customer_name'] = 'Customer_' + customer_metrics['customer_id'].astype(str)
        customer_metrics['customer_segment'] = pd.cut(
            customer_metrics['total_revenue'],
            bins=[0, 1000000, 5000000, float('inf')],
            labels=['Bronze', 'Silver', 'Gold']
        )
        
        # Lưu dataset
        customer_metrics.to_parquet('data/processed/customer_analysis.parquet', index=False)
        logger.info(f"Created customer_analysis: {len(customer_metrics)} customers")
        
        return customer_metrics
        
    except Exception as e:
        logger.error(f"Lỗi tạo customer_analysis: {str(e)}")
        return None

def create_product_analysis(df_main, df_data):
    """Tạo dataset product_analysis"""
    try:
        # Merge dữ liệu
        df = df_data.merge(df_main, on='uniquenum_pri', how='left', suffixes=('_data', '_main'))
        
        # Tìm cột amount_local sau merge
        amount_col = None
        for col in ['amount_local_data', 'amount_local']:
            if col in df.columns:
                amount_col = col
                break
        
        if amount_col is None:
            logger.error("Không tìm thấy cột amount_local sau merge")
            return None
        
        # Tìm cột party_code sau merge
        party_col = None
        for col in ['party_code_data', 'party_code_main', 'party_code']:
            if col in df.columns:
                party_col = col
                break
        
        if party_col is None:
            logger.error("Không tìm thấy cột party_code sau merge")
            return None
        
        # Tính toán metrics cho sản phẩm
        product_metrics = df.groupby('stkcode_code').agg({
            amount_col: ['sum', 'mean', 'count'],
            'qnty_total': 'sum',
            party_col: 'nunique'
        }).reset_index()
        
        # Đặt tên cột
        product_metrics.columns = [
            'product_code',
            'revenue',
            'avg_price',
            'total_transactions',
            'quantity_sold',
            'unique_customers'
        ]
        
        # Thêm thông tin khác
        product_metrics['product_name'] = 'Product_' + product_metrics['product_code'].astype(str)
        product_metrics['category_name'] = 'Category_' + (pd.to_numeric(product_metrics['product_code'], errors='coerce') % 10).astype(str)
        product_metrics['brand_desc'] = 'Brand_' + (pd.to_numeric(product_metrics['product_code'], errors='coerce') % 5).astype(str)
        
        # Lưu dataset
        product_metrics.to_parquet('data/processed/product_analysis.parquet', index=False)
        logger.info(f"Created product_analysis: {len(product_metrics)} products")
        
        return product_metrics
        
    except Exception as e:
        logger.error(f"Lỗi tạo product_analysis: {str(e)}")
        return None

def create_sales_trend(df_main, df_data):
    """Tạo dataset sales_trend"""
    try:
        # Merge dữ liệu
        df = df_data.merge(df_main, on='uniquenum_pri', how='left')
        
        # Chuyển đổi ngày
        df['date_trans_x'] = pd.to_datetime(df['date_trans_x'])
        df['month'] = df['date_trans_x'].dt.month
        df['quarter'] = df['date_trans_x'].dt.quarter
        df['day_of_week'] = df['date_trans_x'].dt.dayofweek
        
        # Tính toán xu hướng
        trend_metrics = df.groupby(['month', 'quarter', 'day_of_week']).agg({
            'amount_local_x': 'sum',
            'uniquenum_pri': 'nunique'
        }).reset_index()
        
        # Đặt tên cột
        trend_metrics.columns = [
            'month',
            'quarter',
            'day_of_week',
            'total_revenue',
            'total_transactions'
        ]
        
        # Tính tăng trưởng
        trend_metrics['revenue_growth'] = trend_metrics['total_revenue'].pct_change()
        
        # Lưu dataset
        trend_metrics.to_parquet('data/processed/sales_trend.parquet', index=False)
        logger.info(f"Created sales_trend: {len(trend_metrics)} records")
        
        return trend_metrics
        
    except Exception as e:
        logger.error(f"Lỗi tạo sales_trend: {str(e)}")
        return None

def create_customer_retention(df_main, df_data):
    """Tạo dataset customer_retention"""
    try:
        # Merge dữ liệu
        df = df_data.merge(df_main, on='uniquenum_pri', how='left')
        
        # Tính toán retention
        customer_retention = df.groupby('party_code_x').agg({
            'date_trans_x': ['min', 'max', 'nunique'],
            'amount_local_x': 'sum'
        }).reset_index()
        
        # Đặt tên cột
        customer_retention.columns = [
            'customer_id',
            'first_purchase',
            'last_purchase',
            'total_purchases',
            'total_spent'
        ]
        
        # Tính churn (khách hàng không mua trong 90 ngày gần đây)
        customer_retention['days_since_last_purchase'] = (
            pd.to_datetime('today') - pd.to_datetime(customer_retention['last_purchase'])
        ).dt.days
        
        customer_retention['churn'] = (customer_retention['days_since_last_purchase'] > 90).astype(int)
        customer_retention['churn_risk'] = pd.cut(
            customer_retention['days_since_last_purchase'],
            bins=[0, 30, 60, 90, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # Lưu dataset
        customer_retention.to_parquet('data/processed/customer_retention.parquet', index=False)
        logger.info(f"Created customer_retention: {len(customer_retention)} customers")
        
        return customer_retention
        
    except Exception as e:
        logger.error(f"Lỗi tạo customer_retention: {str(e)}")
        return None

def main():
    """Hàm chính"""
    logger.info("Bắt đầu tạo datasets...")
    
    # Đọc dữ liệu
    df_main, df_data = load_sales_data()
    
    if df_main is None or df_data is None:
        logger.error("Không thể đọc dữ liệu")
        return
    
    # Tạo các datasets
    create_customer_analysis(df_main, df_data)
    create_product_analysis(df_main, df_data)
    create_sales_trend(df_main, df_data)
    create_customer_retention(df_main, df_data)
    
    logger.info("Hoàn thành tạo datasets!")

if __name__ == "__main__":
    main()