"""
Script to create necessary datasets for AI Query Interface
"""

import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sales_data():
    """Read sales data"""
    try:
        df_main = pd.read_parquet('data/processed/sales_main.parquet')
        df_data = pd.read_parquet('data/processed/sales_data.parquet')
        return df_main, df_data
    except Exception as e:
        logger.error(f"Error reading data: {str(e)}")
        return None, None


def create_customer_analysis(df_main, df_data):
    """Create customer_analysis dataset"""
    try:
        # Merge data
        df = df_data.merge(df_main, on='uniquenum_pri', how='left')
        
        # Calculate metrics for customer (using columns from sales_data)
        customer_metrics = df.groupby('party_code_x').agg({
            'amount_local_x': ['sum', 'mean', 'count'],
            'stkcode_code': 'nunique',
            'date_trans_x': ['min', 'max']
        }).reset_index()
        
        # Set column names
        customer_metrics.columns = [
            'customer_id', 'total_revenue', 'avg_order_value', 'total_purchases',
            'unique_products', 'first_purchase_date', 'last_purchase_date'
        ]
        
        # Add other information
        customer_metrics['customer_name'] = 'Customer_' + customer_metrics['customer_id'].astype(str)
        
        # Save dataset
        customer_metrics.to_parquet('data/processed/customer_analysis.parquet', index=False)
        logger.info(f"Created customer_analysis: {len(customer_metrics)} records")
        return customer_metrics
    except Exception as e:
        logger.error(f"Error creating customer_analysis: {str(e)}")
        return None


def create_product_analysis(df_main, df_data):
    """Create product_analysis dataset"""
    try:
        # Merge data
        df = df_data.merge(df_main, on='uniquenum_pri', how='left', suffixes=('_data', '_main'))
        
        # Find amount_local column after merge
        amount_col = None
        for col in df.columns:
            if 'amount_local' in col:
                amount_col = col
                break
        
        if amount_col is None:
            logger.error("Cannot find amount_local column after merge")
            return None
        
        # Find party_code column after merge
        party_col = None
        for col in df.columns:
            if 'party_code' in col:
                party_col = col
                break
        
        if party_col is None:
            logger.error("Cannot find party_code column after merge")
            return None
        
        # Calculate metrics for product
        product_metrics = df.groupby('stkcode_code').agg({
            amount_col: ['sum', 'mean', 'count'],
            'qnty_total': 'sum',
            party_col: 'nunique',
            'date_trans_data': ['min', 'max']
        }).reset_index()
        
        # Set column names
        product_metrics.columns = [
            'product_code', 'total_revenue', 'avg_price', 'total_transactions',
            'total_quantity', 'unique_customers', 'first_sale_date', 'last_sale_date'
        ]
        
        # Add other information
        product_metrics['product_name'] = 'Product_' + product_metrics['product_code'].astype(str)
        
        # Save dataset
        product_metrics.to_parquet('data/processed/product_analysis.parquet', index=False)
        logger.info(f"Created product_analysis: {len(product_metrics)} records")
        return product_metrics
    except Exception as e:
        logger.error(f"Error creating product_analysis: {str(e)}")
        return None


def create_sales_trend(df_main, df_data):
    """Create sales_trend dataset"""
    try:
        # Merge data
        df = df_data.merge(df_main, on='uniquenum_pri', how='left')
        
        # Convert date
        df['date_trans_x'] = pd.to_datetime(df['date_trans_x'])
        
        # Calculate trends
        trend_metrics = df.groupby(['month', 'quarter', 'day_of_week']).agg({
            'amount_local_x': ['sum', 'mean', 'count'],
            'qnty_total': 'sum'
        }).reset_index()
        
        # Set column names
        trend_metrics.columns = [
            'month', 'quarter', 'day_of_week', 'total_revenue', 'avg_transaction_value',
            'total_transactions', 'total_quantity'
        ]
        
        # Calculate growth
        trend_metrics['revenue_growth'] = trend_metrics['total_revenue'].pct_change()
        
        # Save dataset
        trend_metrics.to_parquet('data/processed/sales_trend.parquet', index=False)
        logger.info(f"Created sales_trend: {len(trend_metrics)} records")
        return trend_metrics
    except Exception as e:
        logger.error(f"Error creating sales_trend: {str(e)}")
        return None


def create_customer_retention(df_main, df_data):
    """Create customer_retention dataset"""
    try:
        # Merge data
        df = df_data.merge(df_main, on='uniquenum_pri', how='left')
        
        # Calculate retention
        customer_retention = df.groupby('party_code_x').agg({
            'date_trans_x': ['min', 'max', 'count'],
            'amount_local_x': 'sum'
        }).reset_index()
        
        # Set column names
        customer_retention.columns = [
            'customer_id', 'first_purchase_date', 'last_purchase_date',
            'total_purchases', 'total_spent'
        ]
        
        # Calculate churn (customers who haven't purchased in last 90 days)
        customer_retention['days_since_last_purchase'] = (
            pd.to_datetime('today') - pd.to_datetime(customer_retention['last_purchase_date'])
        ).dt.days
        
        customer_retention['is_churned'] = (customer_retention['days_since_last_purchase'] > 90).astype(int)
        
        # Save dataset
        customer_retention.to_parquet('data/processed/customer_retention.parquet', index=False)
        logger.info(f"Created customer_retention: {len(customer_retention)} records")
        return customer_retention
    except Exception as e:
        logger.error(f"Error creating customer_retention: {str(e)}")
        return None
    
def create_isolated_revenue_report(df_main, df_data):
    """
    NEW HANDLER: Create a specific dataset for date-based revenue analysis.
    This is isolated from general trends to provide clean, filtered data.
    """
    try:
        df = df_data.merge(df_main, on='uniquenum_pri', how='left')
        
        df['date_trans_x'] = pd.to_datetime(df['date_trans_x'])
        df['year'] = df['date_trans_x'].dt.year
        df['month'] = df['date_trans_x'].dt.month
        
        isolated_revenue = df[df['amount_local_x'] > 0].groupby(['year', 'month']).agg({
            'amount_local_x': 'sum',
            'uniquenum_pri': 'nunique', # Đếm số hóa đơn duy nhất
            'party_code_x': 'nunique'   # Đếm số khách hàng trong tháng
        }).reset_index()
        
        isolated_revenue.columns = [
            'report_year', 'report_month', 'total_revenue', 
            'transaction_count', 'active_customers'
        ]
        
        output_path = 'data/processed/revenue_report_by_date.parquet'
        isolated_revenue.to_parquet(output_path, index=False)
        
        logger.info(f"SUCCESS: Created isolated revenue report: {len(isolated_revenue)} records at {output_path}")
        return isolated_revenue
        
    except Exception as e:
        logger.error(f"Error creating isolated revenue report: {str(e)}")
        return None


def main():
    """Main function"""
    logger.info("Starting to create datasets...")
    
    # Read data
    df_main, df_data = load_sales_data()
    if df_main is None or df_data is None:
        logger.error("Cannot read data")
        return
    
    # Create datasets
    create_customer_analysis(df_main, df_data)
    create_product_analysis(df_main, df_data)
    create_sales_trend(df_main, df_data)
    create_customer_retention(df_main, df_data)
    create_isolated_revenue_report(df_main, df_data)
    
    logger.info("Completed creating datasets!")


if __name__ == "__main__":
    main()