"""
Scheduled Training Script
Script chạy training tự động theo lịch
"""

import logging
import schedule
import time
from datetime import datetime
import sys
import os

# Thêm parent directory vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extractors.sales_extractor import SalesExtractor
from src.transformers.data_transformer import DataTransformer
from src.transformers.feature_engineer import FeatureEngineer
from src.trainers.churn_predictor import ChurnPredictor
from src.trainers.sales_forecaster import SalesForecaster

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scheduled_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_daily_training():
    """Chạy training hàng ngày"""
    try:
        logger.info("=== Bắt đầu daily training ===")
        
        # 1. Extract data
        logger.info("Bước 1: Extract data")
        extractor = SalesExtractor()
        
        # Lấy data 30 ngày gần nhất
        from datetime import timedelta
        date_from = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        date_to = datetime.now().strftime('%Y-%m-%d')
        
        df_main = extractor.extract_sales_main(date_from=date_from, date_to=date_to)
        df_data = extractor.extract_sales_data(date_from=date_from, date_to=date_to)
        
        logger.info(f"Extracted {len(df_main)} main records, {len(df_data)} detail records")
        
        # 2. Transform data
        logger.info("Bước 2: Transform data")
        transformer = DataTransformer()
        
        df_main_clean = transformer.clean_sales_main(df_main)
        df_data_clean = transformer.clean_sales_data(df_data)
        
        # Lưu processed data
        transformer.save_transformed_data(df_main_clean, 'data/processed/sales_main_latest.parquet')
        transformer.save_transformed_data(df_data_clean, 'data/processed/sales_data_latest.parquet')
        
        # 3. Retrain models (nếu có đủ data)
        if len(df_main_clean) > 100:
            logger.info("Bước 3: Retrain models")
            
            # Churn prediction
            churn_predictor = ChurnPredictor()
            df_retention = extractor.extract_customer_retention_data()
            df_retention_prep = churn_predictor.prepare_churn_data(df_retention)
            
            if len(df_retention_prep) > 50:
                churn_result = churn_predictor.train(df_retention_prep)
                logger.info(f"Churn model retrained: F1={churn_result['metrics']['f1_score']:.4f}")
            
            # Sales forecast
            forecaster = SalesForecaster()
            df_trend = extractor.extract_sales_trend_data(date_from=date_from, date_to=date_to)
            df_trend_prep = forecaster.prepare_forecast_data(df_trend)
            
            if len(df_trend_prep) > 30:
                forecast_result = forecaster.train(df_trend_prep)
                logger.info(f"Forecast model retrained: R2={forecast_result['metrics']['r2']:.4f}")
        
        extractor.close()
        logger.info("=== Hoàn thành daily training ===")
        
    except Exception as e:
        logger.error(f"Lỗi daily training: {str(e)}")


def run_weekly_training():
    """Chạy training hàng tuần (full retrain)"""
    try:
        logger.info("=== Bắt đầu weekly training ===")
        
        # 1. Extract full data
        logger.info("Bước 1: Extract full data")
        extractor = SalesExtractor()
        
        # Lấy data 1 năm gần nhất
        from datetime import timedelta
        date_from = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        date_to = datetime.now().strftime('%Y-%m-%d')
        
        # Customer analysis
        df_customer = extractor.extract_customer_analysis_data(date_from=date_from, date_to=date_to)
        
        # Product analysis
        df_product = extractor.extract_product_analysis_data(date_from=date_from, date_to=date_to)
        
        # Sales trend
        df_trend = extractor.extract_sales_trend_data(date_from=date_from, date_to=date_to)
        
        # Customer retention
        df_retention = extractor.extract_customer_retention_data(lookback_days=365)
        
        logger.info(f"Extracted: {len(df_customer)} customer, {len(df_product)} product, {len(df_trend)} trend records")
        
        # 2. Transform and save
        logger.info("Bước 2: Transform and save")
        transformer = DataTransformer()
        
        df_customer_clean = transformer.transform_customer_analysis(df_customer)
        df_product_clean = transformer.transform_product_analysis(df_product)
        df_trend_clean = transformer.transform_sales_trend(df_trend)
        df_retention_clean = transformer.transform_customer_retention(df_retention)
        
        transformer.save_transformed_data(df_customer_clean, 'data/processed/customer_analysis.parquet')
        transformer.save_transformed_data(df_product_clean, 'data/processed/product_analysis.parquet')
        transformer.save_transformed_data(df_trend_clean, 'data/processed/sales_trend.parquet')
        transformer.save_transformed_data(df_retention_clean, 'data/processed/customer_retention.parquet')
        
        # 3. Full model retrain
        logger.info("Bước 3: Full model retrain")
        
        # Churn prediction
        churn_predictor = ChurnPredictor()
        df_retention_prep = churn_predictor.prepare_churn_data(df_retention_clean)
        churn_result = churn_predictor.train(df_retention_prep, hyperparameter_tuning=True)
        logger.info(f"Churn model: F1={churn_result['metrics']['f1_score']:.4f}")
        
        # Sales forecast
        forecaster = SalesForecaster()
        df_trend_prep = forecaster.prepare_forecast_data(df_trend_clean)
        forecast_result = forecaster.train(df_trend_prep)
        logger.info(f"Forecast model: R2={forecast_result['metrics']['r2']:.4f}")
        
        extractor.close()
        logger.info("=== Hoàn thành weekly training ===")
        
    except Exception as e:
        logger.error(f"Lỗi weekly training: {str(e)}")


def main():
    """Main function"""
    logger.info("Khởi động Scheduled Training Service")
    
    # Tạo thư mục logs nếu chưa có
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/models', exist_ok=True)
    
    # Lên lịch training
    schedule.every().day.at("02:00").do(run_daily_training)  # 2h sáng hàng ngày
    schedule.every().sunday.at("03:00").do(run_weekly_training)  # 3h sáng Chủ nhật
    
    logger.info("Đã lên lịch:")
    logger.info("  - Daily training: 02:00 hàng ngày")
    logger.info("  - Weekly training: 03:00 Chủ nhật")
    
    # Chạy ngay lần đầu
    logger.info("Chạy initial training...")
    run_daily_training()
    
    # Loop
    logger.info("Bắt đầu monitoring loop...")
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    main()