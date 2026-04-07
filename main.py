"""
AI Training System - Main Entry Point
AI training system for sales data analysis
"""

import logging
import argparse
import sys
from datetime import datetime

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_extraction(args):
    """Run data extraction"""
    from src.extractors.sales_extractor import SalesExtractor
    from src.transformers.data_transformer import DataTransformer
    
    logger.info("Starting data extraction")
    
    extractor = SalesExtractor()
    transformer = DataTransformer()
    
    # Nếu không truyền date range thì lấy tất cả dữ liệu có sẵn
    if not args.date_from or not args.date_to:
        # Lấy khoảng thời gian thực tế từ database
        min_date, max_date = extractor.get_available_date_range()
        date_from = args.date_from or min_date
        date_to = args.date_to or max_date
        logger.info(f"Auto detected full date range: {date_from} to {max_date}")
    else:
        date_from = args.date_from
        date_to = args.date_to
        logger.info(f"Using custom date range: {date_from} to {date_to}")
    
    # Extract
    df_main = extractor.extract_sales_main(date_from=date_from, date_to=date_to)
    df_data = extractor.extract_sales_data(date_from=date_from, date_to=date_to)
    
    # Transform
    df_main_clean = transformer.clean_sales_main(df_main)
    df_data_clean = transformer.clean_sales_data(df_data)
    
    # Save
    transformer.save_transformed_data(df_main_clean, 'data/processed/sales_main.parquet')
    transformer.save_transformed_data(df_data_clean, 'data/processed/sales_data.parquet')
    
    extractor.close()
    
    logger.info(f"Extraction completed: {len(df_main)} main, {len(df_data)} detail records")


def run_training(args):
    """Run model training"""
    from src.extractors.sales_extractor import SalesExtractor
    from src.transformers.data_transformer import DataTransformer
    from src.trainers.churn_predictor import ChurnPredictor
    from src.trainers.sales_forecaster import SalesForecaster
    
    logger.info("Starting model training")
    
    extractor = SalesExtractor()
    transformer = DataTransformer()
    
    # Churn prediction
    if args.model in ['churn', 'all']:
        logger.info("Training churn prediction model")
        df_retention = extractor.extract_customer_retention_data()
        df_retention_clean = transformer.transform_customer_retention(df_retention)
        
        churn_predictor = ChurnPredictor()
        df_retention_prep = churn_predictor.prepare_churn_data(df_retention_clean)
        result = churn_predictor.train(df_retention_prep)
        logger.info(f"Churn model: F1={result['metrics']['f1_score']:.4f}")
    
    # Sales forecast
    if args.model in ['forecast', 'all']:
        logger.info("Training sales forecast model")
        df_trend = extractor.extract_sales_trend_data()
        df_trend_clean = transformer.transform_sales_trend(df_trend)
        
        forecaster = SalesForecaster()
        df_trend_prep = forecaster.prepare_forecast_data(df_trend_clean)
        result = forecaster.train(df_trend_prep)
        logger.info(f"Forecast model: R2={result['metrics']['r2']:.4f}")
    
    extractor.close()
    logger.info("Training completed")


def run_query(args):
    """Run AI query"""
    from src.query.ai_query_interface import AIQueryInterface
    
    logger.info("Starting AI query")
    
    interface = AIQueryInterface(companyfn=args.companyfn)
    
    if args.interactive:
        # Interactive mode
        print("\n=== AI Query Interface ===")
        print("Enter questions about sales data (type 'quit' to exit)")
        print("Examples: 'Top customers by purchases', 'Monthly sales trends'")
        print()
        
        while True:
            try:
                query = input("Question: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if not query:
                    continue
                
                result = interface.process_query(query)
                response = interface.format_response(result)
                print(f"\n{response}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {str(e)}")
    else:
        # Single query
        query = args.query or "Sales overview"
        result = interface.process_query(query)
        response = interface.format_response(result)
        print(response)


def run_scheduled(args):
    """Run scheduled training"""
    from scripts.scheduled_training import main as scheduled_main
    
    logger.info("Starting scheduled training service")
    scheduled_main()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='AI Training System - Sales Data Analysis System'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract and transform data')
    extract_parser.add_argument('--date-from', help='Start date (YYYY-MM-DD)')
    extract_parser.add_argument('--date-to', help='End date (YYYY-MM-DD)')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train ML models')
    train_parser.add_argument('--model', choices=['churn', 'forecast', 'all'], default='all',
                             help='Model to train')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query AI for insights')
    query_parser.add_argument('--query', '-q', help='Query string')
    query_parser.add_argument('--companyfn', '-c', help='Company code for data isolation')
    query_parser.add_argument('--interactive', '-i', action='store_true',
                             help='Interactive mode')
    
    # Scheduled command
    scheduled_parser = subparsers.add_parser('scheduled', help='Run scheduled training')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'extract':
            run_extraction(args)
        elif args.command == 'train':
            run_training(args)
        elif args.command == 'query':
            run_query(args)
        elif args.command == 'scheduled':
            run_scheduled(args)
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()