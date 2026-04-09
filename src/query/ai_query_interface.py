"""
AI Query Interface Module
Prompt-based interface for querying AI about data
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import json
import re

logger = logging.getLogger(__name__)


class AIQueryInterface:
    """Class for AI query interface"""
    
    def __init__(self, data_path: str = "data/processed", companyfn: Optional[str] = None):
        """
        Initialize AIQueryInterface
        
        Args:
            data_path: Path to processed data
            companyfn: Company code for data isolation (unique per company)
        """
        self.data_path = data_path
        # ✅ DEFAULT COMPANY FN - DÙNG NẾU KHÔNG TRUYỀN
        self.companyfn = companyfn or "p11011004464072155"
        self.data_cache = {}
        self.query_history = []
    
    def load_data(self, dataset_name: str) -> pd.DataFrame:
        """
        Read data from cache or file, filtered by companyfn for data isolation
        
        Args:
            dataset_name: Dataset name
            
        Returns:
            DataFrame filtered by companyfn
        """
        try:
            # Create cache key with companyfn for data isolation
            cache_key = f"{dataset_name}_{self.companyfn}" if self.companyfn else dataset_name
            if cache_key in self.data_cache:
                return self.data_cache[cache_key]
            
            import os
            from pathlib import Path
            
            # Find file
            data_dir = Path(self.data_path)
            for ext in ['parquet', 'csv', 'json']:
                file_path = data_dir / f"{dataset_name}.{ext}"
                if file_path.exists():
                    if ext == 'parquet':
                        df = pd.read_parquet(file_path)
                    elif ext == 'csv':
                        df = pd.read_csv(file_path)
                    else:
                        df = pd.read_json(file_path)
                    
                    # Filter by companyfn for data isolation
                    if self.companyfn and 'companyfn' in df.columns:
                        df = df[df['companyfn'] == self.companyfn]
                        logger.info(f"Filtered {dataset_name} by companyfn={self.companyfn}: {len(df)} records")
                    
                    self.data_cache[cache_key] = df
                    return df
            
            raise FileNotFoundError(f"Dataset not found: {dataset_name}")
            
        except Exception as e:
            logger.error(f"Error reading data: {str(e)}")
            raise
    
    def process_query(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Process query from user
        
        Args:
            query: User question
            context: Additional context
            
        Returns:
            Dict with results
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Analyze query
            query_lower = query.lower()
            
            # ✅ KIỂM TRA TRUY VẤN TREND SẢN PHẨM
            trend_keywords = ['triển vọng', 'xu hướng', 'tương lai', 'sắp tới', 'tiềm năng', 'top 10', 'trend', 'potential', 'forecast', 'potential products', 'top products', 'top 10 products']
            is_trend_query = any(keyword in query_lower for keyword in trend_keywords)
            
            if is_trend_query:
                logger.info("✅ Phát hiện truy vấn phân tích xu hướng sản phẩm")
                from src.extractors.sales_extractor import SalesExtractor
                from src.analysis.product_trend_analyzer import ProductTrendAnalyzer
                
                with SalesExtractor(companyfn=self.companyfn) as extractor:
                    analyzer = ProductTrendAnalyzer(extractor)
                    trend_result = analyzer.analyze_product_trends(
                        companyfn=self.companyfn,
                        days_history=90,
                        top_n=10
                    )
                    
                    return {
                        'query_type': 'product_trend_analysis',
                        'status': 'success',
                        'result': trend_result,
                        'query': query
                    }
            
            # Determine query type
            query_types = self._classify_query(query_lower)
            combined_results = {
                'summary': "",
                'data': {},
                'insights': []
            }
            for q_type in query_types:
                result = None
                if q_type == 'revenue_report_by_date':
                    result = self._handle_date_revenue_query(query_lower, None)
                elif q_type == 'customer_analysis':
                    result = self._handle_customer_query(query_lower, context)
                elif q_type == 'product_analysis':
                    result = self._handle_product_query(query_lower, context)
                elif q_type == 'sales_trend':
                    result = self._handle_sales_trend_query(query_lower, context)
                elif q_type == 'churn_prediction':
                    result = self._handle_churn_query(query_lower, context)
                elif q_type == 'sales_forecast':
                    result = self._handle_forecast_query(query_lower, context)

                # --- Xử lý đổ result vào result combine ---
                if result:
                    if result.get('summary'):
                        combined_results['summary'] += f"\n{result['summary']}"
                    
                    combined_results['data'][q_type] = result.get('data', {})
                    
                    if result.get('insights'):
                        combined_results['insights'].extend(result['insights'])
                # ------------------------------

            if not combined_results['data']:
                combined_results = self._handle_general_query(query_lower, context)
            
            # Save history
            self.query_history.append({
                'query': query,
                'query_type': query_types,
                'timestamp': datetime.now().isoformat()
            })
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'error': str(e),
                'query': query
            }
    
    def _classify_query(self, query: str) -> str:
        """Classify query type"""
        query = query.lower()
        customer_keywords = ['customer', 'purchase', 'repurchase', 'churn', 'retention', 'active customers', 'average order', 'repeat customer', 'order value']
        product_keywords = ['product', 'item', 'goods', 'bestseller', 'potential products', 'top products', 'best selling']
        trend_keywords = ['trend', 'over time', 'monthly', 'yearly', 'growth', 'sales trends', 'monthly sales']
        forecast_keywords = ['forecast', 'predict', 'future', 'projection', 'plan', 'next 30 days', 'sales forecast']
        revenue_report_keywords = ['revenue report', 'revenue by date', 'revenue date', 'revenue by month', 'show me revenue', 'revenue for']
        
        targets = []

        # Match MM/YYYY, MM.YYYY, MM-YYYY, text month, year only
        date_pattern = r'(\d{1,2}/\d{4})|(\d{1,2}\.\d{4})|(\d{1,2}-\d{4})|(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*\d{4}|year\s*\d{4}|\d{4}\s+year|revenue\s+\d{4}'
        if re.search(date_pattern, query, re.IGNORECASE) or any(kw in query for kw in revenue_report_keywords) or any(word.isdigit() and len(word)==4 for word in query.split()):
            targets.append('revenue_report_by_date')    

        if any(kw in query for kw in customer_keywords):
            targets.append('customer_analysis')
        if any(kw in query for kw in product_keywords):
            targets.append('product_analysis')
        if any(kw in query for kw in trend_keywords):
            targets.append('sales_trend')
        if any(kw in query for kw in forecast_keywords):
            targets.append('sales_forecast')

        if not targets:
            return ['general']

        return targets
    
    def _handle_customer_query(self, query: str, context: Optional[Dict]) -> Dict:
        """Handle customer-related query"""
        try:
            # Read data
            df = self.load_data('customer_analysis')
            
            result = {
                'query_type': 'customer_analysis',
                'summary': '',
                'data': {},
                'insights': []
            }

            date_match = re.search(r'(\d{1,2})/(\d{4})', query)
            period_text = "all-time"
            
            if date_match:
                month = int(date_match.group(1))
                year = int(date_match.group(2))
                
                # Check if date columns exist before filtering
                if 'month' in df.columns and 'year' in df.columns:
                    df = df[(df['month'] == month) & (df['year'] == year)]
                    period_text = f"for {month}/{year}"
                else:
                    result['insights'].append("Warning: Monthly filtering requested but 'month/year' columns missing in data.")

            revenue_metrics = ['line_amount', 'total_revenue', 'amt_local', 'amount', 'revenue']
            revenue_col = next((col for col in revenue_metrics if col in df.columns), None)
            
            # Top customers
            if 'top' in query or 'most' in query or 'best' in query:                
                if revenue_col:
                    # Group by theo tên khách hàng và tính tổng doanh thu
                    top_customers = df.groupby('customer_name')[revenue_col].sum().sort_values(ascending=False).head(10)
                    
                    if not top_customers.empty:
                        result['data']['top_customers'] = top_customers.to_dict()
                        result['summary'] = f"Top 10 customers by revenue ({period_text})"
                        result['insights'].append(f"Number 1 customer {period_text}: {top_customers.index[0]} (${top_customers.iloc[0]:,.2f})")
                    else:
                        result['summary'] = f"No customer data found {period_text}"
                else:
                    result['summary'] = "No valid revenue column found for ranking"            
            # Repeat customers
            elif 'repeat' in query or 'return' in query:
                if 'customer_id' in df.columns and 'transaction_date' in df.columns:
                    # Đếm số ngày mua hàng duy nhất của mỗi khách
                    repeat_counts = df.groupby('customer_id')['transaction_date'].nunique()
                    repeat_rate = (repeat_counts > 1).mean() * 100
                    result['data']['repeat_rate'] = repeat_rate
                    result['summary'] = f"Customer repeat rate ({period_text}): {repeat_rate:.1f}%"
                    result['insights'].append(f"{(repeat_counts > 1).sum()} customers made repeat purchases")
                else:
                    result['summary'] = "Missing ID or Date columns to calculate repeat rate"
            
            # Customer segments
            elif 'segment' in query:
                if 'customer_segment' in df.columns:
                    # Đếm số lượng khách hàng trong từng phân khúc
                    segments = df['customer_segment'].value_counts()
                    result['data']['customer_segments'] = segments.to_dict()
                    result['summary'] = f"Customer segments distribution ({period_text})"
                    
                    # Thêm insight về phân khúc lớn nhất
                    top_seg = segments.index[0]
                    result['insights'].append(f"Largest segment {period_text}: {top_seg} with {segments.iloc[0]} customers")
                else:
                    result['summary'] = f"Segmentation column not found in data"
            
            # Default: Overview
            else:
                # Đếm số lượng khách hàng duy nhất
                total_customers = df['customer_id'].nunique() if 'customer_id' in df.columns else len(df)
                
                # Tự động tìm cột doanh thu phù hợp
                revenue_metrics = ['line_amount', 'total_revenue', 'amount', 'revenue', 'total']
                revenue_col = next((col for col in revenue_metrics if col in df.columns), None)
                
                if revenue_col:
                    total_revenue = df[revenue_col].sum()
                    # Tính giá trị trung bình trên mỗi khách hàng
                    avg_order_value = df.groupby('customer_id')[revenue_col].sum().mean() if 'customer_id' in df.columns else 0
                    
                    result['data'] = {
                        'total_customers': total_customers,
                        'total_revenue': total_revenue,
                        'avg_customer_value': avg_order_value
                    }
                    result['summary'] = f"Customer overview ({period_text}): {total_customers} customers, Total Revenue: ${total_revenue:,.2f}"
                    result['insights'].append(f"Average value per customer {period_text}: ${avg_order_value:,.2f}")
                else:
                    result['data'] = {'total_customers': total_customers}
                    result['summary'] = f"Customer overview ({period_text}): {total_customers} customers"
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling customer query: {str(e)}")
            return {'error': str(e)}
    
    def _handle_product_query(self, query: str, context: Optional[Dict]) -> Dict:
        """Handle product-related query"""
        try:
            df = self.load_data('product_analysis')
            
            result = {
                'query_type': 'product_analysis',
                'summary': '',
                'data': {},
                'insights': []
            }
            
            # Bestselling products
            if 'bestseller' in query or 'top' in query:
                # ✅ AUTO DETECT REVENUE COLUMN
                revenue_metrics = ['line_amount', 'total_revenue', 'amt_local', 'amount', 'revenue']
                revenue_col = next((col for col in revenue_metrics if col in df.columns), None)
                if revenue_col:
                    top_products = df.groupby('product_name')[revenue_col].sum().sort_values(ascending=False).head(10)
                    result['data']['top_products'] = top_products.to_dict()
                    result['summary'] = f"Top 10 bestselling products"
                    if not top_products.empty:
                        result['insights'].append(f"Bestselling product: {top_products.index[0]}")
                else:
                    result['summary'] = "No revenue column found in product dataset"
            
            # By category
            elif 'category' in query:
                category_sales = df.groupby('category_name')['revenue'].sum().sort_values(ascending=False)
                result['data']['category_sales'] = category_sales.to_dict()
                result['summary'] = f"Sales by category"
            
            # By brand
            elif 'brand' in query:
                brand_sales = df.groupby('brand_desc')['revenue'].sum().sort_values(ascending=False)
                result['data']['brand_sales'] = brand_sales.to_dict()
                result['summary'] = f"Sales by brand"
            
            # Default
            else:
                total_products = df['product_code'].nunique()
                total_revenue = df['revenue'].sum()
                total_quantity = df['quantity_sold'].sum()
                
                result['data'] = {
                    'total_products': total_products,
                    'total_revenue': total_revenue,
                    'total_quantity': total_quantity
                }
                result['summary'] = f"Product overview: {total_products} products, revenue {total_revenue:,.0f}"
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling product query: {str(e)}")
            return {'error': str(e)}
    
    def _handle_sales_trend_query(self, query: str, context: Optional[Dict]) -> Dict:
        """Handle sales trend query"""
        try:
            df = self.load_data('sales_trend')
            
            result = {
                'query_type': 'sales_trend',
                'summary': '',
                'data': {},
                'insights': []
            }
            
            # Monthly
            if 'monthly' in query:
                monthly_sales = df.groupby('month')['total_revenue'].sum()
                result['data']['monthly_sales'] = monthly_sales.to_dict()
                result['summary'] = f"Monthly sales"
                
                # Find best month
                best_month = monthly_sales.idxmax()
                result['insights'].append(f"Best month: Month {best_month}")
            
            # Quarterly
            elif 'quarterly' in query or 'quarter' in query:
                quarterly_sales = df.groupby('quarter')['total_revenue'].sum()
                result['data']['quarterly_sales'] = quarterly_sales.to_dict()
                result['summary'] = f"Quarterly sales"
            
            # Day of week
            elif 'day' in query or 'weekday' in query:
                daily_sales = df.groupby('day_of_week')['total_revenue'].sum()
                result['data']['daily_sales'] = daily_sales.to_dict()
                result['summary'] = f"Sales by day of week"
            
            # Default
            else:
                total_revenue = df['total_revenue'].sum()
                avg_daily = df['total_revenue'].mean()
                growth = df['revenue_growth'].mean() * 100 if 'revenue_growth' in df.columns else 0
                
                result['data'] = {
                    'total_revenue': total_revenue,
                    'avg_daily_revenue': avg_daily,
                    'avg_growth_rate': growth
                }
                result['summary'] = f"Total revenue: {total_revenue:,.0f}, average growth: {growth:.1f}%"
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling sales trend query: {str(e)}")
            return {'error': str(e)}
    
    def _handle_churn_query(self, query: str, context: Optional[Dict]) -> Dict:
        """Handle churn-related query"""
        try:
            df = self.load_data('customer_retention')
            
            result = {
                'query_type': 'churn_prediction',
                'summary': '',
                'data': {},
                'insights': []
            }
            
            # Churn rate
            if 'churn' in df.columns:
                churn_rate = df['churn'].mean() * 100
                result['data']['churn_rate'] = churn_rate
                result['summary'] = f"Customer churn rate: {churn_rate:.1f}%"
                result['insights'].append(f"{df['churn'].sum()} customers at risk of churning")
            
            # High risk customers
            if 'at risk' in query or 'high risk' in query:
                at_risk = df[df['churn_risk'] == 'High'] if 'churn_risk' in df.columns else pd.DataFrame()
                result['data']['at_risk_customers'] = len(at_risk)
                result['insights'].append(f"{len(at_risk)} customers at high risk of churning")
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling churn query: {str(e)}")
            return {'error': str(e)}
    
    def _handle_forecast_query(self, query: str, context: Optional[Dict]) -> Dict:
        """Handle forecast-related query"""
        try:
            result = {
                'query_type': 'sales_forecast',
                'summary': 'Sales forecast',
                'data': {},
                'insights': []
            }
            
            # Read historical data
            df = self.load_data('sales_trend')
            
            # Simple forecast (average)
            avg_daily = df['total_revenue'].mean()
            
            # Forecast next 30 days
            forecast_30d = avg_daily * 30
            
            result['data'] = {
                'avg_daily_sales': avg_daily,
                'forecast_30_days': forecast_30d,
                'forecast_90_days': avg_daily * 90
            }
            result['summary'] = f"30-day sales forecast: {forecast_30d:,.0f}"
            result['insights'].append(f"Average daily sales: {avg_daily:,.0f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling forecast query: {str(e)}")
            return {'error': str(e)}
        
    def _handle_date_revenue_query(self, query: str, context: Optional[Dict]) -> Dict:
        """
        Specialized handler for time-specific revenue queries (MM/YYYY).
        """
        try:            
            # 1. Trích xuất Tháng và Năm từ query
            # Match cả format số và text
            month_map = {
                'jan':1, 'january':1,
                'feb':2, 'february':2,
                'mar':3, 'march':3,
                'apr':4, 'april':4,
                'may':5,
                'jun':6, 'june':6,
                'jul':7, 'july':7,
                'aug':8, 'august':8,
                'sep':9, 'september':9,
                'oct':10, 'october':10,
                'nov':11, 'november':11,
                'dec':12, 'december':12
            }
            
            result = {
                'query_type': 'revenue_report_by_date',
                'summary': '',
                'data': {},
                'insights': []
            }

            # 2. Load dữ liệu TRƯỚC KHI SỬ DỤNG DF
            df = self.load_data('revenue_report_by_date')

            # THỬ NẾU CHỈ CÓ NĂM - KHÔNG CÓ THÁNG TRONG QUERY
            year_only_match = re.search(r'^(?!.*\d{1,2}[./-]\d{4})(?!.*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec).*\d{4}).*\b(20\d{2})\b', query, re.IGNORECASE)
            if year_only_match:
                year_target = int(year_only_match.group(2))
                
                # Tính tổng cả năm
                year_data = df[df['report_year'].astype(float).astype(int) == year_target]
                
                if not year_data.empty:
                    total_year_revenue = year_data['total_revenue'].sum()
                    total_transactions = year_data['transaction_count'].sum()
                    
                    result['data'] = {
                        'period': f"Year {year_target}",
                        'total_revenue': total_year_revenue,
                        'transaction_count': total_transactions,
                        'month_count': len(year_data)
                    }
                    
                    result['summary'] = (
                        f"Total Revenue for year {year_target}: ${total_year_revenue:,.2f}, "
                        f"Total Transactions: {total_transactions:,}, "
                        f"Months with data: {len(year_data)}."
                    )
                    
                    return result

            # Thử match format số trước - HỖ TRỢ TẤT CẢ DẤU NGĂN CÁCH
            match = re.search(r'(\d{1,2})[./-](\d{4})', query)
            
            if match:
                month_target = int(match.group(1))
                year_target = int(match.group(2))
            else:
                # Thử match text month
                month_match = re.search(r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*(\d{4})', query, re.IGNORECASE)
                if month_match:
                    month_text = month_match.group(1).lower()
                    month_target = month_map[month_text]
                    year_target = int(month_match.group(2))
                else:
                    return {'error': 'No valid date format (MM/YYYY or Month YYYY) found in query.'}
            
            # 2. Load dữ liệu
            df = self.load_data('revenue_report_by_date')

            target_company = context.get('companyfn') if context else None

            if target_company and 'companyfn' in df.columns:
                df = df[df['companyfn'] == target_company]
            
            result = {
                'query_type': 'revenue_report_by_date',
                'summary': '',
                'data': {},
                'insights': []
            }

            # 3. Lọc dữ liệu theo tháng và năm
            if 'report_month' in df.columns and 'report_year' in df.columns:
                mask = (df['report_month'].astype(float).astype(int) == month_target) & \
                   (df['report_year'].astype(float).astype(int) == year_target)
                
                target_data = df[mask]
                
                if not target_data.empty:
                    row = target_data.iloc[0]                    
                    rev = float(row.get('total_revenue', 0))
                    trans = int(row.get('transaction_count', 0))
                    growth = 0.0
                    prev_month = 12 if month_target == 1 else month_target - 1
                    prev_year = year_target - 1 if month_target == 1 else year_target
                    prev_mask = (df['report_month'].astype(float).astype(int) == prev_month) & \
                            (df['report_year'].astype(float).astype(int) == prev_year)
                    prev_data = df[prev_mask]
                    if not prev_data.empty:
                        prev_rev = float(prev_data.iloc[0].get('total_revenue', 0))
                        if prev_rev > 0:
                            growth = ((rev - prev_rev) / prev_rev) * 100
                            
                    # Đổ dữ liệu vào kết quả
                    result['data'] = {
                        'period': f"{month_target}/{year_target}",
                        'total_revenue': rev,
                        'transaction_count': trans,
                        'growth_rate_percentage': f"{growth:.2f}%"
                    }
                    
                    result['summary'] = (
						f"Report for {month_target}/{year_target}: "
						f"Total Revenue: ${rev:,.2f}, "
						f"Transactions: {trans:,}, "
						f"Growth: {growth:.2f}%."
					)
                    
                    result['insights'] = [
						f"Successfully retrieved {trans} transactions from the database.",
						f"The average revenue per transaction is ${rev/trans:,.2f}." if trans > 0 else "No transaction volume to calculate average."
					]
                else:
                    result['summary'] = f"No specific records found for {month_target}/{year_target} in the database."
            else:
                result['summary'] = f"The dataset columns are: {list(df.columns)}. Expected 'report_month' and 'report_year' columns."

            return result

        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Error in DateRevenueAnalyst: {str(e)}")
            return {'error': f"Processing failed: {str(e)}"}
        
    def _handle_general_query(self, query: str, context: Optional[Dict]) -> Dict:
        """Handle general query"""
        return {
            'query_type': 'general',
            'summary': 'General data query',
            'data': {},
            'insights': ['Please ask specifically about customers, products, trends, or forecasts']
        }
    
    def get_query_history(self) -> List[Dict]:
        """Get query history"""
        return self.query_history
    
    def format_response(self, result: Dict) -> str:
        """
        Format result into readable text
        
        Args:
            result: Result Dict
            
        Returns:
            String response
        """
        try:
            response = []
            
            if 'error' in result:
                return f"Error: {result['error']}"
            
            response.append(f"[Report] {result.get('summary', 'Query result')}")
            response.append("")
            
            # Insights
            if result.get('insights'):
                response.append("* Insights:")
                for insight in result['insights']:
                    response.append(f"  • {insight}")
                response.append("")
            
            # Data summary
            if result.get('data'):
                response.append("* Data:")
                for key, value in result['data'].items():
                    if isinstance(value, (int, float)):
                        if value > 1000000:
                            response.append(f"  • {key}: {value:,.0f}")
                        elif value > 100:
                            response.append(f"  • {key}: {value:,.2f}")
                        else:
                            response.append(f"  • {key}: {value}")
                    else:
                        response.append(f"  • {key}: {value}")
            
            # ✅ AUTO GỢI Ý CÂU HỎI TIẾP THEO
            response.append("")
            response.append("---")
            response.append("[Suggested] **Câu hỏi bạn có thể hỏi tiếp:**")
            
            followup_questions = []
            
            # Tự động tạo câu hỏi followup dựa trên loại query
            if result.get('data') and 'period' in result['data']:
                period = result['data']['period']
                
                # Nếu là tháng thì gợi ý câu hỏi liên quan tháng
                if '/' in period:
                    m,y = period.split('/')
                    next_m = int(m)+1 if int(m) < 12 else 1
                    next_y = int(y) if int(m) < 12 else int(y)+1
                    prev_m = int(m)-1 if int(m) > 1 else 12
                    prev_y = int(y) if int(m) > 1 else int(y)-1
                    
                    followup_questions.append(f"Doanh thu tháng {next_m}/{next_y} là bao nhiêu?")
                    followup_questions.append(f"So sánh với tháng {prev_m}/{prev_y}")
                    followup_questions.append(f"Top khách hàng tháng {m}/{y}")
                    followup_questions.append(f"Xu hướng doanh thu 3 tháng gần nhất")
                
                # Nếu là cả năm
                elif 'Year' in period:
                    year = period.replace('Year ','')
                    followup_questions.append(f"Doanh thu từng tháng năm {year}")
                    followup_questions.append(f"So sánh với năm {int(year)-1}")
                    followup_questions.append(f"Top sản phẩm bán chạy năm {year}")
                    
            # ✅ Thêm vào JSON result cho frontend render button
            if followup_questions:
                result['suggested_queries'] = followup_questions[:4]
            
            for q in followup_questions[:4]:
                # Format như button clickable cho giao diện chat
                response.append(f"  > **[{q}](query:{q})**")

            return "\n".join(response)
            
        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            return f"Format error: {str(e)}"
