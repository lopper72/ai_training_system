"""
AI Query Interface Module
Giao diện prompt-based để hỏi AI về dữ liệu
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import json

logger = logging.getLogger(__name__)


class AIQueryInterface:
    """Class giao diện query AI"""
    
    def __init__(self, data_path: str = "data/processed"):
        """
        Khởi tạo AIQueryInterface
        
        Args:
            data_path: Đường dẫn dữ liệu đã xử lý
        """
        self.data_path = data_path
        self.data_cache = {}
        self.query_history = []
    
    def load_data(self, dataset_name: str) -> pd.DataFrame:
        """
        Đọc dữ liệu từ cache hoặc file
        
        Args:
            dataset_name: Tên dataset
            
        Returns:
            DataFrame
        """
        try:
            if dataset_name in self.data_cache:
                return self.data_cache[dataset_name]
            
            import os
            from pathlib import Path
            
            # Tìm file
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
                    
                    self.data_cache[dataset_name] = df
                    return df
            
            raise FileNotFoundError(f"Không tìm thấy dataset: {dataset_name}")
            
        except Exception as e:
            logger.error(f"Lỗi đọc dữ liệu: {str(e)}")
            raise
    
    def process_query(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Xử lý query từ người dùng
        
        Args:
            query: Câu hỏi từ người dùng
            context: Context bổ sung
            
        Returns:
            Dict kết quả
        """
        try:
            logger.info(f"Xử lý query: {query}")
            
            # Phân tích query
            query_lower = query.lower()
            
            # Xác định loại query
            query_type = self._classify_query(query_lower)
            
            # Thực hiện query
            if query_type == 'customer_analysis':
                result = self._handle_customer_query(query_lower, context)
            elif query_type == 'product_analysis':
                result = self._handle_product_query(query_lower, context)
            elif query_type == 'sales_trend':
                result = self._handle_sales_trend_query(query_lower, context)
            elif query_type == 'churn_prediction':
                result = self._handle_churn_query(query_lower, context)
            elif query_type == 'sales_forecast':
                result = self._handle_forecast_query(query_lower, context)
            else:
                result = self._handle_general_query(query_lower, context)
            
            # Lưu lịch sử
            self.query_history.append({
                'query': query,
                'query_type': query_type,
                'timestamp': datetime.now().isoformat(),
                'result_summary': str(result.get('summary', ''))[:100]
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Lỗi xử lý query: {str(e)}")
            return {
                'error': str(e),
                'query': query
            }
    
    def _classify_query(self, query: str) -> str:
        """Phân loại loại query"""
        customer_keywords = ['khách hàng', 'customer', 'mua hàng', 'mua lại', 'rời bỏ', 'churn']
        product_keywords = ['sản phẩm', 'product', 'mặt hàng', 'hàng hóa', 'bán chạy', 'doanh thu']
        trend_keywords = ['xu hướng', 'trend', 'theo thời gian', 'theo tháng', 'theo năm', 'tăng trưởng']
        forecast_keywords = ['dự đoán', 'forecast', 'tương lai', 'dự báo', 'kế hoạch']
        
        if any(kw in query for kw in customer_keywords):
            return 'customer_analysis'
        elif any(kw in query for kw in product_keywords):
            return 'product_analysis'
        elif any(kw in query for kw in trend_keywords):
            return 'sales_trend'
        elif any(kw in query for kw in forecast_keywords):
            return 'sales_forecast'
        else:
            return 'general'
    
    def _handle_customer_query(self, query: str, context: Optional[Dict]) -> Dict:
        """Xử lý query về khách hàng"""
        try:
            # Đọc dữ liệu
            df = self.load_data('customer_analysis')
            
            result = {
                'query_type': 'customer_analysis',
                'summary': '',
                'data': {},
                'insights': []
            }
            
            # Top khách hàng
            if 'top' in query or 'nhiều nhất' in query:
                # Tìm cột doanh số phù hợp
                revenue_col = None
                for col in ['total_revenue', 'line_amount', 'amount', 'revenue', 'total']:
                    if col in df.columns:
                        revenue_col = col
                        break
                
                if revenue_col:
                    top_customers = df.groupby('customer_name')[revenue_col].sum().sort_values(ascending=False).head(10)
                    result['data']['top_customers'] = top_customers.to_dict()
                    result['summary'] = f"Top 10 khách hàng có doanh số cao nhất"
                    result['insights'].append(f"Khách hàng hàng đầu: {top_customers.index[0]} với doanh số {top_customers.iloc[0]:,.0f}")
                else:
                    result['summary'] = f"Không tìm thấy cột doanh số trong dữ liệu"
            
            # Khách hàng mua lại
            elif 'mua lại' in query or 'quay lại' in query:
                repeat_customers = df.groupby('customer_id')['transaction_date'].nunique()
                repeat_rate = (repeat_customers > 1).mean() * 100
                result['data']['repeat_rate'] = repeat_rate
                result['summary'] = f"Tỷ lệ khách hàng mua lại: {repeat_rate:.1f}%"
                result['insights'].append(f"Có {(repeat_customers > 1).sum()} khách hàng mua lại trong kỳ")
            
            # Phân khúc khách hàng
            elif 'phân khúc' in query or 'segment' in query:
                if 'customer_segment' in df.columns:
                    segments = df['customer_segment'].value_counts()
                    result['data']['customer_segments'] = segments.to_dict()
                    result['summary'] = f"Phân khúc khách hàng"
            
            # Mặc định: Tổng quan
            else:
                total_customers = df['customer_id'].nunique()
                
                # Tìm cột doanh số phù hợp
                revenue_col = None
                for col in ['total_revenue', 'line_amount', 'amount', 'revenue', 'total']:
                    if col in df.columns:
                        revenue_col = col
                        break
                
                if revenue_col:
                    total_revenue = df[revenue_col].sum()
                    avg_order_value = df.groupby('customer_id')[revenue_col].mean().mean()
                    
                    result['data'] = {
                        'total_customers': total_customers,
                        'total_revenue': total_revenue,
                        'avg_order_value': avg_order_value
                    }
                    result['summary'] = f"Tổng quan khách hàng: {total_customers} khách hàng, doanh số {total_revenue:,.0f}"
                else:
                    result['data'] = {
                        'total_customers': total_customers
                    }
                    result['summary'] = f"Tổng quan khách hàng: {total_customers} khách hàng"
            
            return result
            
        except Exception as e:
            logger.error(f"Lỗi xử lý customer query: {str(e)}")
            return {'error': str(e)}
    
    def _handle_product_query(self, query: str, context: Optional[Dict]) -> Dict:
        """Xử lý query về sản phẩm"""
        try:
            df = self.load_data('product_analysis')
            
            result = {
                'query_type': 'product_analysis',
                'summary': '',
                'data': {},
                'insights': []
            }
            
            # Sản phẩm bán chạy
            if 'bán chạy' in query or 'top' in query:
                top_products = df.groupby('product_name')['revenue'].sum().sort_values(ascending=False).head(10)
                result['data']['top_products'] = top_products.to_dict()
                result['summary'] = f"Top 10 sản phẩm bán chạy nhất"
                result['insights'].append(f"Sản phẩm bán chạy nhất: {top_products.index[0]}")
            
            # Theo danh mục
            elif 'danh mục' in query or 'category' in query:
                category_sales = df.groupby('category_name')['revenue'].sum().sort_values(ascending=False)
                result['data']['category_sales'] = category_sales.to_dict()
                result['summary'] = f"Doanh số theo danh mục"
            
            # Theo thương hiệu
            elif 'thương hiệu' in query or 'brand' in query:
                brand_sales = df.groupby('brand_desc')['revenue'].sum().sort_values(ascending=False)
                result['data']['brand_sales'] = brand_sales.to_dict()
                result['summary'] = f"Doanh số theo thương hiệu"
            
            # Mặc định
            else:
                total_products = df['product_code'].nunique()
                total_revenue = df['revenue'].sum()
                total_quantity = df['quantity_sold'].sum()
                
                result['data'] = {
                    'total_products': total_products,
                    'total_revenue': total_revenue,
                    'total_quantity': total_quantity
                }
                result['summary'] = f"Tổng quan sản phẩm: {total_products} sản phẩm, doanh số {total_revenue:,.0f}"
            
            return result
            
        except Exception as e:
            logger.error(f"Lỗi xử lý product query: {str(e)}")
            return {'error': str(e)}
    
    def _handle_sales_trend_query(self, query: str, context: Optional[Dict]) -> Dict:
        """Xử lý query về xu hướng bán hàng"""
        try:
            df = self.load_data('sales_trend')
            
            result = {
                'query_type': 'sales_trend',
                'summary': '',
                'data': {},
                'insights': []
            }
            
            # Theo tháng
            if 'tháng' in query or 'monthly' in query:
                monthly_sales = df.groupby('month')['total_revenue'].sum()
                result['data']['monthly_sales'] = monthly_sales.to_dict()
                result['summary'] = f"Doanh số theo tháng"
                
                # Tìm tháng cao nhất
                best_month = monthly_sales.idxmax()
                result['insights'].append(f"Tháng có doanh số cao nhất: Tháng {best_month}")
            
            # Theo quý
            elif 'quý' in query or 'quarter' in query:
                quarterly_sales = df.groupby('quarter')['total_revenue'].sum()
                result['data']['quarterly_sales'] = quarterly_sales.to_dict()
                result['summary'] = f"Doanh số theo quý"
            
            # Theo ngày trong tuần
            elif 'ngày' in query or 'day' in query:
                daily_sales = df.groupby('day_of_week')['total_revenue'].sum()
                result['data']['daily_sales'] = daily_sales.to_dict()
                result['summary'] = f"Doanh số theo ngày trong tuần"
            
            # Mặc định
            else:
                total_revenue = df['total_revenue'].sum()
                avg_daily = df['total_revenue'].mean()
                growth = df['revenue_growth'].mean() * 100 if 'revenue_growth' in df.columns else 0
                
                result['data'] = {
                    'total_revenue': total_revenue,
                    'avg_daily_revenue': avg_daily,
                    'avg_growth_rate': growth
                }
                result['summary'] = f"Tổng doanh số: {total_revenue:,.0f}, tăng trưởng trung bình: {growth:.1f}%"
            
            return result
            
        except Exception as e:
            logger.error(f"Lỗi xử lý sales trend query: {str(e)}")
            return {'error': str(e)}
    
    def _handle_churn_query(self, query: str, context: Optional[Dict]) -> Dict:
        """Xử lý query về churn"""
        try:
            df = self.load_data('customer_retention')
            
            result = {
                'query_type': 'churn_prediction',
                'summary': '',
                'data': {},
                'insights': []
            }
            
            # Tỷ lệ churn
            if 'churn' in df.columns:
                churn_rate = df['churn'].mean() * 100
                result['data']['churn_rate'] = churn_rate
                result['summary'] = f"Tỷ lệ khách hàng rời bỏ: {churn_rate:.1f}%"
                result['insights'].append(f"Có {df['churn'].sum()} khách hàng có nguy cơ rời bỏ")
            
            # Khách hàng có nguy cơ cao
            if 'at_risk' in query or 'nguy cơ' in query:
                at_risk = df[df['churn_risk'] == 'High'] if 'churn_risk' in df.columns else pd.DataFrame()
                result['data']['at_risk_customers'] = len(at_risk)
                result['insights'].append(f"{len(at_risk)} khách hàng có nguy cơ rời bỏ cao")
            
            return result
            
        except Exception as e:
            logger.error(f"Lỗi xử lý churn query: {str(e)}")
            return {'error': str(e)}
    
    def _handle_forecast_query(self, query: str, context: Optional[Dict]) -> Dict:
        """Xử lý query về dự báo"""
        try:
            result = {
                'query_type': 'sales_forecast',
                'summary': 'Dự báo doanh số',
                'data': {},
                'insights': []
            }
            
            # Đọc dữ liệu lịch sử
            df = self.load_data('sales_trend')
            
            # Tính dự báo đơn giản (trung bình)
            avg_daily = df['total_revenue'].mean()
            
            # Dự báo 30 ngày tới
            forecast_30d = avg_daily * 30
            
            result['data'] = {
                'avg_daily_sales': avg_daily,
                'forecast_30_days': forecast_30d,
                'forecast_90_days': avg_daily * 90
            }
            result['summary'] = f"Dự báo doanh số 30 ngày tới: {forecast_30d:,.0f}"
            result['insights'].append(f"Doanh số trung bình mỗi ngày: {avg_daily:,.0f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Lỗi xử lý forecast query: {str(e)}")
            return {'error': str(e)}
    
    def _handle_general_query(self, query: str, context: Optional[Dict]) -> Dict:
        """Xử lý query chung"""
        return {
            'query_type': 'general',
            'summary': 'Câu hỏi chung về dữ liệu',
            'data': {},
            'insights': ['Vui lòng hỏi cụ thể về khách hàng, sản phẩm, xu hướng, hoặc dự báo']
        }
    
    def get_query_history(self) -> List[Dict]:
        """Lấy lịch sử queries"""
        return self.query_history
    
    def format_response(self, result: Dict) -> str:
        """
        Format kết quả thành text dễ đọc
        
        Args:
            result: Dict kết quả
            
        Returns:
            String response
        """
        try:
            response = []
            
            if 'error' in result:
                return f"Lỗi: {result['error']}"
            
            response.append(f"📊 {result.get('summary', 'Kết quả truy vấn')}")
            response.append("")
            
            # Insights
            if result.get('insights'):
                response.append("💡 Insights:")
                for insight in result['insights']:
                    response.append(f"  • {insight}")
                response.append("")
            
            # Data summary
            if result.get('data'):
                response.append("📈 Dữ liệu:")
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
            
            return "\n".join(response)
            
        except Exception as e:
            logger.error(f"Lỗi format response: {str(e)}")
            return f"Lỗi format: {str(e)}"