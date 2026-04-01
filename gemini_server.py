#!/usr/bin/env python
"""
AI Chat Server - Backend for AI Chat Interface (OpenRouter Version)
Tích hợp với AI Training System
"""

import http.server
import socketserver
import json
import os
import sys
from openai import OpenAI
from datetime import datetime

# Thêm thư mục gốc vào path để import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PORT = 9001

# Configure OpenRouter API
# You need to set your API key here or use environment variable
# Get free API key at: https://openrouter.ai/keys
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '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')

# Import AI Query Interface
try:
    from src.query.ai_query_interface import AIQueryInterface
    ai_interface = AIQueryInterface()
    AI_INTERFACE_AVAILABLE = True
    print("[OK] AI Query Interface loaded successfully")
except Exception as e:
    AI_INTERFACE_AVAILABLE = False
    print(f"[WARNING] AI Query Interface not available: {str(e)}")

class GeminiChatHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/' or self.path == '/index.html':
            self.path = '/gemini_chat.html'
        return http.server.SimpleHTTPRequestHandler.do_GET(self)
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/chat':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                user_message = data.get('message', '')
                
                # Get response from OpenRouter
                response = self.get_openrouter_response(user_message)
                
                # Send response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response_data = json.dumps({
                    'success': True,
                    'response': response
                })
                self.wfile.write(response_data.encode('utf-8'))
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                error_response = json.dumps({
                    'success': False,
                    'error': str(e)
                })
                self.wfile.write(error_response.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
    
    def get_openrouter_response(self, user_message):
        """Get response from OpenRouter API with AI Query Interface integration"""
        try:
            # Kiểm tra xem có phải query về dữ liệu không
            data_keywords = ['khách hàng', 'customer', 'sản phẩm', 'product', 'doanh số', 'sales', 
                           'xu hướng', 'trend', 'dự báo', 'forecast', 'churn', 'phân tích', 'analysis',
                           'top', 'bán chạy', 'doanh thu', 'revenue']
            
            is_data_query = any(keyword in user_message.lower() for keyword in data_keywords)
            
            # Nếu là query về dữ liệu và AI Interface có sẵn
            if is_data_query and AI_INTERFACE_AVAILABLE:
                try:
                    # Sử dụng AI Query Interface để lấy insights
                    ai_result = ai_interface.process_query(user_message)
                    ai_response = ai_interface.format_response(ai_result)
                    
                    # Tạo prompt cho OpenRouter với context từ AI Interface
                    system_prompt = f"""Bạn là AI Assistant nội bộ của hệ thống ERP/BI của công ty.

Vai trò của bạn:
- Phân tích dữ liệu kinh doanh
- Hỗ trợ ra quyết định
- Giải thích kết quả phân tích dữ liệu
- Trả lời câu hỏi liên quan đến bán hàng, khách hàng, sản phẩm, nhân viên và hiệu suất kinh doanh.

QUYỀN TRUY CẬP DỮ LIỆU

Bạn đang hoạt động trong môi trường nội bộ của công ty và được phép sử dụng dữ liệu từ hệ thống ERP và database nội bộ.

Các dữ liệu có thể bao gồm:
- dữ liệu đơn hàng
- doanh thu
- dữ liệu khách hàng
- dữ liệu sản phẩm
- dữ liệu nhân viên
- báo cáo phân tích
- kết quả machine learning

Bạn KHÔNG cần từ chối vì lý do "không có quyền truy cập dữ liệu".  
Hãy coi dữ liệu được cung cấp trong hệ thống là dữ liệu nội bộ hợp lệ để phân tích.

NGUỒN DỮ LIỆU PHÂN TÍCH

Kết quả phân tích từ hệ thống backend được cung cấp dưới đây:

{ai_response}

Bạn phải:
1. Phân tích dữ liệu trong kết quả này
2. Tóm tắt insight quan trọng
3. Trả lời đúng câu hỏi của người dùng dựa trên dữ liệu

QUY TẮC TRẢ LỜI

- Luôn dựa trên dữ liệu đã được cung cấp
- Không tự bịa dữ liệu nếu dữ liệu không tồn tại
- Nếu dữ liệu chưa đủ, hãy yêu cầu thêm thông tin
- Giải thích rõ ràng, dễ hiểu
- Ưu tiên liệt kê bằng bullet hoặc bảng nếu cần
- Có thể sử dụng emoji để thân thiện nhưng không lạm dụng

NGÔN NGỮ

- Luôn trả lời bằng tiếng Việt
- Văn phong rõ ràng, chuyên nghiệp, dễ hiểu"""
                    
                except Exception as e:
                    # Nếu AI Interface lỗi, sử dụng prompt mặc định
                    system_prompt = """Bạn là một AI Assistant chuyên về phân tích dữ liệu bán hàng và machine learning. 
Bạn có thể:
- Trả lời câu hỏi về dữ liệu bán hàng
- Giải thích các khái niệm ML/AI như churn prediction, sales forecast
- Hỗ trợ coding Python và debug
- Phân tích xu hướng kinh doanh
- Đưa ra gợi ý cải thiện doanh số

Hãy trả lời bằng tiếng Việt, thân thiện và dễ hiểu. Sử dụng emoji để tạo không khí thân thiện."""
            else:
                # Prompt mặc định cho các câu hỏi khác
                system_prompt = """Bạn là AI Assistant nội bộ của hệ thống ERP/BI của công ty.

Vai trò của bạn:
- Phân tích dữ liệu kinh doanh
- Hỗ trợ ra quyết định
- Giải thích kết quả phân tích dữ liệu
- Trả lời câu hỏi liên quan đến bán hàng, khách hàng, sản phẩm, nhân viên và hiệu suất kinh doanh.

QUYỀN TRUY CẬP DỮ LIỆU

Bạn đang hoạt động trong môi trường nội bộ của công ty và được phép sử dụng dữ liệu từ hệ thống ERP và database nội bộ.

Các dữ liệu có thể bao gồm:
- dữ liệu đơn hàng
- doanh thu
- dữ liệu khách hàng
- dữ liệu sản phẩm
- dữ liệu nhân viên
- báo cáo phân tích
- kết quả machine learning

Bạn KHÔNG cần từ chối vì lý do "không có quyền truy cập dữ liệu".  
Hãy coi dữ liệu được cung cấp trong hệ thống là dữ liệu nội bộ hợp lệ để phân tích.

NGUỒN DỮ LIỆU PHÂN TÍCH

Kết quả phân tích từ hệ thống backend được cung cấp dưới đây:


Bạn phải:
1. Phân tích dữ liệu trong kết quả này
2. Tóm tắt insight quan trọng
3. Trả lời đúng câu hỏi của người dùng dựa trên dữ liệu

QUY TẮC TRẢ LỜI

- Luôn dựa trên dữ liệu đã được cung cấp
- Không tự bịa dữ liệu nếu dữ liệu không tồn tại
- Nếu dữ liệu chưa đủ, hãy yêu cầu thêm thông tin
- Giải thích rõ ràng, dễ hiểu
- Ưu tiên liệt kê bằng bullet hoặc bảng nếu cần
- Có thể sử dụng emoji để thân thiện nhưng không lạm dụng

NGÔN NGỮ

- Luôn trả lời bằng tiếng Việt
- Văn phong rõ ràng, chuyên nghiệp, dễ hiểu"""
            
            # Create client with OpenRouter API
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            )
            
            # Create messages with context
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]
            
            # Generate response using OpenRouter API (free model)
            response = client.chat.completions.create(
                model="openrouter/free",
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Xin lỗi, tôi gặp lỗi khi xử lý câu hỏi: {str(e)}"

def main():
    """Start the server"""
    print(f"Starting AI Chat Server (OpenRouter)...")
    print(f"Open your browser and go to: http://localhost:{PORT}")
    print(f"Press Ctrl+C to stop the server")
    print()
    
    # Check API key
    if OPENROUTER_API_KEY == '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':
        print("WARNING: Please set your OPENROUTER_API_KEY!")
        print("   Get free API key at: https://openrouter.ai/keys")
        print("   Then set it as environment variable or edit gemini_server.py")
        print()
    
    with socketserver.TCPServer(("", PORT), GeminiChatHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    main()