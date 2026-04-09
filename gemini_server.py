#!/usr/bin/env python
"""
AI Chat Server - Backend for AI Chat Interface (OpenRouter Version)
Integrated with AI Training System
"""

import http.server
import socketserver
import json
import os
import sys
from openai import OpenAI
from datetime import datetime

# Add root directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PORT = 9001

# ✅ MULTI AI PROVIDER CONFIGURATION WITH AUTOMATIC FALLBACK
# Khi 1 cái hết quota/error, hệ thống sẽ tự động chuyển sang cái kế tiếp

AI_PROVIDERS = [
    # 1. OpenRouter (Mặc định - ưu tiên 1)
    {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv('OPENROUTER_API_KEY', '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'),
        "model": "openrouter/free",
        "enabled": True,
        "working": True
    },

    # 2. Google Gemini Official (MIỄN PHÍ 1M token/tháng - ưu tiên 2)
    {
        "name": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": os.getenv('GEMINI_API_KEY', ''),
        "model": "gemini-2.0-flash",
        "enabled": True,
        "working": True
    },

    # 3. Groq (MIỄN PHÍ cực nhanh - ưu tiên 3)
    {
        "name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": os.getenv('GROQ_API_KEY', ''),
        "model": "llama-3.1-8b-instant",
        "enabled": True,
        "working": True
    },

    # 4. 🔥 Ollama - RUN LOCAL OFFLINE 100% MIỄN PHÍ (tải model về rồi mới bật)
    {
        "name": "Ollama (Local)",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": "llama3.2",
        "enabled": False,
        "working": True,
        "local": True
    },
    
    # 2. Google Gemini Official (Miễn phí 15RPM / 1M token/tháng - ưu tiên 2)
    {
        "name": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": os.getenv('GEMINI_API_KEY', 'AIzaSyCkfB4X8Zp4n5qQz7vX9zK6mN2pP8sR0tU2wY4a'),
        "model": "gemini-2.0-flash",
        "enabled": True,
        "working": True
    },
    
    # 3. Groq (Miễn phí cực nhanh - ưu tiên 3)
    {
        "name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": os.getenv('GROQ_API_KEY', 'gsk_8bKfD5eG7hJ9kL0mN2pP4rT6vX8zB0dF2hJ4lM6n'),
        "model": "llama-3.1-8b-instant",
        "enabled": True,
        "working": True
    },
    
    # 4. Together.ai (Free credit $5 khi đăng ký mới)
    {
        "name": "Together.ai",
        "base_url": "https://api.together.xyz/v1",
        "api_key": os.getenv('TOGETHER_API_KEY', '79a2c48f1e3b7d9a0f6c2e5b8d4a7f1e3c9b2d4a6f8e0c2a'),
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "enabled": True,
        "working": True
    },
    
    # 5. OpenAI Official
    {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "api_key": os.getenv('OPENAI_API_KEY', 'sk-proj-abcdefghijklmnopqrstuvwxyz1234567890'),
        "model": "gpt-3.5-turbo",
        "enabled": True,
        "working": True
    }
]

# Trạng thái theo dõi provider nào đang hoạt động
current_provider_index = 0

# Import AI Query Interface
# Dictionary to store AI interfaces per company
ai_interfaces = {}

def get_ai_interface(companyfn=None):
    """Get or create AI interface for a specific company"""
    if companyfn not in ai_interfaces:
        try:
            from src.query.ai_query_interface import AIQueryInterface
            ai_interfaces[companyfn] = AIQueryInterface(companyfn=companyfn)
            if len(ai_interfaces) == 1:
                print("[OK] AI Query Interface loaded successfully")
        except Exception as e:
            print(f"[WARNING] AI Query Interface not available: {str(e)}")
            return None
    return ai_interfaces[companyfn]

AI_INTERFACE_AVAILABLE = True

class GeminiChatHandler(http.server.SimpleHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
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
                # Extract companyfn from request context for data isolation
                companyfn = data.get('companyfn', None)
                
                # Get response from OpenRouter
                response = self.get_openrouter_response(user_message, companyfn)
                
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
    
    def get_openrouter_response(self, user_message, companyfn=None):
        """
        ✅ SMART MULTI-PROVIDER AI RESPONSE WITH AUTOMATIC FALLBACK
        Khi 1 provider lỗi / hết quota / rate limit, hệ thống sẽ TỰ ĐỘNG chuyển sang cái kế tiếp
        Không có lỗi nữa, chỉ chuyển đổi liên tục cho đến khi có provider hoạt động
        """
        global current_provider_index
        
        try:
            # Get company-specific AI interface for data isolation
            ai_interface = get_ai_interface(companyfn)
            
            # Check if this is a data query
            data_keywords = [
                'customer', 'product', 'sales', 'trend', 'forecast', 'churn', 'analysis', 'top', 'bestseller', 'revenue', 'order',
                'month', 'year', 'date', 'daily', 'monthly', 'yearly', 'period',
                'january', 'february', 'march', 'april', 'may', 'june',
                'july', 'august', 'september', 'october', 'november', 'december',
                'triển vọng', 'xu hướng', 'phổ biến', 'nổi bật', 'tương lai', 'sắp tới', 'hot trend', 'tiềm năng'
            ]
            
            is_data_query = any(keyword in user_message.lower() for keyword in data_keywords)
            
            # If it's a data query and AI Interface is available
            if is_data_query and ai_interface:
                try:
                    # Use AI Query Interface to get insights (filtered by companyfn)
                    ai_result = ai_interface.process_query(user_message)
                    ai_response = ai_interface.format_response(ai_result)
                    
                    # Create prompt for OpenRouter with context from AI Interface
                    system_prompt = f"""You are an internal AI Assistant for the company's ERP/BI system.

                    Your role:
                    - Analyze business data
                    - Support decision making
                    - Explain data analysis results
                    - Answer questions related to sales, customers, products, employees, and business performance.
                    - If the user asks for a 'summary', prioritize mentioning Total Revenue, Total Transactions, and Growth Rate in a structured list.

                    - TIME FILTERING LOGIC:
                    - If the user asks for a specific month (e.g., "01/2010"), filter the DATA SOURCE to only include January 2010.
                    - If the user asks for a specific date (e.g., "28/12/2010"), focus strictly on that day's data.
                    - If the timeframe is not found in the DATA SOURCE, state that no records exist for that period.

                    DATA ACCESS PERMISSIONS

                    You are operating in the company's internal environment and are authorized to use data from the internal ERP and database systems.

                    Available data may include:
                    - Order data
                    - Revenue
                    - Customer data
                    - Product data
                    - Employee data
                    - Analysis reports
                    - Machine learning results

                    You do NOT need to refuse due to "no data access permission".
                    Consider the data provided in the system as valid internal data for analysis.

                    DATA SOURCE FOR ANALYSIS

                    Analysis results from the backend system are provided below:

                    {ai_response}

                    You must:
                    1. Identify if there is a specific Date or Month in the user's question.
                    2. Analyze the data from the results provided below *strictly* for that timeframe.
                    3. Summarize important insights and answer the user's question.

                    RESPONSE RULES

                    - Always base responses on provided data
                    - Do not fabricate data if it doesn't exist
                    - If data is insufficient, request additional information
                    - Explain clearly and understandably
                    - Prefer bullet points or tables when needed
                    - You may use emojis for friendliness but don't overuse them
                    - STRICT DATA SOURCE ADHERENCE: You are ONLY allowed to use metrics present in the {ai_response}. 
                    - ANTI-HALLUCINATION: If the data does not contain "Customer names", "Product names", or "Employee names", DO NOT mention or invent them. 
                    - NULL DATA HANDLING: If a metric (e.g., num_transactions) is 0 or null in the data source, report it as 0. Do not assume there is hidden data.
                    - REVENUE vs TRANSACTIONS: Use the 'amt_local' for revenue and 'num_transactions' for the count. If 'num_transactions' is available, use it to calculate 'Average Revenue per Transaction' as an extra insight.
                    - NO HISTORICAL ASSUMPTION: Do not compare with previous months unless the specific growth_rate or previous_data is provided in the {ai_response}.

                    LANGUAGE

                    - Always respond in English
                    - Clear, professional, and easy-to-understand tone"""
                    
                except Exception as e:
                    # If AI Interface fails, use default prompt
                    system_prompt = """You are an AI Assistant specializing in sales data analysis and machine learning.
                    You can:
                    - Answer questions about sales data
                    - Explain ML/AI concepts like churn prediction, sales forecast
                    - Support Python coding and debugging
                    - Analyze business trends
                    - Provide suggestions to improve sales

                    Respond in English, be friendly and easy to understand. Use emojis to create a friendly atmosphere."""
            else:
                # Default prompt for other questions
                system_prompt = """You are an internal AI Assistant for the company's ERP/BI system.

                Your role:
                - Analyze business data
                - Support decision making
                - Explain data analysis results
                - Answer questions related to sales, customers, products, employees, and business performance.

                DATA ACCESS PERMISSIONS

                You are operating in the company's internal environment and are authorized to use data from the internal ERP and database systems.

                Available data may include:
                - Order data
                - Revenue
                - Customer data
                - Product data
                - Employee data
                - Analysis reports
                - Machine learning results

                You do NOT need to refuse due to "no data access permission".
                Consider the data provided in the system as valid internal data for analysis.

                DATA SOURCE FOR ANALYSIS

                Analysis results from the backend system are provided below:


                You must:
                1. Analyze the data in these results
                2. Summarize important insights
                3. Answer the user's question based on the data

                RESPONSE RULES

                - Always base responses on provided data
                - Do not fabricate data if it doesn't exist
                - If data is insufficient, request additional information
                - Explain clearly and understandably
                - Prefer bullet points or tables when needed
                - You may use emojis for friendliness but don't overuse them

                LANGUAGE

                - Always respond in English
                - Clear, professional, and easy-to-understand tone"""
            
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

            # ✅ FALLBACK LOOP - TRY ALL PROVIDERS UNTIL ONE WORKS
            total_providers = len(AI_PROVIDERS)
            
            for attempt in range(total_providers):
                provider = AI_PROVIDERS[current_provider_index]
                
                if not provider['enabled'] or not provider['working']:
                    # Bỏ qua provider đã tắt / bị lỗi trước đó
                    current_provider_index = (current_provider_index + 1) % total_providers
                    continue
                
                try:
                    print(f"🔄 Đang thử kết nối với: {provider['name']}")
                    
                    # Create client cho provider hiện tại
                    client = OpenAI(
                        base_url=provider['base_url'],
                        api_key=provider['api_key'],
                        timeout=30
                    )
                    
                    # Thêm headers đặc biệt cho OpenRouter
                    extra_headers = {}
                    if provider['name'] == 'OpenRouter':
                        extra_headers = {
                            "HTTP-Referer": f"http://localhost:{PORT}",
                            "X-Title": "ERP AI Chat Assistant",
                        }
                    
                    # Gọi API
                    response = client.chat.completions.create(
                        model=provider['model'],
                        messages=messages,
                        max_tokens=1000,
                        temperature=0.7,
                        extra_headers=extra_headers
                    )
                    
                    # ✅ THÀNH CÔNG!
                    print(f"✅ Kết nối thành công với: {provider['name']}")
                    return response.choices[0].message.content
                    
                except Exception as provider_error:
                    # ❌ Provider này bị lỗi / hết quota
                    print(f"❌ Lỗi với {provider['name']}: {str(provider_error)[:100]}...")
                    print(f"🔀 Tự động chuyển sang provider kế tiếp...")
                    
                    # Đánh dấu provider này là không hoạt động cho đến lần restart
                    AI_PROVIDERS[current_provider_index]['working'] = False
                    
                    # Chuyển sang provider kế tiếp
                    current_provider_index = (current_provider_index + 1) % total_providers
                    
                    # Tiếp tục vòng lặp
                    continue
            
            # ❌ TẤT CẢ PROVIDER ĐỀU LỖI
            return "⚠️ Tất cả các dịch vụ AI hiện đang bận. Vui lòng thử lại sau vài phút."
            
        except Exception as e:
            return f"Sorry, I encountered an error while processing your question: {str(e)}"

def main():
    """Start the server"""
    print(f"Starting AI Chat Server (OpenRouter)...")
    print(f"Open your browser and go to: http://localhost:{PORT}")
    print(f"Press Ctrl+C to stop the server")
    print()
    
    # ✅ CHECK ALL PROVIDERS STATUS
    print("AI Providers Status:")
    print("-" * 60)
    
    for i, provider in enumerate(AI_PROVIDERS):
        status = "ENABLED" if provider['enabled'] else "DISABLED"
        print(f"  {i+1}. {provider['name']:15} | Model: {provider['model']:30} | {status}")
    
    print()
    print(f"System ready! Multi-provider fallback activated.")
    print(f"Khi OpenRouter het luot free se tu dong chuyen sang Gemini, Groq, Together.ai, OpenAI")
    print()
    
    with socketserver.TCPServer(("", PORT), GeminiChatHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    main()