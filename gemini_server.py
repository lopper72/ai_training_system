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

# Configure OpenRouter API
# You need to set your API key here or use environment variable
# Get free API key at: https://openrouter.ai/keys
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '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')

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
        """Get response from OpenRouter API with AI Query Interface integration"""
        try:
            # Get company-specific AI interface for data isolation
            ai_interface = get_ai_interface(companyfn)
            
            # Check if this is a data query
            data_keywords = ['customer', 'product', 'sales', 'trend', 'forecast', 'churn', 
                           'analysis', 'top', 'bestseller', 'revenue', 'order']
            
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
            return f"Sorry, I encountered an error while processing your question: {str(e)}"

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