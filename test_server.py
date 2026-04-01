#!/usr/bin/env python
"""
Simple HTTP Server for AI Training System Test Interface
"""

import http.server
import socketserver
import json
import subprocess
import os
import sys
from urllib.parse import urlparse, parse_qs

PORT = 9000

class TestRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/' or self.path == '/index.html':
            self.path = '/test_interface.html'
        return http.server.SimpleHTTPRequestHandler.do_GET(self)
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/execute':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                command = data.get('command', '')
                
                # Execute command
                result = self.execute_command(command)
                
                # Send response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = json.dumps(result)
                self.wfile.write(response.encode('utf-8'))
                
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
    
    def execute_command(self, command):
        """Execute a command and return the result"""
        try:
            # Change to the project directory
            project_dir = os.path.dirname(os.path.abspath(__file__))
            os.chdir(project_dir)
            
            # Execute command
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                return {
                    'success': True,
                    'output': stdout
                }
            else:
                return {
                    'success': False,
                    'error': stderr if stderr else stdout
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

def main():
    """Start the server"""
    print(f"Starting AI Training System Test Server...")
    print(f"Open your browser and go to: http://localhost:{PORT}")
    print(f"Press Ctrl+C to stop the server")
    print()
    
    with socketserver.TCPServer(("", PORT), TestRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    main()