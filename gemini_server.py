#!/usr/bin/env python
"""
AI Chat Server - Backend for AI Chat Interface (LangChain + Groq)
Integrated with AI Training System
"""

import http.server
import socketserver
import json
import os
import sys
import re

# Add root directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PORT = 9001

# ANSI: bright red diagnostics so [AIQuery] lines stand out in the terminal.
_STYLE_RED = "\033[91m"
_STYLE_RESET = "\033[0m"


def _enable_windows_console_ansi() -> None:
    """Enable VT100 escape sequences in classic Windows console (PowerShell/CMD)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
            return
        enable_vt = 0x0004
        if kernel32.SetConsoleMode(handle, mode.value | enable_vt) == 0:
            return
    except Exception:
        pass


def _print_aiquery_diag(message: str) -> None:
    """Print server-side [AIQuery] hints in a distinct color when stdout is a TTY."""
    line = (message or "").rstrip("\n")
    if not line:
        return
    if sys.stdout.isatty():
        print(f"{_STYLE_RED}{line}{_STYLE_RESET}", flush=True)
    else:
        print(line, flush=True)


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

class GeminiChatHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Disable caching so frontend always uses latest JS/HTML behavior.
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def _send_json(self, status_code: int, payload: dict):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode('utf-8'))

    @staticmethod
    def _print_data_path_hint(ai_result: dict) -> None:
        """Terminal-only hint: intent planner vs LangChain pandas agent."""
        if not isinstance(ai_result, dict) or "error" in ai_result:
            return
        src = ai_result.get("source") or ""
        cfg = ai_result.get("intent_planner_config_enabled")
        if src == "planner_executor":
            pi = ai_result.get("plan_intent") or "-"
            _print_aiquery_diag(f"[AIQuery] data_path=intent_planner  plan_intent={pi}")
            return
        if src == "langchain_agent":
            extra = (
                "planner_on_but_fell_back_to_agent"
                if cfg
                else "intent_planner_off"
            )
            _print_aiquery_diag(f"[AIQuery] data_path=agent_ai  note={extra}")
            return
        if src == "deterministic_churn":
            _print_aiquery_diag("[AIQuery] data_path=deterministic_churn  (sales_main MoM/YoY)")
            return
        if src == "language_policy":
            _print_aiquery_diag(
                f"[AIQuery] data_path=none  reason=language_only  intent_planner_config={cfg}"
            )
            return
        if src in ("rate_limited", "no_data"):
            _print_aiquery_diag(
                f"[AIQuery] data_path=agent_ai  outcome={src}  intent_planner_config={cfg}"
            )
            return
        _print_aiquery_diag(f"[AIQuery] source={src}")

    @staticmethod
    def is_english_query(text):
        if not text or not text.strip():
            return False
        has_latin = bool(re.search(r"[A-Za-z]", text))
        has_non_ascii = any(ord(ch) > 127 for ch in text)
        return has_latin and not has_non_ascii

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/health':
            payload = {
                "status": "ok",
                "service": "ai_training_system",
                "engine": "langchain_groq_hybrid",
                "interfaces_loaded": len(ai_interfaces),
            }
            return self._send_json(200, payload)

        if self.path == '/metrics':
            companyfn = None
            if ai_interfaces:
                companyfn = next(iter(ai_interfaces.keys()))
            ai_interface = get_ai_interface(companyfn)
            if not ai_interface:
                return self._send_json(503, {"status": "degraded", "error": "AI interface unavailable"})
            metrics = ai_interface.get_runtime_metrics()
            return self._send_json(200, metrics)

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
                request_id = data.get('request_id')
                # Extract companyfn from request context for data isolation
                companyfn = data.get('companyfn', None)
                session_id = data.get('session_id', None) or data.get('conversation_id', None)
                if not session_id:
                    session_id = self.headers.get('X-Session-Id', None)
                if not session_id:
                    session_id = self.client_address[0]

                if not self.is_english_query(user_message):
                    response_data = {
                        "success": True,
                        "response": "Please use English for your query. Non-English questions are not supported.",
                        "meta": {"source": "language_policy"},
                        "request_id": request_id,
                    }
                else:
                    response_data = self.get_agent_response(user_message, companyfn, session_id, request_id)
                
                # Send response
                self._send_json(200, response_data)
                
            except Exception as e:
                error_response = {
                    'success': False,
                    'error': str(e)
                }
                self._send_json(500, error_response)
        else:
            self.send_response(404)
            self.end_headers()
    
    def get_agent_response(self, user_message, companyfn=None, session_id=None, request_id=None):
        """Get response from internal hybrid query pipeline."""
        try:
            # Get company-specific AI interface for data isolation
            ai_interface = get_ai_interface(companyfn)

            if ai_interface:
                try:
                    context = {"companyfn": companyfn, "session_id": session_id}
                    ai_result = ai_interface.process_query(user_message, context=context)
                    _print_aiquery_diag(ai_interface.terminal_parquet_and_df_map_line())
                    GeminiChatHandler._print_data_path_hint(ai_result)
                    ai_response = ai_interface.format_response(ai_result)
                    return {
                        "success": True,
                        "response": ai_response,
                        "request_id": request_id,
                        "meta": {
                            "source": ai_result.get("source"),
                            "query_type": ai_result.get("query_type"),
                            "reason_code": ai_result.get("reason_code"),
                            "intent_router": ai_result.get("intent_router"),
                            "active_datasets": ai_result.get("active_datasets"),
                        },
                    }

                except Exception as e:
                    return {
                        "success": False,
                        "response": f"Sorry, I encountered an internal query error: {str(e)}",
                        "request_id": request_id,
                        "meta": {"source": "internal_error"},
                    }
            return {
                "success": False,
                "response": "AI query interface is unavailable. Please check server logs and LangChain/Groq configuration.",
                "request_id": request_id,
                "meta": {"source": "interface_unavailable"},
            }
            
        except Exception as e:
            return {
                "success": False,
                "response": f"Sorry, I encountered an error while processing your question: {str(e)}",
                "request_id": request_id,
                "meta": {"source": "server_error"},
            }

def main():
    """Start the server"""
    _enable_windows_console_ansi()
    print("Starting AI Chat Server (LangChain + Groq)...")
    print(f"Open your browser and go to: http://localhost:{PORT}")
    print(f"Press Ctrl+C to stop the server")
    print()
    
    with socketserver.TCPServer(("", PORT), GeminiChatHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    main()