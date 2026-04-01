#!/usr/bin/env python
"""
Quick test for Gemini Chat System
"""

import sys
import subprocess

def test_imports():
    """Test if required packages are installed"""
    print("Testing imports...")
    
    try:
        from openai import OpenAI
        print("[OK] openai imported successfully")
    except ImportError:
        print("[FAIL] openai not found")
        print("  Run: pip install openai")
        return False
    
    try:
        import http.server
        import socketserver
        print("[OK] http.server and socketserver imported successfully")
    except ImportError:
        print("[FAIL] http.server or socketserver not found")
        return False
    
    return True

def test_files():
    """Test if required files exist"""
    print("\nTesting files...")
    
    import os
    
    files_to_check = [
        'gemini_chat.html',
        'gemini_server.py',
        'CHAT_README.md'
    ]
    
    all_exist = True
    for file in files_to_check:
        if os.path.exists(file):
            print(f"[OK] {file} exists")
        else:
            print(f"[FAIL] {file} not found")
            all_exist = False
    
    return all_exist

def test_api_key():
    """Test if API key is configured"""
    print("\nTesting API key configuration...")
    
    import os
    
    # Check environment variable
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    if api_key and api_key != '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':
        print("[OK] OPENROUTER_API_KEY is set in environment")
        return True
    
    # Check hardcoded key in gemini_server.py
    try:
        with open('gemini_server.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'OPENROUTER_API_KEY' in content and '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' not in content:
                print("[OK] API key is configured in gemini_server.py")
                return True
    except:
        pass
    
    print("[FAIL] OPENROUTER_API_KEY not set or is placeholder")
    print("  Get free API key at: https://openrouter.ai/keys")
    print("  Then set your API key:")
    print("  Windows: set OPENROUTER_API_KEY=your_key_here")
    print("  Linux/Mac: export OPENROUTER_API_KEY=your_key_here")
    print("  Or edit gemini_server.py and add your API key")
    return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Gemini Chat System - Quick Test")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    if test_imports():
        tests_passed += 1
    
    if test_files():
        tests_passed += 1
    
    if test_api_key():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {tests_passed}/{total_tests} passed")
    print("=" * 50)
    
    if tests_passed == total_tests:
        print("\n[OK] All tests passed! System is ready.")
        print("\nTo start the chat server:")
        print("  python gemini_server.py")
        print("\nThen open: http://localhost:9001")
    else:
        print("\n[FAIL] Some tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()