# AI Chat Interface - User Guide

## 🤖 Introduction

This is an AI chat interface using Google Gemini API that allows users to chat and receive support for:
- Sales data analysis
- ML/AI concepts explanation
- Coding and debugging support
- Sales improvement consulting

## 📋 Requirements

1. **Python 3.7+**
2. **Google Gemini API Key** - Register at: https://makersuite.google.com/app/apikey
3. **Python Library**: `google-generativeai`

## 🚀 Installation

### Step 1: Install library

```bash
pip install google-generativeai
```

Or add to `requirements.txt`:
```
google-generativeai>=0.3.0
```

### Step 2: Get API Key

1. Visit: https://makersuite.google.com/app/apikey
2. Sign in with Google account
3. Create new API Key
4. Copy API Key

### Step 3: Configure API Key

**Method 1: Use Environment Variable (Recommended)**

```bash
# Windows
set GEMINI_API_KEY=your_api_key_here

# Linux/Mac
export GEMINI_API_KEY=your_api_key_here
```

**Method 2: Edit gemini_server.py file**

Open `gemini_server.py` and change the line:
```python
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'your_api_key_here')
```

To:
```python
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSy...your_actual_key...')
```

## 🎯 Usage

### Step 1: Start Server

```bash
python gemini_server.py
```

Server will run on port 9001.

### Step 2: Open chat interface

Open browser and access: **http://localhost:9001**

### Step 3: Start chatting

Enter your question and press "Send" or Enter.

## 💡 Features

### Interface
- ✅ Beautiful, responsive design
- ✅ Real-time message display
- ✅ Typing indicator when AI is thinking
- ✅ Quick actions for common questions
- ✅ Clear chat history
- ✅ Error handling

### Quick Actions
- 📊 **Churn Prediction** - Explain customer churn prediction
- 💰 **Improve Sales** - Revenue growth consulting
- 👥 **Customer Analysis** - Guide to customer data analysis
- 🌲 **Random Forest** - Explain Random Forest algorithm

### Example Questions
- "Explain what churn prediction is?"
- "How to improve sales revenue?"
- "How to analyze customer data?"
- "How does Random Forest work?"
- "I have Python error, help me debug"
- "What's the business trend this year?"

## 🔧 Troubleshooting

### Error: "Cannot connect to server"
- Ensure server is running: `python gemini_server.py`
- Check if port 9001 is occupied

### Error: "API Key not valid"
- Check if API Key is set correctly
- Ensure API Key is still valid
- Try creating new API Key

### Error: "Module not found: google.generativeai"
```bash
pip install google-generativeai
```

### Error: Port already in use
Change port in `gemini_server.py`:
```python
PORT = 9002  # Change to different port
```

## 📝 Notes

- API Key is free but has request limits
- Each question consumes API quota
- Should use Environment Variable to secure API Key
- Server only runs on localhost, cannot access from other networks

## 🎨 Customization

### Change colors
Edit CSS in `gemini_chat.html`:
```css
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```

### Add quick actions
Add new button in HTML:
```html
<button class="quick-btn" onclick="sendQuickMessage('Your question')">🎯 Title</button>
```

### Change system prompt
Edit in `gemini_server.py`:
```python
system_prompt = """Your custom prompt"""
```

## 📞 Support

If you encounter issues, please:
1. Check error in browser console (F12)
2. Check error in server terminal
3. Ensure all dependencies are installed

## 🔄 Update

To update Gemini library:
```bash
pip install --upgrade google-generativeai
```

---

**Have fun using it! 🚀**