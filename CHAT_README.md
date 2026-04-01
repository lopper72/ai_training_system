# AI Chat Interface - Hướng dẫn sử dụng

## 🤖 Giới thiệu

Đây là giao diện chat AI sử dụng Google Gemini API để người dùng có thể trò chuyện và nhận hỗ trợ về:
- Phân tích dữ liệu bán hàng
- Giải thích các khái niệm ML/AI
- Hỗ trợ coding và debug
- Tư vấn cải thiện doanh số

## 📋 Yêu cầu

1. **Python 3.7+**
2. **Google Gemini API Key** - Đăng ký tại: https://makersuite.google.com/app/apikey
3. **Thư viện Python**: `google-generativeai`

## 🚀 Cài đặt

### Bước 1: Cài đặt thư viện

```bash
pip install google-generativeai
```

Hoặc thêm vào `requirements.txt`:
```
google-generativeai>=0.3.0
```

### Bước 2: Lấy API Key

1. Truy cập: https://makersuite.google.com/app/apikey
2. Đăng nhập bằng tài khoản Google
3. Tạo API Key mới
4. Copy API Key

### Bước 3: Cấu hình API Key

**Cách 1: Sử dụng Environment Variable (Khuyến nghị)**

Windows:
```cmd
set GEMINI_API_KEY=your_api_key_here
```

Linux/Mac:
```bash
export GEMINI_API_KEY=your_api_key_here
```

**Cách 2: Chỉnh sửa file gemini_server.py**

Mở file `gemini_server.py` và thay đổi dòng:
```python
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'YOUR_API_KEY_HERE')
```

Thành:
```python
GEMINI_API_KEY = 'your_actual_api_key_here'
```

## 🎯 Sử dụng

### Bước 1: Khởi động Server

```bash
python gemini_server.py
```

Server sẽ chạy trên port 9001.

### Bước 2: Mở giao diện chat

Mở browser và truy cập: **http://localhost:9001**

### Bước 3: Bắt đầu chat

Nhập câu hỏi và nhấn "Gửi" hoặc nhấn Enter.

## 💡 Tính năng

### Giao diện
- ✅ Thiết kế đẹp, responsive
- ✅ Hiển thị tin nhắn theo thời gian thực
- ✅ Typing indicator khi AI đang suy nghĩ
- ✅ Quick actions cho các câu hỏi phổ biến
- ✅ Xóa lịch sử chat
- ✅ Error handling

### Quick Actions
- 📊 **Churn Prediction** - Giải thích về dự đoán khách hàng rời bỏ
- 💰 **Cải thiện doanh số** - Tư vấn tăng doanh thu
- 👥 **Phân tích KH** - Hướng dẫn phân tích dữ liệu khách hàng
- 🌲 **Random Forest** - Giải thích thuật toán Random Forest

### Ví dụ câu hỏi
- "Giải thích về churn prediction là gì?"
- "Làm thế nào để cải thiện doanh số bán hàng?"
- "Phân tích dữ liệu khách hàng như thế nào?"
- "Random Forest hoạt động ra sao?"
- "Tôi có lỗi Python, giúp tôi debug"
- "Xu hướng kinh doanh năm nay thế nào?"

## 🔧 Troubleshooting

### Lỗi: "Cannot connect to server"
- Đảm bảo server đang chạy: `python gemini_server.py`
- Kiểm tra port 9001 có bị chiếm không

### Lỗi: "API Key not valid"
- Kiểm tra API Key đã được set đúng chưa
- Đảm bảo API Key còn hạn sử dụng
- Thử tạo API Key mới

### Lỗi: "Module not found: google.generativeai"
```bash
pip install google-generativeai
```

### Lỗi: Port already in use
Thay đổi port trong file `gemini_server.py`:
```python
PORT = 9002  # Thay đổi port khác
```

## 📝 Lưu ý

- API Key là miễn phí nhưng có giới hạn số lượng request
- Mỗi câu hỏi sẽ消耗 một lượng API quota
- Nên sử dụng Environment Variable để bảo mật API Key
- Server chỉ chạy trên localhost, không truy cập từ mạng khác

## 🎨 Tùy chỉnh

### Thay đổi màu sắc
Chỉnh sửa CSS trong file `gemini_chat.html`:
```css
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```

### Thêm quick actions
Thêm button mới trong HTML:
```html
<button class="quick-btn" onclick="sendQuickMessage('Câu hỏi của bạn')">🎯 Tiêu đề</button>
```

### Thay đổi system prompt
Chỉnh sửa trong file `gemini_server.py`:
```python
system_prompt = """Prompt tùy chỉnh của bạn"""
```

## 📞 Hỗ trợ

Nếu gặp vấn đề, vui lòng:
1. Kiểm tra lỗi trong console browser (F12)
2. Kiểm tra lỗi trong terminal chạy server
3. Đảm bảo tất cả dependencies đã được cài đặt

## 🔄 Cập nhật

Để cập nhật thư viện Gemini:
```bash
pip install --upgrade google-generativeai
```

---

**Chúc bạn sử dụng vui vẻ! 🚀**