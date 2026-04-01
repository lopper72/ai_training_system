# AI Training System for Sales Data Analysis

Hệ thống AI training để phân tích dữ liệu bán hàng từ bảng `scm_sal_main` và `scm_sal_data`.

## Cấu trúc dự án

```
ai_training_system/
├── config/
│   ├── database.json          # Cấu hình kết nối database
│   └── mapping.json           # JSON mapping cho data transformation
├── data/
│   ├── raw/                   # Dữ liệu gốc từ DB
│   ├── processed/             # Dữ liệu đã xử lý
│   └── models/                # Models đã train
├── src/
│   ├── extractors/            # Module trích xuất dữ liệu
│   │   ├── database_extractor.py
│   │   └── sales_extractor.py
│   ├── transformers/          # Module chuyển đổi dữ liệu
│   │   ├── data_transformer.py
│   │   └── feature_engineer.py
│   ├── trainers/              # Module training ML
│   │   ├── model_trainer.py
│   │   ├── churn_predictor.py
│   │   └── sales_forecaster.py
│   └── query/                 # Module query AI
│       └── ai_query_interface.py
├── scripts/
│   └── scheduled_training.py  # Script chạy theo lịch
├── logs/                      # Log files
├── requirements.txt
├── main.py                    # Entry point
└── README.md
```

## Tính năng chính

1. **Data Extraction**: Trích xuất dữ liệu từ scm_sal_main và scm_sal_data
2. **Data Transformation**: Chuyển đổi và làm sạch dữ liệu theo JSON mapping
3. **Feature Engineering**: Tạo features cho ML models (RFM, temporal, product, customer)
4. **ML Training**: Training các mô hình ML:
   - **Churn Prediction**: Dự đoán khách hàng rời bỏ
   - **Sales Forecast**: Dự đoán doanh số bán hàng
   - **Customer Segmentation**: Phân khúc khách hàng
5. **AI Query**: Giao diện prompt để hỏi AI về dữ liệu
6. **Scheduled Training**: Chạy training tự động theo lịch (daily/weekly)

## Cài đặt

### 1. Cài đặt dependencies

```bash
cd ai_training_system
pip install -r requirements.txt
```

### 2. Cấu hình database

Chỉnh sửa file `config/database.json`:

```json
{
  "database": {
    "type": "postgresql",
    "host": "your_host",
    "port": 5432,
    "database": "your_database",
    "username": "your_username",
    "password": "your_password"
  }
}
```

### 3. Tạo thư mục cần thiết

```bash
mkdir -p data/processed data/models logs
```

## Sử dụng

### 1. Extract Data

Trích xuất và làm sạch dữ liệu từ database:

```bash
# Extract data tháng hiện tại
python main.py extract

# Extract data khoảng thời gian cụ thể
python main.py extract --date-from 2025-01-01 --date-to 2025-12-31
```

### 2. Train Models

Training các mô hình ML:

```bash
# Train tất cả models
python main.py train

# Train chỉ churn prediction
python main.py train --model churn

# Train chỉ sales forecast
python main.py train --model forecast
```

### 3. Query AI

Hỏi AI về dữ liệu:

```bash
# Interactive mode
python main.py query --interactive

# Single query
python main.py query --query "Top khách hàng mua nhiều nhất"
python main.py query --query "Xu hướng doanh số theo tháng"
python main.py query --query "Sản phẩm bán chạy nhất"
python main.py query --query "Dự báo doanh số 30 ngày tới"
```

### 4. Scheduled Training

Chạy training tự động theo lịch:

```bash
python main.py scheduled
```

Lịch trình:
- **Daily**: 02:00 hàng ngày - Incremental training
- **Weekly**: 03:00 Chủ nhật - Full retrain

## JSON Mapping Configuration

File `config/mapping.json` định nghĩa cách chuyển đổi dữ liệu:

### Cấu trúc chính:

```json
{
  "database_tables": {
    "scm_sal_main": {
      "columns": {
        "column_name": {
          "type": "string|decimal|date",
          "description": "Mô tả",
          "ai_usage": "identifier|filter|target_variable"
        }
      }
    }
  },
  "data_transformation": {
    "customer_analysis": {
      "source_tables": ["scm_sal_main", "scm_sal_data"],
      "join_condition": "scm_sal_main.uniquenum_pri = scm_sal_data.uniquenum_pri",
      "output_columns": ["customer_id", "transaction_date", "amount"],
      "filters": {"tag_void_yn": "n"}
    }
  },
  "ml_models": {
    "customer_churn_prediction": {
      "algorithm": "RandomForestClassifier",
      "features": ["recency", "frequency", "monetary"],
      "target": "is_churned"
    }
  }
}
```

### Thêm module mới:

1. Thêm bảng mới vào `database_tables`
2. Thêm transformation vào `data_transformation`
3. Thêm model vào `ml_models`
4. Chạy lại extraction và training

## ML Models

### 1. Churn Prediction

**Mục đích**: Dự đoán khách hàng có nguy cơ rời bỏ

**Features**:
- Recency: Số ngày từ lần mua cuối
- Frequency: Số lần mua hàng
- Monetary: Tổng giá trị mua hàng
- Avg Order Value: Giá trị trung bình mỗi đơn
- Days Since Last Purchase

**Output**:
- `churn_prediction`: 0 (Không rời bỏ) / 1 (Rời bỏ)
- `churn_risk`: Low / High

### 2. Sales Forecast

**Mục đích**: Dự đoán doanh số bán hàng

**Features**:
- Temporal: year, month, quarter, day_of_week
- Cyclical: month_sin, month_cos, day_of_week_sin, day_of_week_cos
- Rolling: revenue_ma_7, revenue_ma_30

**Output**:
- `forecasted_revenue`: Doanh số dự đoán

### 3. Customer Segmentation

**Mục đích**: Phân khúc khách hàng theo RFM

**Features**:
- Recency Score (1-5)
- Frequency Score (1-5)
- Monetary Score (1-5)

**Segments**:
- Champion: RFM cao
- Potential Loyalist: RFM trung bình cao
- Needs Attention: RFM trung bình
- At Risk: RFM thấp

## AI Query Examples

### Khách hàng:
- "Top 10 khách hàng mua nhiều nhất"
- "Tỷ lệ khách hàng mua lại"
- "Khách hàng có nguy cơ rời bỏ cao"
- "Phân khúc khách hàng"

### Sản phẩm:
- "Sản phẩm bán chạy nhất"
- "Doanh số theo danh mục"
- "Doanh số theo thương hiệu"
- "Sản phẩm nên nhập thêm"

### Xu hướng:
- "Xu hướng doanh số theo tháng"
- "Doanh số theo quý"
- "Ngày trong tuần có doanh số cao nhất"
- "Tăng trưởng doanh số"

### Dự báo:
- "Dự báo doanh số 30 ngày tới"
- "Dự báo doanh số 90 ngày tới"
- "Kế hoạch nhập hàng dựa trên dự báo"

## Logging

Logs được lưu trong thư mục `logs/`:
- `scheduled_training.log`: Log của scheduled training

## Mở rộng hệ thống

### Thêm data source mới:

1. Tạo extractor mới trong `src/extractors/`
2. Thêm mapping trong `config/mapping.json`
3. Tạo transformer nếu cần
4. Cập nhật `main.py`

### Thêm ML model mới:

1. Tạo trainer mới trong `src/trainers/`
2. Thêm model config trong `config/mapping.json`
3. Cập nhật scheduled training nếu cần
4. Thêm query handler trong `ai_query_interface.py`

### Thêm query type mới:

1. Thêm keywords trong `_classify_query()`
2. Tạo handler method `_handle_xxx_query()`
3. Thêm vào `process_query()`

## Troubleshooting

### Lỗi kết nối database:
- Kiểm tra `config/database.json`
- Đảm bảo database đang chạy
- Kiểm tra firewall/network

### Lỗi training:
- Kiểm tra dữ liệu có đủ không (>100 records)
- Kiểm tra missing values
- Xem logs trong `logs/`

### Lỗi query:
- Đảm bảo đã chạy extraction trước
- Kiểm tra file data trong `data/processed/`
- Xem query history

## License

Internal use only.
