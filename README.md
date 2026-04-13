# AI Training System for Sales Data Analysis

AI training system for analyzing sales data from `scm_sal_main` and `scm_sal_data` tables.

## Project Structure

```
ai_training_system/
├── config/
│   ├── database.json          # Database connection configuration
│   └── mapping.json           # JSON mapping for data transformation
├── data/
│   ├── raw/                   # Raw data from DB
│   ├── processed/             # Processed data
│   └── models/                # Trained models
├── src/
│   ├── extractors/            # Data extraction modules
│   │   ├── database_extractor.py
│   │   └── sales_extractor.py
│   ├── transformers/          # Data transformation modules
│   │   ├── data_transformer.py
│   │   └── feature_engineer.py
│   ├── trainers/              # ML training modules
│   │   ├── model_trainer.py
│   │   ├── churn_predictor.py
│   │   └── sales_forecaster.py
│   └── query/                 # AI query module
│       └── ai_query_interface.py
├── scripts/
│   └── scheduled_training.py  # Scheduled training script
├── logs/                      # Log files
├── requirements.txt
├── main.py                    # Entry point
└── README.md
```

## Key Features

1. **Data Extraction**: Extract data from scm_sal_main and scm_sal_data
2. **Data Transformation**: Transform and clean data according to JSON mapping
3. **Feature Engineering**: Create features for ML models (RFM, temporal, product, customer)
4. **ML Training**: Train ML models:
   - **Churn Prediction**: Predict customer churn
   - **Sales Forecast**: Predict sales revenue
   - **Customer Segmentation**: Segment customers
5. **AI Query**: Prompt-based interface to query AI about data
6. **Scheduled Training**: Run automatic training on schedule (daily/weekly)

## Installation

### 1. Install dependencies

```bash
cd ai_training_system
pip install -r requirements.txt
```

### 2. Configure database

Edit `config/database.json`:

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

### 3. Create necessary directories

```bash
mkdir -p data/processed data/models logs
```

## Usage

### 1. Extract Data

Extract and clean data from database:

```bash
# Extract current month data
python main.py extract

# Extract specific date range
python main.py extract --date-from 2025-01-01 --date-to 2025-12-31
```

### 2. Train Models

Train ML models:

```bash
# Train all models
python main.py train

# Train only churn prediction
python main.py train --model churn

# Train only sales forecast
python main.py train --model forecast
```

### 3. Query AI

Query AI about data:

```bash
# Interactive mode
python main.py query --interactive

# Single query
python main.py query --query "Top customers by purchases"
python main.py query --query "Monthly sales trends"
python main.py query --query "Bestselling products"
python main.py query --query "30-day sales forecast"
```

### 4. Scheduled Training

Run automatic training on schedule:

```bash
python main.py scheduled
```

Schedule:
- **Daily**: 02:00 daily - Incremental training
- **Weekly**: 03:00 Sunday - Full retrain

## JSON Mapping Configuration

`config/mapping.json` defines how to transform data:

### Main structure:

```json
{
  "database_tables": {
    "scm_sal_main": {
      "columns": {
        "column_name": {
          "type": "string|decimal|date",
          "description": "Description",
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

### Adding new module:

1. Add new table to `database_tables`
2. Add transformation to `data_transformation`
3. Add model to `ml_models`
4. Run extraction and training again

## ML Models

### 1. Churn Prediction

**Purpose**: Predict customers at risk of churning

**Features**:
- Recency: Days since last purchase
- Frequency: Number of purchases
- Monetary: Total purchase value
- Avg Order Value: Average order value
- Days Since Last Purchase

**Output**:
- `churn_prediction`: 0 (No churn) / 1 (Churn)
- `churn_risk`: Low / High

### 2. Sales Forecast

**Purpose**: Predict sales revenue

**Features**:
- Temporal: year, month, quarter, day_of_week
- Cyclical: month_sin, month_cos, day_of_week_sin, day_of_week_cos
- Rolling: revenue_ma_7, revenue_ma_30

**Output**:
- `forecasted_revenue`: Predicted revenue

### 3. Customer Segmentation

**Purpose**: Segment customers by RFM

**Features**:
- Recency Score (1-5)
- Frequency Score (1-5)
- Monetary Score (1-5)

**Segments**:
- Champion: High RFM
- Potential Loyalist: Medium-high RFM
- Needs Attention: Medium RFM
- At Risk: Low RFM

## AI Query Examples

### Customers:
- "Top 10 customers by purchases"
- "Customer repeat rate"
- "Customers at high risk of churning"
- "Customer segments"

### Products:
- "Bestselling products"
- "Sales by category"
- "Sales by brand"
- "Products to restock"

### Trends:
- "Monthly sales trends"
- "Quarterly sales"
- "Best day of week for sales"
- "Sales growth"

### Forecasts:
- "30-day sales forecast"
- "90-day sales forecast"
- "Inventory planning based on forecast"

## Logging

Logs are stored in `logs/`:
- `scheduled_training.log`: Scheduled training logs

## Extending the System

### Adding new data source:

1. Create new extractor in `src/extractors/`
2. Add mapping in `config/mapping.json`
3. Create transformer if needed
4. Update `main.py`

### Adding new ML model:

1. Create new trainer in `src/trainers/`
2. Add model config in `config/mapping.json`
3. Update scheduled training if needed
4. Add query handler in `ai_query_interface.py`

### Extending AI query capability:

1. Update prompt rules in `src/query/pandas_agent.py` (`AGENT_PREFIX`, `AGENT_FORMAT_INSTRUCTIONS`)
2. Update schema/context in `src/query/data_context.py`
3. Keep `src/query/ai_query_interface.py` as hybrid planner-first orchestration

## Troubleshooting

### Database connection error:
- Check `config/database.json`
- Ensure database is running
- Check firewall/network

### Training error:
- Check if data is sufficient (>100 records)
- Check for missing values
- View logs in `logs/`

### Query error:
- Ensure extraction has been run first
- Check data files in `data/processed/`
- View query history

## License

Internal use only.