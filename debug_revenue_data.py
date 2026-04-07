"""Dump and debug revenue_report_by_date dataset"""
import pandas as pd
from src.query.ai_query_interface import AIQueryInterface

print("=== DEBUG revenue_report_by_date ===")

# Load dataset directly
df = pd.read_parquet('data/processed/revenue_report_by_date.parquet')

print("\nFile exists, shape:", df.shape)
print("\nColumns:", list(df.columns))

print("\nAll data in dataset:")
print("=" * 80)
print(df.to_string())

print("\nCheck February 2010 specifically:")
feb_2010 = df[(df['report_year'] == 2010) & (df['report_month'] == 2)]
if len(feb_2010) > 0:
    print(feb_2010)
else:
    print("No data for February 2010 in dataset")

print("\nTest query directly:")
interface = AIQueryInterface()
result = interface.process_query("Show me revenue for February 2010")

print("\nQuery result summary:", result.get('summary', 'NO SUMMARY'))
print("\nData keys:", list(result.get('data', {}).keys()))

if result.get('data'):
    print("\nFull data:", result['data'])
