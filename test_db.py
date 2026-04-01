#!/usr/bin/env python
"""Test database connection and data"""

from src.extractors.database_extractor import DatabaseExtractor
from datetime import datetime

print("Bat dau test database...")

try:
    db = DatabaseExtractor('config/database.json')
    print("Da ket noi database thanh cong")
    
    # Test 1: Ngay hien tai
    print(f"Ngay hien tai: {datetime.now().strftime('%Y-%m-%d')}")
    
    # Test 2: Ngay giao dich
    result = db.execute_query("SELECT MIN(date_trans), MAX(date_trans) FROM scm_sal_main WHERE tag_void_yn = 'n' AND tag_table_usage = 'sal_inv'")
    print(f"Ngay giao dich: tu {result[0][0]} den {result[0][1]}")
    
    # Test 3: So ngay tu giao dich cuoi den hien tai
    result = db.execute_query("SELECT EXTRACT(DAYS FROM CURRENT_DATE - MAX(date_trans)) FROM scm_sal_main WHERE tag_void_yn = 'n' AND tag_table_usage = 'sal_inv'")
    print(f"So ngay tu giao dich cuoi den hien tai: {result[0][0]}")
    
    # Test 4: Phan phoi is_churned
    result = db.execute_query("""
        WITH customer_stats AS (
            SELECT 
                party_unique as customer_id,
                MAX(date_trans) as last_purchase_date
            FROM scm_sal_main
            WHERE tag_void_yn = 'n' AND tag_table_usage = 'sal_inv'
            GROUP BY party_unique
        )
        SELECT 
            CASE 
                WHEN EXTRACT(DAYS FROM CURRENT_DATE - last_purchase_date) > 90 THEN 1 
                ELSE 0 
            END as is_churned,
            COUNT(*) as cnt
        FROM customer_stats
        GROUP BY is_churned
    """)
    print("Phan phoi is_churned:")
    for row in result:
        print(f"  is_churned={row[0]}: {row[1]} customers")
    
    # Test 5: Tong so records
    result = db.execute_query("SELECT COUNT(*) FROM scm_sal_main")
    print(f"Tong so records trong scm_sal_main: {result[0][0]}")
    
    db.close()
    print("Da dong ket noi database")
    
except Exception as e:
    print(f"Loi: {e}")
    import traceback
    traceback.print_exc()