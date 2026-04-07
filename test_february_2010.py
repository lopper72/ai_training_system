"""Test script to verify February 2010 revenue query works"""
from src.query.ai_query_interface import AIQueryInterface

def test_february_2010_query():
    print("Testing February 2010 revenue query...")
    
    query_interface = AIQueryInterface()
    
    # Test query for February 2010
    test_query = "Show me revenue for February 2010"
    
    result = query_interface.process_query(test_query)
    
    print("\nQuery Result:")
    print("=" * 60)
    print(query_interface.format_response(result))
    
    # Check if we have data
    if 'revenue_report_by_date' in result.get('data', {}):
        data = result['data']['revenue_report_by_date']
        print(f"\n✅ SUCCESS: February 2010 data retrieved!")
        print(f"   Total Revenue: ${data.get('total_revenue', 0):,.2f}")
        print(f"   Transactions: {data.get('transaction_count', 0)}")
        return True
    else:
        print(f"\n❌ FAILED: No data returned for February 2010")
        print(f"   Summary: {result.get('summary', 'No summary')}")
        return False

if __name__ == "__main__":
    success = test_february_2010_query()
    exit(0 if success else 1)