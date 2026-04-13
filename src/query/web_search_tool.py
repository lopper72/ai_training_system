"""
Web Search Tool - Tra cứu xu hướng thị trường thế giới cho các sản phẩm.
Sử dụng DuckDuckGo Search (miễn phí, không cần API key).

v2: Thêm search_global_top_product() để tìm sản phẩm bán chạy nhất thế giới
    trong cùng ngành với sản phẩm nội bộ top 1.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


def search_market_trends(product_names_str: str) -> str:
    """
    Tra cứu xu hướng thị trường thế giới cho danh sách sản phẩm.

    Hỗ trợ 2 chế độ qua prefix trong input:
    - Mặc định: tra cứu trend cho từng sản phẩm trong danh sách
    - Prefix "TOP_GLOBAL: <product> | ...": tìm sản phẩm bán chạy nhất thế giới
      trong cùng ngành với <product>, để gợi ý chiến lược mua hàng năm sau.

    Args:
        product_names_str: Tên các sản phẩm cách nhau bởi dấu phẩy.
                           Hoặc "TOP_GLOBAL: <product_name> | <label>"
                           Ví dụ: "ACMV System, Fire Protection, Lift Installation"
                           Ví dụ: "TOP_GLOBAL: ACMV System | top selling global product in same industry"

    Returns:
        Chuỗi tóm tắt thông tin thị trường (bằng tiếng Việt).
    """
    # ── Chế độ TOP_GLOBAL: tìm sản phẩm bán chạy nhất thế giới theo ngành ──
    if product_names_str.strip().upper().startswith("TOP_GLOBAL:"):
        try:
            # Parse tên sản phẩm sau "TOP_GLOBAL:"
            after_prefix = product_names_str.split(":", 1)[1]
            top_product = after_prefix.split("|")[0].strip()
            return search_global_top_product(top_product)
        except Exception as e:
            logger.warning(f"[WebSearch] TOP_GLOBAL parse error: {e}")
            return f"Error processing TOP_GLOBAL request: {e}"

    # ── Chế độ mặc định: tra cứu trend cho từng sản phẩm ───────────────────
    try:
        from ddgs import DDGS

        # Parse danh sách sản phẩm
        products = [p.strip() for p in product_names_str.split(",") if p.strip()]
        if not products:
            return "Không có tên sản phẩm để tra cứu."

        # Giới hạn tối đa 5 sản phẩm để tránh quá nhiều request
        products = products[:5]

        results_summary = []
        results_summary.append("--- GLOBAL MARKET TRENDS ---\n")

        ddgs = DDGS()
        for product in products:
            try:
                query = f"{product} global market trend 2024 2025 growth demand"
                search_results = list(ddgs.text(query, max_results=3))

                if search_results:
                    snippets = []
                    for r in search_results:
                        body = r.get("body", "").strip()
                        if body and len(body) > 30:
                            snippets.append(body[:200])

                    combined = " | ".join(snippets[:2]) if snippets else "Không tìm thấy thông tin."
                    results_summary.append(f"📦 {product}:")
                    results_summary.append(f"   {combined}\n")
                else:
                    results_summary.append(f"📦 {product}: No market data found.\n")

            except Exception as e:
                logger.warning(f"[WebSearch] Error searching '{product}': {e}")
                results_summary.append(f"📦 {product}: Lỗi tra cứu - {str(e)[:80]}\n")

        return "\n".join(results_summary)

    except ImportError:
        return "Thư viện duckduckgo-search chưa được cài. Chạy: pip install duckduckgo-search"
    except Exception as e:
        logger.error(f"[WebSearch] Search failed: {e}")
        return f"Market search failed: {str(e)}"


def search_global_top_product(product_name: str) -> str:
    """
    Tìm sản phẩm bán chạy nhất thế giới trong cùng ngành với sản phẩm nội bộ.

    Chiến lược:
    1. Tìm ngành kinh doanh của product_name (ví dụ: ACMV → MEP Engineering)
    2. Tìm sản phẩm / dịch vụ bán chạy nhất toàn cầu trong ngành đó
    3. Trả về tên + lý do ngắn gọn để giúp công ty quyết định nhập hàng

    Args:
        product_name: Tên sản phẩm nội bộ top 1 (ví dụ: "ACMV System")

    Returns:
        Chuỗi gợi ý chiến lược mua hàng năm sau (tiếng Anh).
    """
    try:
        from ddgs import DDGS
        results_summary = []
        results_summary.append("--- STRATEGIC RECOMMENDATION FOR NEXT YEAR ---\n")

        ddgs = DDGS()

        # ── Bước 1: Xác định ngành của sản phẩm ─────────────────────────────
        industry_query = f'"{product_name}" industry sector market segment'
        industry_results = list(ddgs.text(industry_query, max_results=2))

        industry_context = ""
        if industry_results:
            for r in industry_results:
                body = r.get("body", "").strip()
                if body and len(body) > 30:
                    industry_context += body[:150] + " "
            industry_context = industry_context.strip()

        results_summary.append(f"🏭 Sản phẩm nội bộ top 1: **{product_name}**")
        if industry_context:
            results_summary.append(f"   Ngành: {industry_context[:200]}\n")

        # ── Bước 2: Tìm sản phẩm bán chạy nhất thế giới trong ngành đó ──────
        global_query = (
            f"{product_name} industry top selling product worldwide 2024 2025 "
            f"best market demand global trend"
        )
        global_results = list(ddgs.text(global_query, max_results=4))

        if global_results:
            results_summary.append("🌍 Sản phẩm bán chạy nhất thế giới trong cùng ngành:")
            snippets_added = 0
            for r in global_results:
                body = r.get("body", "").strip()
                title = r.get("title", "").strip()
                if body and len(body) > 40 and snippets_added < 3:
                    results_summary.append(f"   • {title}: {body[:220]}")
                    snippets_added += 1

            results_summary.append("")
        else:
            results_summary.append(f"🌍 No global data found for the industry of {product_name}.\n")

        # ── Bước 3: Tìm dự báo tăng trưởng ngành ────────────────────────────
        forecast_query = (
            f"{product_name} market growth forecast 2025 2026 CAGR demand"
        )
        forecast_results = list(ddgs.text(forecast_query, max_results=2))

        if forecast_results:
            results_summary.append("📈 Dự báo tăng trưởng ngành:")
            for r in forecast_results[:2]:
                body = r.get("body", "").strip()
                if body and len(body) > 40:
                    results_summary.append(f"   • {body[:200]}")
            results_summary.append("")

        # ── Conclusion ─────────────────────────────────────────────────────────
        results_summary.append(
            "📌 Recommendation: Based on global trends, the company should consider "
            f"investing more in the **{product_name}** category and related growing products "
            "for next year's planning."
        )

        return "\n".join(results_summary)

    except ImportError:
        return "Thư viện duckduckgo-search chưa được cài. Chạy: pip install duckduckgo-search"
    except Exception as e:
        logger.error(f"[WebSearch] search_global_top_product failed: {e}")
        return f"Tra cứu sản phẩm toàn cầu thất bại: {str(e)}"
