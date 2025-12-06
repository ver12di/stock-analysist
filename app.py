import textwrap
from numbers import Number
from typing import Dict, Optional

import akshare as ak
import pandas as pd
import streamlit as st
from google.generativeai import GenerativeModel, configure
from openai import OpenAI


def load_secret(key: str, label: str, help_text: str = "", default: str = "") -> str:
    """Fetch a secret from st.secrets or fall back to a sidebar input."""
    if key in st.secrets:
        return st.secrets[key]
    return st.sidebar.text_input(label, value=default, type="password", help=help_text)


def get_api_keys() -> Dict[str, str]:
    st.sidebar.header("🔑 API 密钥配置")
    doubao_key = load_secret(
        "doubao_api_key",
        "豆包 API Key",
        "用于豆包评级 (火山方舟 Ark)",
    )
    doubao_base_url = load_secret(
        "doubao_base_url",
        "豆包 Base URL",
        "可选，默认火山方舟 https://ark.cn-beijing.volces.com/api/v3",
        default="https://ark.cn-beijing.volces.com/api/v3",
    )
    deepseek_key = load_secret(
        "deepseek_api_key",
        "Deepseek Key",
        "用于 DeepSeek-V3.2 思考模式审计 (OpenAI 兼容)",
    )
    deepseek_base_url = load_secret(
        "deepseek_base_url",
        "Deepseek Base URL",
        "可选，默认官方 https://api.deepseek.com",
        default="https://api.deepseek.com",
    )
    gemini_key = load_secret(
        "gemini_api_key",
        "Gemini API Key",
        "用于 Gemini 乐观建构 (google-generativeai)",
    )
    return {
        "doubao_key": doubao_key.strip(),
        "doubao_base_url": doubao_base_url.strip(),
        "deepseek_key": deepseek_key.strip(),
        "deepseek_base_url": deepseek_base_url.strip(),
        "gemini_key": gemini_key.strip(),
    }


def build_openai_client(api_key: str, base_url: Optional[str] = None) -> Optional[OpenAI]:
    if not api_key:
        return None
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    try:
        return OpenAI(**kwargs)
    except Exception:
        return None


def build_gemini_client(api_key: str) -> Optional[GenerativeModel]:
    if not api_key:
        return None
    try:
        configure(api_key=api_key)
        return GenerativeModel("gemini-2.5-pro")
    except Exception:
        return None


def fetch_spot_info(code: str) -> Optional[Dict]:
    """Fetch spot info with a broad Akshare snapshot to avoid single-endpoint failures."""

    def _from_spot_snapshot() -> Optional[Dict]:
        try:
            snapshot_df = ak.stock_zh_a_spot_em()
        except Exception:
            return None

        if snapshot_df is None or snapshot_df.empty:
            return None

        row = snapshot_df[snapshot_df["代码"].astype(str).str.fullmatch(code)]
        if row.empty:
            return None

        record = row.iloc[0]
        return {
            "名称": record.get("名称", "未知"),
            "所属行业": record.get("所属行业", "未知"),
            "最新价": record.get("最新价", 0),
            "涨跌幅": record.get("涨跌幅", 0),
            "市盈率-动态": record.get("市盈率-动态", "N/A"),
            "总市值": record.get("总市值", "N/A"),
        }

    def _from_individual_and_hist() -> Optional[Dict]:
        try:
            info_df = ak.stock_individual_info_em(symbol=code)
            if info_df is None or info_df.empty:
                return None

            info_dict = {}
            for _, row in info_df.iterrows():
                info_dict[row.iloc[0]] = row.iloc[1]

            hist_df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
            if hist_df is None or hist_df.empty:
                return None

            latest = hist_df.iloc[-1]

            return {
                "名称": info_dict.get("股票简称", "未知"),
                "所属行业": info_dict.get("行业", "未知"),
                "最新价": latest.get("收盘", 0),
                "涨跌幅": latest.get("涨跌幅", 0),
                "市盈率-动态": info_dict.get("市盈率-动态", "N/A"),
                "总市值": info_dict.get("总市值", "N/A"),
            }
        except Exception:
            return None

    spot_info = _from_spot_snapshot() or _from_individual_and_hist()
    if spot_info is None:
        st.warning("实时行情获取失败: Akshare 接口返回空数据")
    return spot_info


def fetch_fund_flow(code: str) -> pd.DataFrame:
    try:
        df = ak.stock_individual_fund_flow(stock=code)
        if df is None:
            return pd.DataFrame()
        # 只保留近 10 日
        df = df.head(10)
        return df
    except Exception as exc:
        st.warning(f"主力资金流获取失败: {exc}")
        return pd.DataFrame()


def fetch_lhb(code: str) -> pd.DataFrame:
    try:
        df = ak.stock_lhb_stock_detail_date_em(symbol=code)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.sort_values(by="交易日", ascending=False)
        return df.head(5)
    except Exception as exc:
        st.warning(f"龙虎榜数据获取失败: {exc}")
        return pd.DataFrame()


def fetch_financials(code: str) -> pd.DataFrame:
    try:
        df = ak.stock_financial_abstract(symbol=code)
        return df if df is not None else pd.DataFrame()
    except Exception as exc:
        st.warning(f"财务摘要获取失败: {exc}")
        return pd.DataFrame()


def fetch_business_and_products(code: str) -> pd.DataFrame:
    try:
        df = ak.stock_zyjs_ths(symbol=code)
        return df if df is not None else pd.DataFrame()
    except Exception as exc:
        st.warning(f"主营业务获取失败: {exc}")
        return pd.DataFrame()


def fetch_company_profile(code: str) -> Dict[str, str]:
    try:
        info_df = ak.stock_individual_info_em(symbol=code)
        if info_df is None or info_df.empty:
            return {}
        info_dict: Dict[str, str] = {}
        for _, row in info_df.iterrows():
            info_dict[str(row.iloc[0])] = row.iloc[1]
        return {
            "行业": info_dict.get("行业", "未知"),
            "总股本": info_dict.get("总股本", "N/A"),
            "上市日期": info_dict.get("上市日期", "N/A"),
            "股票简称": info_dict.get("股票简称", "未知"),
            "市盈率-动态": info_dict.get("市盈率-动态", "N/A"),
            "总市值": info_dict.get("总市值", "N/A"),
        }
    except Exception as exc:
        st.warning(f"公司概况获取失败: {exc}")
        return {}


def fetch_recent_news(code: str, limit: int = 5) -> pd.DataFrame:
    try:
        df = ak.stock_news_em(symbol=code)
        if df is None:
            return pd.DataFrame()
        return df.head(limit)
    except Exception as exc:
        st.warning(f"新闻获取失败: {exc}")
        return pd.DataFrame()


def fetch_financial_indicators(code: str) -> pd.DataFrame:
    try:
        df = ak.stock_financial_analysis_indicator(symbol=code)
        return df if df is not None else pd.DataFrame()
    except Exception as exc:
        st.warning(f"关键财务指标获取失败: {exc}")
        return pd.DataFrame()


def fetch_valuation_indicator(code: str) -> pd.DataFrame:
    try:
        df = ak.stock_a_lg_indicator(symbol=code)
        return df if df is not None else pd.DataFrame()
    except Exception as exc:
        st.warning(f"估值分位获取失败: {exc}")
        return pd.DataFrame()


def fetch_sector_perf() -> pd.DataFrame:
    try:
        df = ak.stock_board_industry_name_em()
        return df if df is not None else pd.DataFrame()
    except Exception as exc:
        st.warning(f"行业表现获取失败: {exc}")
        return pd.DataFrame()


def gemini_bull_case(
    model: Optional[GenerativeModel],
    stock_name: str,
    sector: str,
    business_desc: str = "",
    news_digest: str = "",
) -> str:
    if not model:
        return "Gemini 未配置 API Key，无法生成乐观论点。"
    prompt = textwrap.dedent(
        f"""
        你是“市场情绪猎手”，关注 A 股题材炒作与政策催化。
        股票名称: {stock_name or '未知'}
        所属行业: {sector or '未知'}
        公司主营业务与产品: {business_desc or '未提供'}
        近期新闻摘要: {news_digest or '未提供'}
        请构造一段牛市故事：
        - 仅依据界面展示的硬数据生成，不得杜撰或引用外部未提供的信息
        - 热点叙事与政策想象
        - 媒体与资金可能放大的亮点
        - 情绪驱动的潜在超预期
        若缺少必要数据，请明确标注“数据不足”。
        输出以要点形式呈现。
        """
    )
    try:
        response = model.generate_content(prompt)
        return response.text or "Gemini 未返回内容。"
    except Exception as exc:
        return f"Gemini 调用失败: {exc}"


def deepseek_attack(client: Optional[OpenAI], base_prompt: str, data_points: str) -> str:
    if not client:
        return "Deepseek 未配置 API Key，无法进行审计。"
    messages = [
        {
            "role": "system",
            "content": "你是毫不留情的量化审计师，只输出严厉的风险点。",
        },
        {
            "role": "user",
            "content": (
                "Gemini 的牛市叙事如下：\n"
                f"{base_prompt}\n\n"
                "结合以下硬数据（重点关注 ROE、毛利率、估值分位、净利润增速），列出至少 3 个风险或矛盾，格式为要点：\n"
                f"{data_points}"
            ),
        },
    ]
    try:
        resp = client.chat.completions.create(
            model="deepseek-reasoner", messages=messages
        )
        return resp.choices[0].message.content
    except Exception as exc:
        return f"Deepseek 调用失败: {exc}"


def doubao_judge(
    client: Optional[OpenAI], hope: str, audit: str
) -> str:
    if not client:
        return "豆包未配置 API Key，无法给出评级。"
    messages = [
        {
            "role": "system",
            "content": "你是投资组合经理，需要给出 Buy/Wait/Sell 评级与信心分数 (0-100)。",
        },
        {
            "role": "user",
            "content": (
                "乐观观点：\n" + hope + "\n\n"
                "审计观点：\n" + audit + "\n\n"
                "请综合给出最终评级、理由与执行建议。"
            ),
        },
    ]
    try:
        resp = client.chat.completions.create(
            model="doubao-seed-1-6-thinking-250715", messages=messages
        )
        return resp.choices[0].message.content
    except Exception as exc:
        return f"豆包调用失败: {exc}"


def render_fund_flow_chart(df: pd.DataFrame):
    if df.empty:
        st.info("暂无主力资金流数据")
        return
    chart_df = df.copy()
    date_col = "日期" if "日期" in chart_df.columns else chart_df.columns[0]
    value_col = (
        "主力净流入-净额"
        if "主力净流入-净额" in chart_df.columns
        else chart_df.columns[-1]
    )
    chart_df[date_col] = pd.to_datetime(chart_df[date_col])
    chart_df = chart_df.sort_values(date_col)
    st.line_chart(chart_df.set_index(date_col)[value_col])


def _format_number(value, decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    if isinstance(value, Number):
        return f"{float(value):,.{decimals}f}"
    try:
        numeric_value = float(str(value).replace(",", ""))
        return f"{numeric_value:,.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_percentage(value, decimals: int = 2) -> str:
    formatted = _format_number(value, decimals)
    return formatted if formatted == "N/A" else f"{formatted}%"


def df_to_markdown(df: pd.DataFrame, max_rows: int = 5) -> str:
    if df is None or df.empty:
        return "无数据"
    limited = df.head(max_rows)
    try:
        return limited.to_markdown(index=False)
    except Exception:
        return limited.to_string(index=False)


def format_data_points(spot: Optional[Dict], flow_df: pd.DataFrame, lhb_df: pd.DataFrame, fin_df: pd.DataFrame) -> str:
    lines = []
    if spot:
        lines.append(
            "最新价: "
            f"{_format_number(spot.get('最新价', 'N/A'))}"
            " | 涨跌幅: "
            f"{_format_percentage(spot.get('涨跌幅', 'N/A'))}"
        )
        lines.append(
            "市盈率: "
            f"{_format_number(spot.get('市盈率-动态', 'N/A'))}"
            " | 总市值: "
            f"{_format_number(spot.get('总市值', 'N/A'), decimals=0)}"
        )
    if not flow_df.empty:
        latest = flow_df.iloc[0]
        lines.append(
            "近一日主力净额: "
            f"{_format_number(latest.get('主力净流入-净额', 'N/A'))}"
            " ("
            f"{latest.get('日期', '')}"
            ")"
        )
    if not lhb_df.empty:
        lines.append(f"龙虎榜上榜次数: {len(lhb_df)}")

    def pick_metric(df: pd.DataFrame, keywords) -> Optional[str]:
        if df.empty:
            return None
        subject_col = df.columns[0]
        if len(df.columns) < 2:
            return None
        latest_period_col = df.columns[1]
        for keyword in keywords:
            hit = df[df[subject_col].astype(str).str.contains(keyword, na=False)]
            if not hit.empty:
                return hit.iloc[0].get(latest_period_col)
        return None

    if not fin_df.empty:
        roe = pick_metric(fin_df, ["ROE", "净资产收益率", "加权ROE"])
        net_profit = pick_metric(fin_df, ["净利润", "归母净利润"])
        if roe is not None or net_profit is not None:
            lines.append(
                "ROE: "
                f"{_format_percentage(roe) if roe is not None else 'N/A'}"
                " | 净利润: "
                f"{_format_number(net_profit, decimals=0) if net_profit is not None else 'N/A'}"
            )
    return "\n".join(lines) or "无可用数据"


def stock_verification_flow(api_keys: Dict[str, str]):
    st.header("模式一：个股验证 (Construct-Destroy-Rebuild)")
    code = st.text_input("输入股票代码 (如 000001)")
    if st.button("启动三人小组"):
        if not code:
            st.error("请先输入股票代码")
            return

        with st.status("⏳ 正在抓取深度基本面与新闻…", state="running") as status:
            profile = fetch_company_profile(code)
            spot = fetch_spot_info(code)
            flow_df = fetch_fund_flow(code)
            lhb_df = fetch_lhb(code)
            fin_df = fetch_financials(code)
            business_df = fetch_business_and_products(code)
            news_df = fetch_recent_news(code)
            fin_indicator_df = fetch_financial_indicators(code)
            valuation_df = fetch_valuation_indicator(code)
            status.update(label="✅ 基本面数据就绪，开始建模", state="complete")

        stock_name = (
            profile.get("股票简称")
            if profile
            else (spot.get("名称", "未知") if spot else "未知")
        )
        sector = profile.get("行业", "未知") if profile else (spot.get("所属行业", "未知") if spot else "未知")
        data_points = format_data_points(spot, flow_df, lhb_df, fin_df)

        def summarize_business(df: pd.DataFrame) -> str:
            if df.empty:
                return "未获取到主营业务信息"
            lines = []
            business_col = next((c for c in df.columns if "主营" in str(c)), None)
            product_col = next((c for c in df.columns if "产品" in str(c)), None)
            for _, row in df.head(5).iterrows():
                biz = row.get(business_col, "") if business_col else ""
                product = row.get(product_col, "") if product_col else ""
                lines.append(f"- {biz or '业务'} / {product or '产品'}")
            return "\n".join(lines)

        def summarize_news(df: pd.DataFrame) -> str:
            if df.empty:
                return "暂无新闻"
            title_col = next((c for c in df.columns if "标题" in str(c)), df.columns[0])
            summary_col = next((c for c in df.columns if "摘要" in str(c) or "内容" in str(c)), None)
            items = []
            for _, row in df.iterrows():
                title = row.get(title_col, "")
                summary = row.get(summary_col, "") if summary_col else ""
                items.append(f"- {title}: {summary}")
            return "\n".join(items)

        def summarize_financial_indicators(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return pd.DataFrame()
            columns_priority = ["报告期", "销售毛利率", "净资产收益率", "扣非净利润同比增长(%)"]
            selected_cols = [c for c in columns_priority if c in df.columns]
            if selected_cols:
                return df[selected_cols].head(3)
            return df.head(3)

        business_desc = summarize_business(business_df)
        news_digest = summarize_news(news_df)
        fin_indicator_focus = summarize_financial_indicators(fin_indicator_df)
        valuation_focus = valuation_df.head(3) if not valuation_df.empty else pd.DataFrame()

        st.subheader(f"{stock_name} ({code}) 数据概览")
        overview_data = {
            "价格": spot.get("最新价") if spot else "N/A",
            "涨跌幅": spot.get("涨跌幅") if spot else "N/A",
            "PE(TTM)": profile.get("市盈率-动态") if profile else spot.get("市盈率-动态") if spot else "N/A",
            "总市值": profile.get("总市值") if profile else spot.get("总市值") if spot else "N/A",
            "行业": sector,
            "总股本": profile.get("总股本", "N/A") if profile else "N/A",
            "上市日期": profile.get("上市日期", "N/A") if profile else "N/A",
        }
        st.write(overview_data)
        if not fin_df.empty:
            st.markdown("**财务摘要 (部分)**")
            st.dataframe(fin_df.head())
        if not lhb_df.empty:
            st.markdown("**龙虎榜记录 (部分)**")
            st.dataframe(lhb_df.head())

        st.markdown("**主力资金流趋势**")
        render_fund_flow_chart(flow_df)

        with st.expander("📊 查看深度基本面数据"):
            st.markdown("**主营业务与产品**")
            if business_df.empty:
                st.info("暂无主营业务数据")
            else:
                st.dataframe(business_df)

            st.markdown("**关键财务指标**")
            if fin_indicator_df.empty:
                st.info("暂无关键财务指标")
            else:
                st.dataframe(fin_indicator_df)

            st.markdown("**估值分位 (乐咕)**")
            if valuation_df.empty:
                st.info("暂无估值分位数据")
            else:
                st.dataframe(valuation_df)

            st.markdown("**近期新闻 (Top 5)**")
            if news_df.empty:
                st.info("暂无新闻数据")
            else:
                st.dataframe(news_df)

        gemini_model = build_gemini_client(api_keys.get("gemini_key", ""))
        deepseek_client = build_openai_client(
            api_keys.get("deepseek_key", ""), api_keys.get("deepseek_base_url", "")
        )
        doubao_client = build_openai_client(
            api_keys.get("doubao_key", ""), api_keys.get("doubao_base_url", "")
        )

        gemini_business_text = business_desc or "未提供主营业务"
        gemini_news_text = news_digest or "未提供新闻"
        deepseek_data = "\n".join(
            [
                "基础行情与资金:",
                data_points,
                "\n关键财务指标:",
                df_to_markdown(fin_indicator_focus, max_rows=3),
                "\n估值分位:",
                df_to_markdown(valuation_focus, max_rows=3),
            ]
        )

        with st.spinner("Gemini 正在基于主营和新闻构建叙事…"):
            bull_case = gemini_bull_case(
                gemini_model, stock_name, sector, gemini_business_text, gemini_news_text
            )
        with st.spinner("Deepseek 正在用硬指标拆解…"):
            attack = deepseek_attack(deepseek_client, bull_case, deepseek_data)
        with st.spinner("豆包正在综合评级…"):
            verdict = doubao_judge(doubao_client, bull_case, attack)

        col1, col2, col3 = st.columns(3)
        col1.markdown("### 🚀 Gemini 乐观论点")
        col1.write(bull_case)

        col2.markdown("### 🛡️ Deepseek 审计")
        col2.write(attack)

        col3.markdown("### ⚖️ 豆包评级")
        col3.write(verdict)


def sector_rotation_flow(api_keys: Dict[str, str]):
    st.header("模式二：行业轮动")
    if st.button("扫描行业"):
        sector_df = fetch_sector_perf()
        if sector_df.empty:
            st.error("未获取到行业数据")
            return

        top_sectors = sector_df.head(5)
        st.dataframe(top_sectors)

        sector_text = "\n".join(
            [
                f"{row['行业名称']}: 涨跌幅 {row.get('涨跌幅', 'N/A')}%"
                for _, row in top_sectors.iterrows()
            ]
        )

        gemini_model = build_gemini_client(api_keys.get("gemini_key", ""))
        deepseek_client = build_openai_client(
            api_keys.get("deepseek_key", ""), api_keys.get("deepseek_base_url", "")
        )
        doubao_client = build_openai_client(
            api_keys.get("doubao_key", ""), api_keys.get("doubao_base_url", "")
        )

        with st.spinner("Deepseek 正在扫描强势板块…"):
            deepseek_msg = deepseek_attack(
                deepseek_client,
                "行业强势列表",
                f"高景气行业：\n{sector_text}",
            )
        with st.spinner("Gemini 正在添加宏观背景…"):
            macro_msg = gemini_bull_case(
                gemini_model,
                "市场热点组合",
                "政策与外部宏观",
            )
        with st.spinner("豆包正在生成次日脚本…"):
            script_msg = doubao_judge(doubao_client, macro_msg, deepseek_msg)

        col1, col2, col3 = st.columns(3)
        col1.markdown("### 🚀 Gemini 宏观亮点")
        col1.write(macro_msg)

        col2.markdown("### 🛡️ Deepseek 板块要点")
        col2.write(deepseek_msg)

        col3.markdown("### ⚖️ 豆包次日剧本")
        col3.write(script_msg)


def main():
    st.set_page_config(page_title="A股三位一体指挥舱", layout="wide")
    st.title("A股三位一体指挥舱 🚀🛡️⚖️")
    st.caption("Construct - Destroy - Rebuild: Gemini × Deepseek × 豆包")

    api_keys = get_api_keys()

    mode = st.sidebar.radio("选择模式", ["个股验证", "行业轮动"])
    if mode == "个股验证":
        stock_verification_flow(api_keys)
    else:
        sector_rotation_flow(api_keys)

    st.sidebar.markdown(
        """
        **使用提示**
        - 未配置 st.secrets 时，可在侧边栏输入密钥
        - 所有数据抓取均做了异常捕获，网络不稳定时不会导致崩溃
        - 建议在 Streamlit Cloud 部署时设置 st.secrets
        """
    )


if __name__ == "__main__":
    main()
