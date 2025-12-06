import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

import akshare as ak
from openai import OpenAI
import google.generativeai as genai


st.set_page_config(
    page_title="A股三位一体指挥舱 | A-Share Trinity Command Center",
    layout="wide",
)


# ----------------------------
# Utility and Cache Helpers
# ----------------------------
@st.cache_data(show_spinner=False, ttl=600)
def get_stock_spot(code: str) -> pd.Series | None:
    df = ak.stock_zh_a_spot_em()
    df = df.rename(columns={"代码": "code"})
    row = df.loc[df["code"] == code]
    return row.iloc[0] if not row.empty else None


@st.cache_data(show_spinner=False, ttl=600)
def get_dragon_tiger(code: str) -> pd.DataFrame:
    return ak.stock_lhb_detail_em(symbol=code)


@st.cache_data(show_spinner=False, ttl=600)
def get_fund_flow(code: str) -> pd.DataFrame:
    return ak.stock_individual_fund_flow(stock=code)


@st.cache_data(show_spinner=False, ttl=600)
def get_hist_data(code: str, days: int = 30) -> pd.DataFrame:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days * 2)
    df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
        adjust="qfq",
    )
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values("日期").tail(days)
    return df


@st.cache_data(show_spinner=False, ttl=600)
def get_financials(code: str) -> pd.DataFrame:
    return ak.stock_financial_abstract(symbol=code)


@st.cache_data(show_spinner=False, ttl=600)
def get_northbound() -> pd.DataFrame:
    return ak.stock_hsgt_hist_em()


@st.cache_data(show_spinner=False, ttl=600)
def get_sector_perf() -> pd.DataFrame:
    return ak.stock_board_industry_name_em()


@st.cache_data(show_spinner=False, ttl=600)
def get_sector_fund_flow() -> pd.DataFrame:
    return ak.stock_sector_fund_flow_rank(indicator="今日")


# ----------------------------
# Model Helpers
# ----------------------------
def call_gemini(api_key: str, prompt: str) -> str:
    if not api_key:
        return "[缺少 Gemini API Key，无法生成乐观叙事。]"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text or "Gemini 未返回内容。"
    except Exception as err:  # noqa: BLE001
        return f"[Gemini 调用失败: {err}]"


def call_deepseek(base_url: str, api_key: str, prompt: str) -> str:
    if not base_url or not api_key:
        return "[缺少 Deepseek Base URL 或 API Key，无法进行严谨审计。]"
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content
    except Exception as err:  # noqa: BLE001
        return f"[Deepseek 调用失败: {err}]"


def call_chatgpt(api_key: str, prompt: str) -> str:
    if not api_key:
        return "[缺少 OpenAI API Key，无法给出投资决策。]"
    try:
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content
    except Exception as err:  # noqa: BLE001
        return f"[ChatGPT 调用失败: {err}]"


# ----------------------------
# Prompt Builders
# ----------------------------
def build_gemini_prompt(stock_name: str, sector: str, price_info: pd.Series, hist_df: pd.DataFrame) -> str:
    recent_change = price_info.get("涨跌幅") if price_info is not None else None
    latest_close = price_info.get("最新价") if price_info is not None else None
    recent_trend = hist_df.tail(5)["收盘"].tolist() if not hist_df.empty else []

    return f"""你是“多头叙事者”，需要为中国A股股票讲述一个乐观故事。
股票: {stock_name}
所属行业: {sector}
最新价: {latest_close}，日内涨跌幅: {recent_change}
最近5日收盘价: {recent_trend}
请给出可能的利好催化剂、市场情绪和资金面想象，形成一份看多报告。"""


def build_deepseek_prompt(gemini_view: str, price_info: pd.Series, flow_df: pd.DataFrame, lhb_df: pd.DataFrame, fin_df: pd.DataFrame, north_df: pd.DataFrame) -> str:
    latest_flow = flow_df.head(1).to_dict(orient="records") if not flow_df.empty else []
    lhb_count = len(lhb_df) if lhb_df is not None else 0
    roe = fin_df.loc[fin_df["指标"] == "净资产收益率", "2023"].max() if not fin_df.empty else None
    north_latest = north_df.tail(3).to_dict(orient="records") if not north_df.empty else []

    return f"""你是“冷酷审计师”，需要基于数据拆穿多头故事。
Gemini 的多头观点: {gemini_view}
价格信息: {price_info.to_dict() if price_info is not None else {}}
资金流向(最新): {latest_flow}
龙虎榜出现次数: {lhb_count}
财务指标(ROE示例): {roe}
北向资金最近3日: {north_latest}
请列出至少3条基于数据的风险和质疑点，要求严谨且直击要害。"""


def build_chatgpt_prompt(gemini_view: str, deepseek_view: str) -> str:
    return f"""你是投资组合经理，需要综合乐观叙事与冷酷审计。
多头叙事: {gemini_view}
审计意见: {deepseek_view}
请给出最终判定（买入/观望/卖出），并给出0-100的信心分。请简洁回答。"""


def build_sector_prompts(sector_df: pd.DataFrame, flow_df: pd.DataFrame) -> tuple[str, str, str]:
    top_sector = sector_df.sort_values("涨跌幅", ascending=False).iloc[0]
    flow_top = flow_df.sort_values("主力净额", ascending=False).iloc[0]

    deepseek_prompt = f"""你是量化分析师，基于板块表现和资金流向寻找最强板块。
行业涨幅榜首: {top_sector.to_dict()}
行业资金流向榜首: {flow_top.to_dict()}
请给出当前最强势板块及技术面理由。"""

    gemini_prompt = f"""你是宏观叙事者，基于最强板块提供宏观/政策背景。
最强板块: {top_sector.get('板块名称')}
请结合国内外政策、利率、汇率、商品价格等，推演对该板块的潜在影响。"""

    chatgpt_prompt = f"""你是交易脚本作者，需要为次日提供操作方案。
参考板块: {top_sector.get('板块名称')}
请输出“次日剧本”：高开/平开/低开场景下的买卖节奏。"""

    return deepseek_prompt, gemini_prompt, chatgpt_prompt


# ----------------------------
# UI Construction
# ----------------------------
st.title("A股三位一体指挥舱 | A-Share Trinity Command Center")
st.caption("构建-拆解-重构：Gemini × Deepseek × ChatGPT 协同分析")

with st.sidebar:
    st.header("🔑 API 配置")
    openai_key = st.text_input("OpenAI API Key", type="password")
    gemini_key = st.text_input("Google Gemini API Key", type="password")
    deepseek_base = st.text_input("Deepseek Base URL", placeholder="https://api.deepseek.com")
    deepseek_key = st.text_input("Deepseek API Key", type="password")

    st.header("🧭 模式选择")
    mode = st.selectbox("分析模式", ["个股验资", "板块轮动"])

if mode == "个股验资":
    stock_code = st.text_input("输入股票代码 (例如: 600000)", value="600000")

    if st.button("启动三位一体分析", type="primary"):
        with st.spinner("获取数据与调度模型中..."):
            spot_info = get_stock_spot(stock_code)
            if spot_info is None:
                st.error("未找到该代码的实时行情，请检查输入。")
            else:
                lhb_df = get_dragon_tiger(stock_code)
                flow_df = get_fund_flow(stock_code)
                hist_df = get_hist_data(stock_code)
                fin_df = get_financials(stock_code)
                north_df = get_northbound()

                stock_name = spot_info.get("名称", stock_code)
                sector = spot_info.get("板块", "未知板块")

                st.subheader(f"📌 基本概览 - {stock_name} ({stock_code})")
                cols = st.columns(4)
                cols[0].metric("最新价", spot_info.get("最新价"), f"{spot_info.get('涨跌幅')}%")
                cols[1].metric("市盈率", spot_info.get("市盈率"))
                cols[2].metric("总市值", spot_info.get("总市值") )
                cols[3].metric("换手率", spot_info.get("换手率") )

                gemini_prompt = build_gemini_prompt(stock_name, sector, spot_info, hist_df)
                gemini_view = call_gemini(gemini_key, gemini_prompt)

                deepseek_prompt = build_deepseek_prompt(gemini_view, spot_info, flow_df, lhb_df, fin_df, north_df)
                deepseek_view = call_deepseek(deepseek_base, deepseek_key, deepseek_prompt)

                chatgpt_prompt = build_chatgpt_prompt(gemini_view, deepseek_view)
                decision = call_chatgpt(openai_key, chatgpt_prompt)

                st.markdown("---")
                st.subheader("⚔️ 三位一体战场")
                battle_cols = st.columns(3)
                battle_cols[0].markdown("### 🚀 Gemini 看多叙事")
                battle_cols[0].write(gemini_view)

                battle_cols[1].markdown("### 🛡️ Deepseek 严厉审计")
                battle_cols[1].write(deepseek_view)

                battle_cols[2].markdown("### ⚖️ ChatGPT 最终判定")
                battle_cols[2].write(decision)

                st.markdown("---")
                st.subheader("📈 价格与资金可视化")
                chart_cols = st.columns(2)
                if not hist_df.empty:
                    chart_cols[0].line_chart(hist_df.set_index("日期")["收盘"], height=260)
                else:
                    chart_cols[0].info("暂无历史K线数据。")

                if not flow_df.empty:
                    flow_chart = flow_df.copy()
                    flow_chart["日期"] = pd.to_datetime(flow_chart["日期"])
                    chart_cols[1].line_chart(
                        flow_chart.set_index("日期")["主力净流入"],
                        height=260,
                    )
                else:
                    chart_cols[1].info("暂无资金流向数据。")

                with st.expander("数据明细：龙虎榜 & 财务摘要"):
                    st.markdown("#### 龙虎榜记录")
                    st.dataframe(lhb_df.head(50))
                    st.markdown("#### 财务指标摘要")
                    st.dataframe(fin_df)

elif mode == "板块轮动":
    if st.button("启动板块轮动分析", type="primary"):
        with st.spinner("获取板块数据与调度模型..."):
            sector_df = get_sector_perf()
            flow_df = get_sector_fund_flow()
            if sector_df.empty or flow_df.empty:
                st.error("无法获取板块数据，请稍后重试。")
            else:
                deepseek_prompt, gemini_prompt, chatgpt_prompt = build_sector_prompts(sector_df, flow_df)

                deepseek_view = call_deepseek(deepseek_base, deepseek_key, deepseek_prompt)
                gemini_view = call_gemini(gemini_key, gemini_prompt)
                decision = call_chatgpt(openai_key, chatgpt_prompt)

                st.subheader("🏁 板块轮动洞察")
                battle_cols = st.columns(3)
                battle_cols[0].markdown("### 🛡️ Deepseek 技术评估")
                battle_cols[0].write(deepseek_view)

                battle_cols[1].markdown("### 🚀 Gemini 宏观叙事")
                battle_cols[1].write(gemini_view)

                battle_cols[2].markdown("### ⚖️ ChatGPT 次日剧本")
                battle_cols[2].write(decision)

                with st.expander("板块排行与资金流向"):
                    st.markdown("#### 行业涨幅榜")
                    st.dataframe(sector_df)
                    st.markdown("#### 资金流向榜")
                    st.dataframe(flow_df)

else:
    st.info("请选择分析模式并输入必要信息。")

