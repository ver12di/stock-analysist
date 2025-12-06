import textwrap
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
    openai_key = load_secret(
        "openai_api_key",
        "OpenAI / ChatGPT Key",
        "用于 ChatGPT 评审 (gpt-4o 系列)",
    )
    deepseek_key = load_secret(
        "deepseek_api_key",
        "Deepseek Key",
        "用于 Deepseek 审核 (OpenAI 兼容)",
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
        "openai_key": openai_key.strip(),
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
        return GenerativeModel("gemini-1.5-flash")
    except Exception:
        return None


def fetch_spot_info(code: str) -> Optional[Dict]:
    try:
        df = ak.stock_zh_a_spot_em()
        if df is None or df.empty:
            return None
        row = df[df["代码"] == code]
        if row.empty:
            return None
        return row.iloc[0].to_dict()
    except Exception as exc:
        st.warning(f"实时行情获取失败: {exc}")
        return None


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
        df = ak.stock_lhb_detail_em(symbol=code)
        return df if df is not None else pd.DataFrame()
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


def fetch_sector_perf() -> pd.DataFrame:
    try:
        df = ak.stock_board_industry_name_em()
        return df if df is not None else pd.DataFrame()
    except Exception as exc:
        st.warning(f"行业表现获取失败: {exc}")
        return pd.DataFrame()


def gemini_bull_case(model: Optional[GenerativeModel], stock_name: str, sector: str) -> str:
    if not model:
        return "Gemini 未配置 API Key，无法生成乐观论点。"
    prompt = textwrap.dedent(
        f"""
        你是“市场情绪猎手”，关注 A 股题材炒作与政策催化。
        股票名称: {stock_name or '未知'}
        所属行业: {sector or '未知'}
        请构造一段牛市故事：
        - 热点叙事与政策想象
        - 媒体与资金可能放大的亮点
        - 情绪驱动的潜在超预期
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
                "结合以下硬数据，列出至少 3 个风险或矛盾，格式为要点：\n"
                f"{data_points}"
            ),
        },
    ]
    try:
        resp = client.chat.completions.create(model="deepseek-chat", messages=messages)
        return resp.choices[0].message.content
    except Exception as exc:
        return f"Deepseek 调用失败: {exc}"


def chatgpt_judge(client: Optional[OpenAI], hope: str, audit: str) -> str:
    if not client:
        return "ChatGPT 未配置 API Key，无法给出评级。"
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
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return resp.choices[0].message.content
    except Exception as exc:
        return f"ChatGPT 调用失败: {exc}"


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


def format_data_points(spot: Optional[Dict], flow_df: pd.DataFrame, lhb_df: pd.DataFrame, fin_df: pd.DataFrame) -> str:
    lines = []
    if spot:
        lines.append(
            f"最新价: {spot.get('最新价', 'N/A')} | 涨跌幅: {spot.get('涨跌幅', 'N/A')}%"
        )
        lines.append(
            f"市盈率: {spot.get('市盈率-动态', 'N/A')} | 总市值: {spot.get('总市值', 'N/A')}"
        )
    if not flow_df.empty:
        latest = flow_df.iloc[0]
        lines.append(
            f"近一日主力净额: {latest.get('主力净流入-净额', 'N/A')} ({latest.get('日期', '')})"
        )
    if not lhb_df.empty:
        lines.append(f"龙虎榜上榜次数: {len(lhb_df)}")
    if not fin_df.empty:
        roe = fin_df.iloc[0].get("ROE加权(%)", "N/A")
        net_profit = fin_df.iloc[0].get("净利润", "N/A")
        lines.append(f"ROE: {roe} | 净利润: {net_profit}")
    return "\n".join(lines) or "无可用数据"


def stock_verification_flow(api_keys: Dict[str, str]):
    st.header("模式一：个股验证 (Construct-Destroy-Rebuild)")
    code = st.text_input("输入股票代码 (如 000001)")
    if st.button("启动三人小组"):
        if not code:
            st.error("请先输入股票代码")
            return

        spot = fetch_spot_info(code)
        flow_df = fetch_fund_flow(code)
        lhb_df = fetch_lhb(code)
        fin_df = fetch_financials(code)

        stock_name = spot.get("名称", "未知") if spot else "未知"
        sector = spot.get("所属行业", "未知") if spot else "未知"
        data_points = format_data_points(spot, flow_df, lhb_df, fin_df)

        st.subheader(f"{stock_name} ({code}) 数据概览")
        if spot:
            st.write({
                "价格": spot.get("最新价"),
                "涨跌幅": spot.get("涨跌幅"),
                "PE(TTM)": spot.get("市盈率-动态"),
                "总市值": spot.get("总市值"),
                "行业": sector,
            })
        if not fin_df.empty:
            st.markdown("**财务摘要 (部分)**")
            st.dataframe(fin_df.head())
        if not lhb_df.empty:
            st.markdown("**龙虎榜记录 (部分)**")
            st.dataframe(lhb_df.head())

        st.markdown("**主力资金流趋势**")
        render_fund_flow_chart(flow_df)

        gemini_model = build_gemini_client(api_keys.get("gemini_key", ""))
        deepseek_client = build_openai_client(
            api_keys.get("deepseek_key", ""), api_keys.get("deepseek_base_url", "")
        )
        chatgpt_client = build_openai_client(api_keys.get("openai_key", ""))

        with st.spinner("Gemini 正在捕捉市场情绪…"):
            bull_case = gemini_bull_case(gemini_model, stock_name, sector)
        with st.spinner("Deepseek 正在无情拆解…"):
            attack = deepseek_attack(deepseek_client, bull_case, data_points)
        with st.spinner("ChatGPT 正在综合评级…"):
            verdict = chatgpt_judge(chatgpt_client, bull_case, attack)

        col1, col2, col3 = st.columns(3)
        col1.markdown("### 🚀 Gemini 乐观论点")
        col1.write(bull_case)

        col2.markdown("### 🛡️ Deepseek 审计")
        col2.write(attack)

        col3.markdown("### ⚖️ ChatGPT 评级")
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
        chatgpt_client = build_openai_client(api_keys.get("openai_key", ""))

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
        with st.spinner("ChatGPT 正在生成次日脚本…"):
            script_msg = chatgpt_judge(chatgpt_client, macro_msg, deepseek_msg)

        col1, col2, col3 = st.columns(3)
        col1.markdown("### 🚀 Gemini 宏观亮点")
        col1.write(macro_msg)

        col2.markdown("### 🛡️ Deepseek 板块要点")
        col2.write(deepseek_msg)

        col3.markdown("### ⚖️ ChatGPT 次日剧本")
        col3.write(script_msg)


def main():
    st.set_page_config(page_title="A股三位一体指挥舱", layout="wide")
    st.title("A股三位一体指挥舱 🚀🛡️⚖️")
    st.caption("Construct - Destroy - Rebuild: Gemini × Deepseek × ChatGPT")

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
