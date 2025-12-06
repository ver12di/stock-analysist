# A-Share Trinity Command Center (A股三位一体指挥舱)

Streamlit app that orchestrates Gemini, Deepseek, and ChatGPT to perform Construct → Destroy → Rebuild analysis on Chinese A-share stocks and sector rotation.

## Running locally
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the app:
   ```bash
   streamlit run app.py
   ```

## Configuring secrets (Streamlit Cloud)
Create a **Secrets** entry in Streamlit Cloud with the following TOML structure:

```toml
openai_api_key = "sk-xxx"          # For ChatGPT ratings
deepseek_api_key = "sk-xxx"        # For Deepseek audit (OpenAI-compatible)
deepseek_base_url = "https://api.deepseek.com"  # Optional override
gemini_api_key = "your-gemini-key" # For Gemini optimism
```

The app will read from `st.secrets` first. If keys are absent (local testing), input them through the sidebar.

## Data sources
The app wraps Akshare interfaces with protective error handling:
- `stock_zh_a_spot_em` for real-time quotes and valuation
- `stock_individual_fund_flow` for main fund flow (plotted as a line chart)
- `stock_lhb_detail_em` for Dragon Tiger List appearances
- `stock_financial_abstract` for key financial metrics
- `stock_board_industry_name_em` for sector performance used in rotation mode
