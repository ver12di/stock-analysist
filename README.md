# A股三位一体指挥舱 (A-Share Trinity Command Center)

Streamlit 驱动的 A 股多模型分析应用，结合 Gemini、Deepseek 与 ChatGPT 完成“构建-拆解-重构”工作流，支持个股验资与板块轮动洞察。

## 快速开始
1. 安装依赖：
   ```bash
   pip install -U streamlit akshare pandas openai google-generativeai
   ```
2. 启动服务并允许外部浏览器访问（0.0.0.0 可在容器/云主机中暴露）：
   ```bash
   streamlit run app.py --server.address 0.0.0.0 --server.port 8501
   ```
3. 在浏览器中打开 `http://<服务器IP>:8501`，在侧边栏填写 OpenAI/Gemini/Deepseek 的密钥与 Base URL，选择“个股验资”或“板块轮动”即可交互。

## Cloudflare 部署与调试
使用 Cloudflare Pages + Functions 可在云端复用同一套逻辑。

1. 安装 Wrangler（Node.js 环境）：
   ```bash
   npm install -g wrangler
   ```
2. 首次登录账户：
   ```bash
   wrangler login
   ```
3. 本地预览 Pages + Functions：
   ```bash
   wrangler dev
   ```
   - `functions/index.ts` 暴露 Worker `fetch` 处理逻辑，同时 `functions/[[path]].ts` 让 Pages Functions 捕获全部路由。
   - `dist/` 作为静态构建产物目录（`wrangler.toml` 中的 `pages_build_output_dir`），可放置前端静态文件或构建结果。
4. 发布到 Pages 项目：
   ```bash
   wrangler pages deploy dist
   ```
5. 部署 Streamlit 代理（可选）：在 `functions/index.ts` 中将 `handleRequest` 修改为将请求转发到已部署的 Streamlit 服务，或增加 API 输出，部署前可再次用 `wrangler dev` 验证。

## 功能概览
- **个股验资**：自动拉取实时行情、龙虎榜、资金流向、财务摘要、北向资金与近30日行情，驱动三模型战场输出看多叙事、数据审计与最终判定。
- **板块轮动**：分析行业涨幅与资金流向，生成技术评估、宏观叙事与次日剧本。
- 关键数据使用 `@st.cache_data` 缓存，避免频繁请求；缺少密钥时会在界面提示。

> **提示**：Akshare 部分接口可能存在访问频率限制或需要代理，请根据运行环境调整。
