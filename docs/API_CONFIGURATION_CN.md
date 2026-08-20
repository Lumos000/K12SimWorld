# VisPhyWorld 模型 API 配置指南

## 1. 配置文件放在哪里

在项目根目录执行：

```bash
cd /path/to/K12SimWorld
cp .env.template .env
```

然后只编辑仓库根目录的 `.env`。`.env` 已被 `.gitignore` 忽略，不要把真实密钥写进 Python 文件、提交到 Git，或发到聊天中。

## 2. 运行哪个模型，就填写哪个 key

| `--model` 类型 | 必填变量 | Base URL | 说明 |
|---|---|---|---|
| `gpt-*` | `OPENAI_API_KEY` | 官方 OpenAI 留空 `OPENAI_BASE_URL`；代理填写其 `/v1` 地址 | 主预测和评测阶段的 GPT GT 文本生成都使用它 |
| `claude-*` | `ANTHROPIC_API_KEY` | 当前固定为 Anthropic 官方 Messages API | 没有 key 时程序会提前报错，不再生成伪兜底结果 |
| `gemini-*` | `GEMINI_API_KEY` | 默认 `https://generativelanguage.googleapis.com` | 用于主预测；视频评测可另设 `GEMINI_EVAL_*` |
| `qwen-*` | `DASHSCOPE_API_KEY` | 必须填写与 key 同区域的 `DASHSCOPE_API_BASE` | 推荐使用这两个标准变量；旧的 `QWEN_API_KEY/QWEN_API_BASE` 仍兼容 |
| `hf:*` | `HF_TOKEN` | 默认 `https://router.huggingface.co/v1` | token 需要 Inference Providers 调用权限 |

最常见的最小配置（只跑一个模型）如下。

### OpenAI

```dotenv
OPENAI_API_KEY="sk-你的密钥"
OPENAI_BASE_URL=""
```

如果使用 OpenAI 兼容代理：

```dotenv
OPENAI_API_KEY="代理分配的密钥"
OPENAI_BASE_URL="https://你的服务地址/v1"
```

### Anthropic Claude

```dotenv
ANTHROPIC_API_KEY="sk-ant-你的密钥"
```

### Google Gemini

```dotenv
GEMINI_API_KEY="你的密钥"
GEMINI_API_BASE="https://generativelanguage.googleapis.com"
GEMINI_API_MODE="google"
```

第三方 OpenAI 兼容 Gemini 服务使用：

```dotenv
GEMINI_API_KEY="第三方密钥"
GEMINI_API_BASE="https://第三方域名/v1"
GEMINI_API_MODE="openai"
GEMINI_MODEL="[L]gemini-3.1-pro-preview"
```

`[L]` 等前缀属于第三方服务的实际模型 ID，必须以其 `/v1/models`
返回值为准。OpenAI 兼容模式使用 Bearer 鉴权和非流式
`/v1/chat/completions`；Google 官方模式仍使用原生 `v1beta` API。

如果主预测走 OpenAI 兼容接口，而视频评分走 Gemini 原生接口，应分开配置。
例如 AiHubMix 的 Gemini 视频评分配置为：

```dotenv
GEMINI_EVAL_API_KEY="第三方密钥"
GEMINI_EVAL_API_BASE="https://aihubmix.com/gemini"
GEMINI_EVAL_MODEL="gemini-2.5-pro"
GEMINI_EVAL_MAX_RETRIES="3"
GEMINI_EVAL_RETRY_BACKOFF_SECONDS="5"
```

评测专用变量优先于通用的 `GEMINI_API_KEY` 和 `GEMINI_MODEL_ID`，因此不会
影响主预测模型。批量评测会缓存成功响应，发生限流后可直接重跑同一命令。

### 通义千问 / DashScope

中国北京区域：

```dotenv
DASHSCOPE_API_KEY="sk-你的密钥"
DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

新加坡区域：

```dotenv
DASHSCOPE_API_KEY="sk-你的密钥"
DASHSCOPE_API_BASE="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
```

美国弗吉尼亚区域：

```dotenv
DASHSCOPE_API_KEY="sk-你的密钥"
DASHSCOPE_API_BASE="https://dashscope-us.aliyuncs.com/compatible-mode/v1"
```

DashScope 的 key 和 URL 不能跨区域混用，否则通常返回 401。

如果使用工作空间专属域名，仍必须使用 OpenAI-compatible 路径，例如：

```dotenv
DASHSCOPE_API_BASE="https://你的WorkspaceId.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
```

不要填写同一域名下的 `/api/v1`；那是原生 DashScope API 路径，而本项目发送的是 OpenAI-compatible Chat Completions 请求。

### Hugging Face Inference Providers

```dotenv
HF_TOKEN="hf_你的令牌"
HF_API_BASE="https://router.huggingface.co/v1"
```

## 3. 检查配置与最小请求

先确认当前仓库入口和参数可用；这一步不会发起网络请求：

```bash
python run_k12simworld.py --help
python run_k12simworld.py generate --help
```

当前 CLI 不提供独立的 `--check-api` 或 `--list-models` 命令。模型名由 `--model` 传入，provider 由 `src.llm_client.LLMClient` 根据模型名选择。真正的最小连通性检查是只生成一个样本且不渲染：

```bash
python run_k12simworld.py generate \
  --benchmark /path/to/k12simbench.jsonl \
  --output-dir /tmp/k12simworld-api-smoke \
  --model qwen3-vl-plus \
  --limit 1 --jobs 1
```

该命令会产生实际 API 费用。先使用 `--jobs 1`，确认成功后再提高并行数。失败原因会写入对应题目的 `manifest.json`。

## 4. 各工作流所需配置

- `curate`、`prepare-prompts`、`partition-physics-tiers`、`simulate-domain`、`validate`、`evaluate` 和 `score-traces` 可以离线运行。
- `generate` 和 `generate-baseline` 只需要所选 `--model` 对应的 provider key。
- `run_k12_screening.py` 固定使用 DeepSeek 文本预筛和 Qwen 多模态终筛，因此需要 `.env.template` 中 screening 小节列出的两组模型、key 和 base URL。
- `--render` 另外需要 Node.js、Puppeteer、FFmpeg；Manim engine 还需要本地安装 Manim。

项目会记录请求内容和模型响应，但不会主动写入 API key。日志仍可能包含输入图像、提示词和模型输出，分享前应检查隐私内容。

## 5. 常见错误

- `401`：key 无效，或 DashScope key 与 Base URL 区域不一致。
- `403`：账号或项目没有模型权限。
- `404 model not found`：账号不能访问传入的模型 ID，或 Base URL 路径不兼容。
- `429`：余额、RPM/TPM/RPD 或并发限制；先把 `--jobs` 降为 1。
- `openai` 或 `dotenv` 库缺失：执行 `python -m pip install -r requirements-k12.txt`。
- 渲染器报告缺少 Puppeteer 或 FFmpeg：执行 README 的可选渲染安装步骤；仅做离线求解和不带 `--render` 的生成不需要这些系统工具。
