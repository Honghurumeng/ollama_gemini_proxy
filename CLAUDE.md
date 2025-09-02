# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个 Ollama 到 Gemini API 代理服务器，使用 FastAPI 构建。它接收 Ollama 兼容的请求格式，并将它们转发给 Google Gemini API，支持流式响应和工具调用。

## 核心架构

### 主要组件

- **app/main.py**: FastAPI 应用程序入口点，配置 CORS、路由和请求日志中间件
- **app/config.py**: 配置管理，处理环境变量和模型注册表，支持运行时配置解析
- **app/routers.py**: API 路由实现，包含 Ollama 兼容的端点实现
- **app/gemini_client.py**: Gemini API 客户端，处理与 Google Gemini 的 HTTP 通信
- **app/models.py**: Pydantic 数据模型定义，涵盖 Ollama 和 Gemini 格式
- **app/logger.py**: 日志配置和工具函数，支持 JSON 和标准格式输出

### API 端点

- `POST /api/chat`: Ollama 兼容的聊天端点，支持流式响应和工具调用
- `POST /api/generate`: Ollama 兼容的文本生成端点
- `GET /api/tags`: 模型列表端点，从环境变量模型注册表获取
- `GET /api/version`: 版本信息端点
- `GET/POST /api/show`: 模型详情端点

### 配置系统

项目使用分层配置系统：
1. 请求头覆盖 (`X-Gemini-Base-Url`, `X-Gemini-Api-Version`, `X-Gemini-Api-Key`)
2. 按模型的环境变量配置 (`MODEL_<ALIAS>_*`)
3. 全局环境变量 (`GEMINI_BASE_URL`, `GEMINI_API_KEY` 等)
4. 默认值

模型别名支持通过 `MODELS` 环境变量声明，每个别名可以有独立的配置。

## 常用开发命令

### 环境设置
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 运行服务
```bash
# 使用 uvicorn (推荐用于开发)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 或使用内置运行函数
python -c "from app.main import run; run()"
```

### 配置文件
复制 `.env.example` 到 `.env` 并配置必要的环境变量，特别是：
- `GEMINI_API_KEY`: Gemini API 密钥
- `MODELS`: 模型别名列表
- 每个模型的 `MODEL_<ALIAS>_ID` 等配置

### 测试端点
```bash
# 测试聊天端点
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "gemini25pro", "messages": [{"role": "user", "content": "Hello"}]}'

# 测试模型列表
curl http://localhost:8000/api/tags
```

## 重要实现细节

### 消息格式转换
- Ollama 的 `user`/`system` 角色映射到 Gemini 的 `user` 角色
- Ollama 的 `assistant` 角色映射到 Gemini 的 `model` 角色
- 工具输出 (`tool` 角色) 转换为 Gemini 的 `functionResponse`

### 流式响应
支持两种流式格式：
- `ndjson`: 每行一个 JSON 对象 (默认，适合大多数 Ollama 客户端)
- `sse`: Server-Sent Events 格式

### 安全性
- API 密钥在日志中自动脱敏
- 支持请求体截断以控制日志大小
- CORS 配置允许跨域访问

### 错误处理
- Gemini API 错误会保持原始 HTTP 状态码转发
- 流式响应中的错误会包装为 Ollama 兼容的错误格式

## 依赖项

主要依赖：
- **fastapi**: Web 框架
- **uvicorn**: ASGI 服务器
- **httpx**: HTTP 客户端，用于调用 Gemini API
- **pydantic**: 数据验证和序列化
- **python-dotenv**: 环境变量管理