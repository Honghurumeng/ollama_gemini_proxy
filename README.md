# Ollama -> Gemini Proxy (FastAPI)

一个FastAPI服务器，接受Ollama兼容的API请求并转发给Google Gemini，支持流式响应和工具调用。完全兼容Cline和VS Code Copilot等客户端。

## 特性

- **多客户端兼容**: 同时支持Cline和VS Code Copilot
- **双API格式**: Ollama格式 + OpenAI格式
- **流式响应**: 支持NDJSON和SSE两种流式格式
- **工具调用**: 完整的函数调用支持
- **模型管理**: 通过.env配置多个模型别名
- **动态配置**: 支持请求头覆盖配置

## API端点

### Ollama兼容接口
- `POST /api/chat`: Ollama聊天接口，支持`stream=true`和`tools`
- `POST /api/generate`: Ollama生成接口，支持`stream=true`和`tools` 
- `GET /api/tags`: 模型列表，从.env注册表读取
- `GET /api/show`: 模型详情，VS Code Copilot兼容格式
- `GET /api/version`: 版本信息（环境变量`OLLAMA_VERSION`，默认`0.11.8`）

### OpenAI兼容接口 (VS Code Copilot)
- `POST /v1/chat/completions`: OpenAI聊天接口，支持流式响应
- `GET /v1/models`: OpenAI模型列表

## 配置说明

通过环境变量和/或请求头配置Gemini连接，请求头优先级更高。

### 环境变量

**基础配置:**
- `GEMINI_BASE_URL`: Gemini API基础地址（默认`https://generativelanguage.googleapis.com`）
- `GEMINI_API_VERSION`: API版本（默认`v1beta`）
- `GEMINI_API_KEY`: 全局API密钥（可选）
- `HOST`: 服务监听地址（默认`0.0.0.0`）
- `PORT`: 服务端口（默认`11434`）

**兼容性配置:**
- `OLLAMA_VERSION`: Ollama版本号（默认`0.11.8`，用于`/api/version`）
- `OLLAMA_STREAM_FORMAT`: 流式格式（`ndjson`或`sse`，默认`ndjson`）

**日志配置:**
- `LOG_LEVEL`: 日志级别（`INFO`/`DEBUG`/`WARN`/`ERROR`）
- `LOG_JSON`: 是否输出JSON格式日志（`true`/`false`）
- `LOG_MAX_BODY`: 日志中请求/响应体最大长度

### 请求头覆盖

可通过以下请求头临时覆盖配置：
- `X-Gemini-Base-Url`
- `X-Gemini-Api-Version`
- `X-Gemini-Api-Key`

### 模型注册表配置

通过`MODELS`环境变量定义模型别名列表（逗号分隔），通过`MODEL_<ALIAS>_*`为每个别名配置具体参数。

#### 配置示例

```env
# 基础连接配置
GEMINI_BASE_URL=https://generativelanguage.googleapis.com
GEMINI_API_VERSION=v1beta
GEMINI_API_KEY=your_global_api_key

# 服务配置
HOST=0.0.0.0
PORT=11434
OLLAMA_VERSION=0.11.8
OLLAMA_STREAM_FORMAT=ndjson

# 声明两个模型别名
MODELS=gemini25pro,gemini25flash

# gemini25pro 别名配置
MODEL_GEMINI25PRO_ID=gemini-2.5-pro
MODEL_GEMINI25PRO_NAME=Gemini 2.5 Pro
MODEL_GEMINI25PRO_TAG=latest
MODEL_GEMINI25PRO_DIGEST=sha256:8648f39daa8fbf5b18c7b4e6a8fb4990c692751d49917417b8842ca5758e7ffc
MODEL_GEMINI25PRO_SIZE=815319791
MODEL_GEMINI25PRO_FORMAT=gguf
MODEL_GEMINI25PRO_FAMILY=gemma3
MODEL_GEMINI25PRO_DESCRIPTION=高质量，多模态
# 可选：该别名专属配置
MODEL_GEMINI25PRO_API_KEY=
MODEL_GEMINI25PRO_BASE_URL=
MODEL_GEMINI25PRO_API_VERSION=

# gemini25flash 别名配置
MODEL_GEMINI25FLASH_ID=gemini-2.5-flash
MODEL_GEMINI25FLASH_NAME=Gemini 2.5 Flash
MODEL_GEMINI25FLASH_TAG=latest
MODEL_GEMINI25FLASH_DESCRIPTION=更快，更便宜
```

#### 配置说明

**必填字段:**
- `MODEL_<ALIAS>_ID`: 真实的Gemini模型ID

**VS Code Copilot兼容字段（建议配置）:**
- `MODEL_<ALIAS>_NAME`: 显示名称
- `MODEL_<ALIAS>_DIGEST`: 模型摘要（需非空才能在某些IDE中显示）
- `MODEL_<ALIAS>_SIZE`: 模型大小（需>0才能显示）
- `MODEL_<ALIAS>_FORMAT`: 格式（推荐`gguf`）
- `MODEL_<ALIAS>_FAMILY`: 模型家族（推荐`gemma3`）

**可选覆盖字段:**
- `MODEL_<ALIAS>_API_KEY`: 该别名专属API密钥
- `MODEL_<ALIAS>_BASE_URL`: 该别名专属基础地址
- `MODEL_<ALIAS>_API_VERSION`: 该别名专属API版本

## 安装运行

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 配置环境变量（或使用.env文件）
export GEMINI_API_KEY=your_api_key_here

# 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 11434
# 或者
python -c "from app.main import run; run()"
```

## 使用示例

### Cline客户端

配置Cline使用本地代理：
- 模型服务器地址：`http://localhost:11434`
- 模型：`gemini25pro` 或 `gemini25flash`

### VS Code Copilot

1. 安装Ollama扩展
2. 配置Ollama服务地址：`http://localhost:11434`
3. 模型会自动从`/api/tags`获取

### API调用示例

#### `/api/generate` 接口

```bash
curl -sS http://localhost:11434/api/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemini25pro",
    "prompt": "写一首关于测试的俳句",
    "options": {"temperature": 0.7, "max_tokens": 128}
  }'
```

#### `/api/chat` 接口

```bash
curl -sS http://localhost:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemini25pro",
    "messages": [
      {"role": "system", "content": "你是一个有用的助手"},
      {"role": "user", "content": "给我3个关于FastAPI的要点"}
    ],
    "options": {"temperature": 0.4}
  }'
```

#### 流式响应示例

```bash
curl -N http://localhost:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemini25flash",
    "messages": [
      {"role": "user", "content": "逐个token流式生成一个短句"}
    ],
    "stream": true
  }'
```

#### 请求头覆盖配置

```bash
curl -sS http://localhost:11434/api/generate \
  -H 'Content-Type: application/json' \
  -H 'X-Gemini-Base-Url: https://your-proxy.example.com' \
  -H 'X-Gemini-Api-Key: your_special_key' \
  -d '{"model": "gemini-2.5-flash", "prompt": "你好"}'
```

## 工具调用示例

声明工具并发起调用：

```bash
curl -sS http://localhost:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemini25pro",
    "messages": [
      {"role": "user", "content": "旧金山的天气怎么样？"}
    ],
    "tools": [
      {
        "function_declarations": [
          {
            "name": "get_weather",
            "description": "根据城市名获取天气信息",
            "parameters": {
              "type": "OBJECT",
              "properties": {"city": {"type": "STRING"}},
              "required": ["city"]
            }
          }
        ]
      }
    ]
  }'
```

如果模型需要调用函数，响应会包含`tool_calls`：

```json
{"tool_calls": [{"type": "function", "name": "get_weather", "args": {"city": "San Francisco"}}]}
```

然后将工具结果作为`tool`消息发送回去：

```bash
curl -sS http://localhost:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemini25pro",
    "messages": [
      {"role": "user", "content": "旧金山的天气怎么样？"},
      {"role": "assistant", "content": "正在调用get_weather(...)"},
      {"role": "tool", "name": "get_weather", "content": "{\"temp\": 21, \"unit\": \"C\"}"}
    ]
  }'
```

## 技术说明

- **参数映射**: 将Ollama的`temperature`、`top_p`、`top_k`、`max_tokens`选项映射到Gemini的`generationConfig`
- **消息转换**: 将Ollama聊天消息转换为Gemini的`contents`格式，角色映射：user/system -> `user`，assistant -> `model`
- **工具输出**: 工具调用结果通过`functionResponse`映射
- **错误转发**: Gemini的错误响应会原样转发HTTP状态码
- **流式兼容**: 支持NDJSON（Cline）和SSE（其他客户端）两种流式格式
- **多客户端**: 同时兼容Ollama协议客户端和OpenAI协议客户端

## 兼容性

### 已测试客户端
- ✅ **Cline**: 完整支持，包括工具调用
- ✅ **VS Code Copilot**: 完整支持，包括模型列表和聊天
- ✅ **Ollama CLI**: 基本兼容

### 流式格式
- **NDJSON**: 每行一个JSON对象，推荐用于Cline
- **SSE**: Server-Sent Events格式，通过`OLLAMA_STREAM_FORMAT=sse`启用

## 开发计划

- 支持图片输入和多部分内容
- 增加安全设置和工具配置助手
- 支持更多OpenAI兼容端点
