# Browser Automation Agent

基于 OmniParser + Florence-2 + GPT + Playwright 的智能浏览器自动化代理。

## 项目概述

本项目实现了一个视觉驱动的浏览器自动化系统，核心流程为：

```
截图 → OmniParser检测UI元素 → Florence-2生成描述 → GPT规划动作 → Playwright执行
```

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser Automation Agent                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
│  │ Observe │ → │  Parse  │ → │  Plan   │ → │   Act   │     │
│  │  截图   │   │ UI解析  │   │ GPT规划 │   │ 执行动作│     │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘     │
│       ↑                                          │          │
│       └──────────────────────────────────────────┘          │
│                      循环执行直到任务完成                     │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件

| 组件 | 文件 | 功能 |
|------|------|------|
| OmniParser | `core/omni_parser.py` | YOLO检测 + Florence-2描述生成 |
| BrowserController | `core/browser_controller.py` | Playwright浏览器控制 |
| ElementMapper | `core/element_mapper.py` | 坐标映射与元素管理 |
| Plan Node | `nodes/plan.py` | GPT动作规划 |
| Act Node | `nodes/act.py` | 动作执行引擎 |

## 核心功能

### 1. UI 元素检测与描述

使用 OmniParser 进行 UI 元素检测：

```python
from browser_agent.core.omni_parser import OmniParserCPU

parser = OmniParserCPU(weights_dir="weights")
elements, element_map = parser.parse_screenshot_to_elements(
    image_path="screenshot.png",
    viewport_width=1280,
    viewport_height=720,
    use_ocr=False,      # 不使用OCR
    caption_all=True,   # 使用Florence-2生成描述
)
```

**两种文本识别方式**：

| 方式 | 优点 | 缺点 |
|------|------|------|
| OCR (EasyOCR) | 快速，适合纯文本 | 无法理解图标含义 |
| Florence-2 Captioning | 能理解图标和复杂UI | 较慢 |

### 2. 批量点击功能 (click_all_matching)

**功能说明**：自动滚动页面并翻页，找到并点击所有匹配指定文本的元素。

**调用方式**：
```python
click_all_matching(
    text_pattern="下载全文",      # 要匹配的文本
    delay_between_clicks=1.0,    # 点击间隔（秒）
    scroll_and_find=True,        # 是否滚动查找更多
    max_scroll_attempts=10,      # 每页最大滚动次数
    paginate=True,               # 是否自动翻页
    next_page_pattern="下一页",   # 翻页按钮文本
    max_pages=50                 # 最大处理页数
)
```

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `text_pattern` | str | 必填 | 要匹配的元素文本 |
| `delay_between_clicks` | float | 1.0 | 每次点击之间的等待时间（秒） |
| `scroll_and_find` | bool | True | 是否滚动页面查找更多元素 |
| `max_scroll_attempts` | int | 10 | 每页最大滚动次数 |
| `paginate` | bool | False | 是否自动点击下一页继续处理 |
| `next_page_pattern` | str | "下一页" | 翻页按钮的文本模式 |
| `max_pages` | int | 50 | 最大处理页数 |

**执行流程**：

```
┌─────────────────────────────────────────────────────────────────┐
│           click_all_matching 执行流程（支持翻页）                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    外层循环：翻页                         │   │
│  │                 (pages_processed < max_pages)            │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │  1. 初始化当前页                                         │   │
│  │     ├── pages_processed += 1                            │   │
│  │     └── clicked_positions.clear() (每页重新计数)         │   │
│  │                                                         │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │              内层循环：滚动当前页                  │   │   │
│  │  │          (page_scroll_attempts < max_scroll)     │   │   │
│  │  ├─────────────────────────────────────────────────┤   │   │
│  │  │                                                 │   │   │
│  │  │  2. 查找匹配元素                                 │   │   │
│  │  │     ├── 遍历 element_map                        │   │   │
│  │  │     ├── 文本匹配 text_pattern                   │   │   │
│  │  │     └── 位置去重 (x/30, y/30)                   │   │   │
│  │  │                                                 │   │   │
│  │  │  3. 点击所有匹配元素                             │   │   │
│  │  │     ├── browser.click_at(x, y)                  │   │   │
│  │  │     ├── 记录位置到 clicked_positions            │   │   │
│  │  │     └── 等待 delay_between_clicks               │   │   │
│  │  │                                                 │   │   │
│  │  │  4. 滚动页面                                     │   │   │
│  │  │     ├── scroll("down", viewport_height-100)     │   │   │
│  │  │     └── 等待 0.5 秒                             │   │   │
│  │  │                                                 │   │   │
│  │  │  5. 重新截图解析                                 │   │   │
│  │  │     ├── capture_screenshot()                    │   │   │
│  │  │     ├── Florence-2 captioning                   │   │   │
│  │  │     └── 更新 element_map                        │   │   │
│  │  │                                                 │   │   │
│  │  │  6. 检查是否有新元素                             │   │   │
│  │  │     ├── 有新元素 → 继续内层循环                  │   │   │
│  │  │     └── 无新元素 → 退出内层循环                  │   │   │
│  │  │                                                 │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │                                                         │   │
│  │  7. 翻页处理 (如果 paginate=True)                       │   │
│  │     ├── 滚动到页面顶部                                  │   │
│  │     ├── 截图并解析查找 next_page_pattern                │   │
│  │     ├── 找到 → 点击翻页按钮，等待加载                    │   │
│  │     ├── 未找到 → 结束（已到最后一页）                    │   │
│  │     └── 重新截图解析新页面                              │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  8. 返回结果                                                    │
│     └── {                                                      │
│           "success": True/False,                               │
│           "clicked_count": 总点击数,                            │
│           "scroll_attempts": 总滚动次数,                        │
│           "pages_processed": 处理的页数                         │
│         }                                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**去重机制**：

```python
def get_position_key(x: float, y: float) -> tuple:
    """基于网格的位置去重，30像素为一个网格单元"""
    return (int(x / 30), int(y / 30))
```

这确保了即使滚动后元素位置略有变化，也不会重复点击同一个元素。
每翻到新页面时，位置集合会清空重新计数。

**使用示例**：

```python
# 示例1：只处理当前页面（滚动查找）
click_all_matching(
    text_pattern="下载全文",
    scroll_and_find=True,
    paginate=False
)

# 示例2：处理所有页面（滚动+翻页）
click_all_matching(
    text_pattern="下载全文",
    scroll_and_find=True,
    paginate=True,
    next_page_pattern="下一页",
    max_pages=100
)

# 示例3：英文网站翻页
click_all_matching(
    text_pattern="Download",
    paginate=True,
    next_page_pattern="Next"
)
```

**适用场景**：
- 批量下载多页文献的"下载全文"按钮
- 批量点击"收藏"或"关注"按钮（跨多页）
- 批量选择列表中的所有项目
- 任何需要遍历多页并点击相同按钮的操作

### 3. 可用的动作工具

| 工具 | 说明 |
|------|------|
| `click` | 点击指定元素 |
| `click_at` | 点击指定坐标 |
| `click_all_matching` | 批量点击所有匹配文本的元素 |
| `type` | 输入文本 |
| `scroll` | 滚动页面 |
| `navigate` | 导航到URL |
| `download` | 触发下载 |
| `wait` | 等待条件 |
| `done` | 标记任务完成 |

## 安装

### 1. 创建虚拟环境

```bash
cd omni_parser
python3 -m venv venv
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
playwright install chromium
```

### 3. 下载模型权重

```bash
python scripts/download_models.py
```

模型会下载到 `weights/` 目录：
- `weights/icon_detect/model.pt` - YOLO检测模型
- `weights/icon_caption_florence/` - Florence-2描述模型

### 4. 配置环境变量

复制 `.env.example` 为 `.env` 并配置：

```bash
# Azure OpenAI 配置
AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# 或 OpenAI 配置
OPENAI_API_KEY=your-api-key
```

## 使用示例

### 基本使用

```python
import asyncio
from browser_agent import BrowserAutomationAgent

async def main():
    agent = BrowserAutomationAgent(use_mock_parser=False)

    result = await agent.execute_task(
        task_goal="点击页面上所有的'下载全文'按钮",
        start_url="https://example.com/articles",
        max_steps=30,
        headless=False,
    )

    print(f"成功: {result.success}")
    print(f"点击数: {result.steps_taken}")

asyncio.run(main())
```

### 运行演示

```bash
# 使用真实解析器
python scripts/run_demo.py --url "https://lczl.med.wanfangdata.com.cn/ADR?pagesize=1" --goal "点击所有下载全文按钮"

python scripts/wanfang_login.py --username "18696799103" --password "Acmilan@2025" 

# 使用模拟解析器（测试用）
python scripts/run_demo.py --mock
```

## 目录结构

```
omni_parser/
├── src/
│   └── browser_agent/
│       ├── agent/           # Agent核心逻辑
│       │   ├── graph.py     # LangGraph工作流
│       │   ├── state.py     # 状态定义
│       │   └── wrapper.py   # 高级API封装
│       ├── core/            # 核心组件
│       │   ├── omni_parser.py      # UI解析器
│       │   ├── browser_controller.py # 浏览器控制
│       │   └── element_mapper.py    # 元素映射
│       ├── nodes/           # 工作流节点
│       │   ├── observe.py   # 观察节点（截图）
│       │   ├── parse.py     # 解析节点
│       │   ├── plan.py      # 规划节点
│       │   ├── act.py       # 执行节点
│       │   └── verify.py    # 验证节点
│       ├── tools/           # GPT工具定义
│       │   └── definitions.py
│       └── utils/           # 工具函数
├── scripts/                 # 脚本
├── weights/                 # 模型权重
└── tests/                   # 测试
```

## 技术栈

- **UI检测**: YOLOv8 (ultralytics)
- **图像描述**: Florence-2 (transformers)
- **LLM规划**: GPT-4o / Azure OpenAI
- **浏览器控制**: Playwright
- **工作流**: LangGraph
- **OCR备选**: EasyOCR
