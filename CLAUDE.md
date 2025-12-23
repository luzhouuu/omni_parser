# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Visual browser automation agent combining OmniParser (YOLO + Florence-2) for UI detection, GPT-4o for action planning, and Playwright for browser control. The system follows a loop: Screenshot → Parse UI → Plan Action → Execute → Verify.

## Build and Development Commands

```bash
# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .  # Development install
playwright install chromium

# Download model weights (required for real parser)
python scripts/download_models.py

# Run tests
pytest tests/
pytest tests/test_specific.py -v  # Single test file

# Linting
ruff check src/
black src/ --check

# Format code
black src/
```

## Running the Agent

```bash
# Basic demo with mock parser (no GPU)
python scripts/run_demo.py --mock --url "https://example.com" --goal "Find contact"

# Real OmniParser (requires GPU and weights/)
python scripts/run_demo.py --url "https://example.com" --goal "Your task"

# Simple browser test (Playwright only)
python scripts/simple_test.py

# Full pipeline test
python scripts/test_full_pipeline.py

# Auto-login script
python scripts/wanfang_login.py --username USER --password PASS
```

## Architecture

### LangGraph Workflow (`src/browser_agent/agent/graph.py`)

```
START → observe → parse → plan → act → verify → [loop or END]
```

- **observe**: Capture screenshot via Playwright
- **parse**: Detect UI elements with OmniParser (YOLO detection + Florence-2 captioning)
- **plan**: GPT-4o decides next action based on screenshot + detected elements
- **act**: Execute action via Playwright (click, type, scroll, navigate, etc.)
- **verify**: Check if action succeeded, task complete, or stuck

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `BrowserController` | `core/browser_controller.py` | Playwright wrapper for all browser operations |
| `OmniParserCPU` | `core/omni_parser.py` | YOLO + Florence-2 UI element detection |
| `ElementMapper` | `core/element_mapper.py` | Coordinate mapping between screenshot and page |
| `BrowserAgentState` | `agent/state.py` | TypedDict state passed through workflow |
| `BrowserAutomationAgent` | `agent/wrapper.py` | High-level API for executing tasks |

### State Flow

State (`BrowserAgentState` in `agent/state.py`) contains:
- `task_goal`, `start_url`, `max_steps` - Task definition
- `screenshot_base64`, `ui_elements`, `element_map` - Observation data
- `next_action`, `action_history` - Planning/execution
- `_browser_controller`, `_omni_parser` - Runtime instances (not serialized)

### Action Types

Defined in `nodes/act.py` and `tools/definitions.py`:
- `click`, `click_at`, `type`, `scroll`, `navigate`
- `click_all_matching` - Bulk click with auto-scroll and pagination
- `download`, `upload`, `wait`, `select`, `screenshot`, `done`

## Environment Configuration

Copy `.env.example` to `.env`:

```bash
LLM_TYPE=azure_openai  # or "openai"
AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
LLM_MODEL_NAME=gpt-4o

BROWSER_HEADLESS=true
SCREENSHOT_DIR=/tmp/browser_agent_screenshots
```

## Model Weights

Downloaded to `weights/` via `scripts/download_models.py`:
- `weights/icon_detect/model.pt` - YOLOv8 UI detection model
- `weights/icon_caption_florence/` - Florence-2 captioning model

## Key Patterns

### Using BrowserAutomationAgent

```python
from browser_agent import BrowserAutomationAgent

agent = BrowserAutomationAgent(use_mock_parser=False)
result = await agent.execute_task(
    task_goal="Click all download buttons",
    start_url="https://example.com",
    max_steps=30,
    headless=False,
)
```

### Direct BrowserController Usage

```python
from browser_agent.core.browser_controller import BrowserController

async with BrowserController(headless=False) as browser:
    await browser.navigate("https://example.com")
    await browser.click_at(100, 200)
    await browser.type_text("hello")
    await browser.drag(start_x, start_y, end_x, end_y)  # For sliders
    screenshot_bytes, base64 = await browser.capture_screenshot()
```

### Text Recognition Options

```python
parser.parse_screenshot_to_elements(
    image_path="screenshot.png",
    use_ocr=True,       # EasyOCR - fast, good for text
    caption_all=False,  # Florence-2 - slower, understands icons
)
```
