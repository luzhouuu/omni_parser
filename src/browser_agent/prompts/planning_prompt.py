"""Planning prompt template for the browser automation agent."""

PLANNING_PROMPT = """## Current Page State
**URL**: {current_url}
**Title**: {page_title}

## Detected UI Elements (with positions)
{element_list}

## Previous Actions
{action_history}

## Instructions
Based on the task goal, current page state, and the element list:

1. **IMPORTANT: Use the element list above** - it contains accurate text and coordinates from OCR
2. **Find target element**: Search the element list for text matching your goal (e.g., "全文下载", "下载", "检索")
3. **Use element coordinates**: When using click_at, use the EXACT coordinates from the element list
4. **Execute**: Choose ONE action to perform

### Key Rules:
- **ALWAYS** check the element list first before clicking
- If you see "全文下载" or "全文下毂" (OCR variation) in the element list, use its coordinates
- The element list format is: [elem_XXX] type at (x, y): "text"
- Use the (x, y) coordinates directly in click_at action
- Download links (全文下载) are usually on the RIGHT side of the page (x > 1000)

### Avoiding Repeated Actions:
- Check the Previous Actions list - don't repeat the same action if it didn't progress the task
- If clicking at a position didn't work, try a DIFFERENT position from the element list
- Look for alternative text matches (e.g., "下载", "下毂", "LinkOut")

**What is your next action?**"""
