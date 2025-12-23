"""Verification prompt template for LLM-based verification."""

VERIFICATION_PROMPT = """## Task Goal
{task_goal}

## Last Action
**Type**: {action_type}
**Target**: {action_target}
**Result**: {action_result}

## Current Page State
**URL**: {current_url}
**Title**: {page_title}

## Success Indicators
{success_indicators}

## Questions
1. Did the last action succeed based on the current page state?
2. Are any success indicators present?
3. Is the task complete?

Please analyze and respond with:
- action_succeeded: true/false
- task_complete: true/false
- confidence: 0.0-1.0
- reasoning: brief explanation"""


COMPLETION_CHECK_PROMPT = """## Task Goal
{task_goal}

## Success Indicators
{success_indicators}

## Current State
**URL**: {current_url}
**Title**: {page_title}

## Action History (last {num_actions} actions)
{action_history}

## Question
Has the task been completed based on the goal and success indicators?

Consider:
1. Does the current page/URL indicate completion?
2. Were all required actions performed?
3. Are success indicators present?

Respond with:
- is_complete: true/false
- confidence: 0.0-1.0
- summary: what was accomplished or what remains"""
