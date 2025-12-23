"""LangGraph workflow definition for browser automation agent."""

from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from browser_agent.agent.state import BrowserAgentState
from browser_agent.nodes.observe import observe_node
from browser_agent.nodes.parse import parse_node
from browser_agent.nodes.plan import plan_node
from browser_agent.nodes.act import act_node
from browser_agent.nodes.verify import verify_node
from browser_agent.utils.log_utils import get_logger

logger = get_logger(__name__)


def should_continue(state: BrowserAgentState) -> Literal["observe", "end"]:
    """Determine next step in workflow.

    Args:
        state: Current agent state

    Returns:
        "observe" to continue loop, "end" to finish
    """
    # Check if complete
    if state.get("is_complete"):
        logger.info("Task marked complete, ending workflow")
        return "end"

    # Check for unrecoverable error
    error = state.get("error")
    if error and state.get("retry_count", 0) >= 3:
        logger.warning(f"Max retries reached with error: {error}")
        return "end"

    # Check max steps
    current_step = state.get("current_step", 0)
    max_steps = state.get("max_steps", 50)
    if current_step >= max_steps:
        logger.warning(f"Max steps ({max_steps}) reached")
        return "end"

    # Check for stuck state
    verification = state.get("verification_result", {})
    if verification.get("stuck_detection"):
        logger.warning("Stuck state detected, ending workflow")
        return "end"

    # Continue loop
    return "observe"


def create_browser_agent_graph(use_checkpointer: bool = True):
    """Create the LangGraph workflow for browser automation.

    The workflow follows the pattern:
    START -> observe -> parse -> plan -> act -> verify -> (loop or END)

    Args:
        use_checkpointer: Whether to enable state checkpointing

    Returns:
        Compiled LangGraph
    """
    logger.info("Creating browser agent graph...")

    # Create graph builder
    builder = StateGraph(BrowserAgentState)

    # Add nodes
    builder.add_node("observe", observe_node)
    builder.add_node("parse", parse_node)
    builder.add_node("plan", plan_node)
    builder.add_node("act", act_node)
    builder.add_node("verify", verify_node)

    # Define edges - linear flow through nodes
    builder.add_edge(START, "observe")
    builder.add_edge("observe", "parse")
    builder.add_edge("parse", "plan")
    builder.add_edge("plan", "act")
    builder.add_edge("act", "verify")

    # Conditional edge from verify - either continue loop or end
    builder.add_conditional_edges(
        "verify",
        should_continue,
        {
            "observe": "observe",
            "end": END,
        }
    )

    # Compile with optional checkpointing
    if use_checkpointer:
        checkpointer = MemorySaver()
        graph = builder.compile(checkpointer=checkpointer)
    else:
        graph = builder.compile()

    logger.info("Browser agent graph created successfully")
    return graph


# Default graph instance
graph = create_browser_agent_graph(use_checkpointer=True)


async def run_browser_agent(
    task_goal: str,
    start_url: str,
    max_steps: int = 50,
    success_indicators: list = None,
    config: dict = None,
    thread_id: str = None,
) -> dict:
    """Run the browser automation agent.

    This is a convenience function for running the agent.

    Args:
        task_goal: The goal or objective to accomplish
        start_url: Initial URL to start from
        max_steps: Maximum number of steps
        success_indicators: Optional list of completion indicators
        config: Optional runtime configuration
        thread_id: Optional thread ID for checkpointing

    Returns:
        Final state dictionary
    """
    from browser_agent.agent.state import create_initial_state

    # Create initial state
    initial_state = create_initial_state(
        task_goal=task_goal,
        start_url=start_url,
        max_steps=max_steps,
        success_indicators=success_indicators or [],
        config=config or {},
    )

    # Run configuration
    run_config = {}
    if thread_id:
        run_config["configurable"] = {"thread_id": thread_id}

    # Execute graph
    logger.info(f"Starting browser agent: {task_goal[:50]}...")

    try:
        final_state = await graph.ainvoke(initial_state, run_config)

        # Clean up browser
        browser = final_state.get("_browser_controller")
        if browser:
            await browser.close()

        logger.info("Browser agent completed")
        return final_state

    except Exception as e:
        logger.error(f"Browser agent failed: {e}")
        raise
