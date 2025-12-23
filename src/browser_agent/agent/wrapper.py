"""High-level API wrapper for browser automation agent."""

import asyncio
from typing import Optional, List, Dict, Any, Callable, Awaitable
from dataclasses import dataclass

from browser_agent.agent.state import create_initial_state, state_summary, BrowserAgentState
from browser_agent.agent.graph import create_browser_agent_graph
from browser_agent.agent.configuration import AgentConfig, default_config
from browser_agent.utils.log_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TaskResult:
    """Result of a browser automation task.

    Attributes:
        success: Whether the task completed successfully
        message: Summary message
        steps_taken: Number of steps executed
        final_url: URL at task completion
        final_title: Page title at task completion
        action_history: List of all actions taken
        errors: List of errors encountered
        screenshots: List of screenshot paths
    """
    success: bool
    message: str
    steps_taken: int
    final_url: str
    final_title: str
    action_history: List[Dict[str, Any]]
    errors: List[str]
    screenshots: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "message": self.message,
            "steps_taken": self.steps_taken,
            "final_url": self.final_url,
            "final_title": self.final_title,
            "action_history": self.action_history,
            "errors": self.errors,
            "screenshots": self.screenshots,
        }


class BrowserAutomationAgent:
    """High-level wrapper for browser automation tasks.

    This class provides a simple interface for executing browser
    automation tasks using the OmniParser + GPT + Playwright pipeline.

    Example:
        agent = BrowserAutomationAgent()

        result = await agent.execute_task(
            task_goal="Login and download the monthly report",
            start_url="https://example.com/login",
            max_steps=30,
            success_indicators=["Report downloaded", "download complete"],
        )

        print(f"Success: {result.success}")
        print(f"Steps: {result.steps_taken}")
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        use_mock_parser: bool = False,
    ):
        """Initialize the browser automation agent.

        Args:
            config: Optional AgentConfig instance
            use_mock_parser: Use mock parser (for testing without models)
        """
        self._config = config or default_config
        self._use_mock_parser = use_mock_parser
        self._graph = None

    def _get_graph(self):
        """Get or create the LangGraph instance."""
        if self._graph is None:
            # Disable checkpointer to avoid serialization issues with browser controller
            self._graph = create_browser_agent_graph(use_checkpointer=False)
        return self._graph

    def _build_config_dict(self) -> Dict[str, Any]:
        """Build configuration dictionary for the agent."""
        config_dict = self._config.to_dict()
        config_dict["use_mock_parser"] = self._use_mock_parser
        return config_dict

    def _extract_result(self, state: BrowserAgentState) -> TaskResult:
        """Extract TaskResult from final state.

        Args:
            state: Final agent state

        Returns:
            TaskResult instance
        """
        verification = state.get("verification_result", {})
        action_history = state.get("action_history", [])
        error_history = state.get("error_history", [])

        # Determine success
        success = verification.get("success", False)
        if state.get("is_complete") and not verification:
            # Task marked complete by done tool
            last_action = action_history[-1] if action_history else {}
            success = last_action.get("action_type") == "done" and last_action.get("success", False)

        # Build message
        message = verification.get("details", "")
        if not message:
            if action_history:
                last_action = action_history[-1]
                if last_action.get("action_type") == "done":
                    message = last_action.get("message", "Task completed")
                else:
                    message = f"Completed after {len(action_history)} actions"
            else:
                message = "No actions taken"

        # Collect screenshots
        screenshots = []
        for action in action_history:
            if action.get("action_type") == "screenshot":
                result = action.get("result", "")
                if "saved to" in result.lower():
                    # Extract path from result
                    screenshots.append(result)

        # Add final screenshot if available
        if state.get("screenshot_path"):
            screenshots.append(state["screenshot_path"])

        return TaskResult(
            success=success,
            message=message,
            steps_taken=state.get("current_step", 0),
            final_url=state.get("current_url", ""),
            final_title=state.get("page_title", ""),
            action_history=action_history,
            errors=error_history,
            screenshots=screenshots,
        )

    async def execute_task(
        self,
        task_goal: str,
        start_url: str,
        max_steps: int = 50,
        success_indicators: Optional[List[str]] = None,
        headless: bool = True,
        thread_id: Optional[str] = None,
    ) -> TaskResult:
        """Execute an automation task end-to-end.

        Args:
            task_goal: Description of what to accomplish
            start_url: URL to start the task from
            max_steps: Maximum number of steps before stopping
            success_indicators: List of strings that indicate success
            headless: Run browser in headless mode
            thread_id: Optional thread ID for state checkpointing

        Returns:
            TaskResult with execution details
        """
        logger.info(f"Starting task: {task_goal[:50]}...")
        logger.info(f"Start URL: {start_url}")

        # Build configuration
        config = self._build_config_dict()
        config["headless"] = headless

        # Create initial state
        initial_state = create_initial_state(
            task_goal=task_goal,
            start_url=start_url,
            max_steps=max_steps,
            success_indicators=success_indicators or [],
            config=config,
        )

        # Run configuration with higher recursion limit
        run_config = {"recursion_limit": 100}
        if thread_id:
            run_config["configurable"] = {"thread_id": thread_id}

        # Execute graph
        graph = self._get_graph()

        try:
            final_state = await graph.ainvoke(initial_state, run_config)

            # Clean up browser
            browser = final_state.get("_browser_controller")
            if browser:
                await browser.close()

            # Extract result
            result = self._extract_result(final_state)
            logger.info(f"Task {'succeeded' if result.success else 'failed'}: {result.message}")

            return result

        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return TaskResult(
                success=False,
                message=f"Execution error: {str(e)}",
                steps_taken=0,
                final_url="",
                final_title="",
                action_history=[],
                errors=[str(e)],
                screenshots=[],
            )

    async def execute_task_with_callback(
        self,
        task_goal: str,
        start_url: str,
        on_step: Callable[[BrowserAgentState], Awaitable[None]],
        max_steps: int = 50,
        success_indicators: Optional[List[str]] = None,
        headless: bool = True,
        thread_id: Optional[str] = None,
        user_data_dir: Optional[str] = None,
    ) -> TaskResult:
        """Execute task with step-by-step callbacks for monitoring.

        Args:
            task_goal: Description of what to accomplish
            start_url: URL to start the task from
            on_step: Async callback function called after each step
            max_steps: Maximum number of steps
            success_indicators: List of strings that indicate success
            headless: Run browser in headless mode
            thread_id: Optional thread ID for checkpointing
            user_data_dir: Directory for browser data persistence (cookies, etc.)

        Returns:
            TaskResult with execution details
        """
        logger.info(f"Starting task with callbacks: {task_goal[:50]}...")

        # Build configuration
        config = self._build_config_dict()
        config["headless"] = headless
        config["user_data_dir"] = user_data_dir

        # Create initial state
        initial_state = create_initial_state(
            task_goal=task_goal,
            start_url=start_url,
            max_steps=max_steps,
            success_indicators=success_indicators or [],
            config=config,
        )

        # Run configuration (higher recursion limit for multi-step workflows)
        run_config = {"recursion_limit": 100}

        # Execute graph with streaming
        graph = self._get_graph()
        final_state = initial_state

        try:
            async for event in graph.astream(initial_state, run_config):
                # Get the latest state
                for node_name, node_output in event.items():
                    if isinstance(node_output, dict):
                        final_state = {**final_state, **node_output}

                # Call the callback
                await on_step(final_state)

            # Clean up browser
            browser = final_state.get("_browser_controller")
            if browser:
                await browser.close()

            return self._extract_result(final_state)

        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return TaskResult(
                success=False,
                message=f"Execution error: {str(e)}",
                steps_taken=final_state.get("current_step", 0),
                final_url=final_state.get("current_url", ""),
                final_title=final_state.get("page_title", ""),
                action_history=final_state.get("action_history", []),
                errors=[str(e)],
                screenshots=[],
            )

    def execute_task_sync(
        self,
        task_goal: str,
        start_url: str,
        **kwargs,
    ) -> TaskResult:
        """Synchronous wrapper for execute_task.

        Args:
            task_goal: Description of what to accomplish
            start_url: URL to start the task from
            **kwargs: Additional arguments passed to execute_task

        Returns:
            TaskResult with execution details
        """
        return asyncio.run(
            self.execute_task(task_goal, start_url, **kwargs)
        )


def create_agent(
    use_mock_parser: bool = False,
    **config_kwargs,
) -> BrowserAutomationAgent:
    """Factory function to create a configured agent.

    Args:
        use_mock_parser: Use mock parser for testing
        **config_kwargs: Configuration overrides

    Returns:
        Configured BrowserAutomationAgent instance
    """
    config = AgentConfig(**config_kwargs) if config_kwargs else None
    return BrowserAutomationAgent(config=config, use_mock_parser=use_mock_parser)
