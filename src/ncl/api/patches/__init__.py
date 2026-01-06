"""Monkey patches for ag-ui-langgraph bugs.

Patches the "Message ID not found in history" bug in thread continuation.
See: https://github.com/CopilotKit/CopilotKit/issues/2402

Also adds Langfuse session tracking for user feedback integration.
"""

import logging
from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from ...observability import get_langfuse_handler, set_session_id

logger = logging.getLogger(__name__)


def apply_agui_thread_patch() -> None:
    """Patch LangGraphAgent to fix thread continuation bug.

    Must be called BEFORE creating LangGraphAGUIAgent instances.

    The bug occurs when continuing a conversation - the library tries to
    "time travel" back to find the user's last message by ID, but frontend
    message IDs don't match server-stored message IDs, causing:
    ValueError: Message ID not found in history

    This patch skips the problematic regeneration logic and continues
    with the merged state instead.
    """
    from ag_ui.core import CustomEvent, EventType, RunFinishedEvent, RunStartedEvent
    from ag_ui_langgraph.agent import LangGraphAgent, dump_json_safe
    from ag_ui_langgraph.types import LangGraphEventTypes
    from ag_ui_langgraph.utils import agui_messages_to_langchain, get_stream_payload_input
    from langgraph.types import Command

    async def patched_prepare_stream(
        self: LangGraphAgent, input: Any, agent_state: Any, config: RunnableConfig
    ) -> dict:
        """Patched prepare_stream that skips problematic regeneration logic."""
        state_input = input.state or {}
        messages = input.messages or []
        forwarded_props = input.forwarded_props or {}
        thread_id = input.thread_id

        # Set up Langfuse tracing with session tracking for user feedback integration
        # Uses thread_id as session_id so all traces in a conversation are grouped
        # 1. Set context variable for LiteLLM calls (embeddings, completions)
        # 2. Add LangChain callback handler with session_id metadata for LangGraph traces
        if thread_id:
            set_session_id(thread_id)
            logger.debug("Session ID set for LiteLLM calls: %s", thread_id)

        langfuse_handler = get_langfuse_handler()
        if langfuse_handler:
            # Ensure callbacks list exists
            if "callbacks" not in config:
                config["callbacks"] = []
            # Add handler if not already present
            if langfuse_handler not in config["callbacks"]:
                config["callbacks"].append(langfuse_handler)
            # Set session_id via metadata - must use camelCase "langfuseSessionId"
            if thread_id:
                if "metadata" not in config:
                    config["metadata"] = {}
                config["metadata"]["langfuseSessionId"] = thread_id
                logger.debug("Langfuse callback configured: langfuseSessionId=%s", thread_id)

        state_input["messages"] = agent_state.values.get("messages", [])
        self.active_run["current_graph_state"] = agent_state.values.copy()
        langchain_messages = agui_messages_to_langchain(messages)
        state = self.langgraph_default_merge_state(state_input, langchain_messages, input)
        self.active_run["current_graph_state"].update(state)
        config["configurable"]["thread_id"] = thread_id

        interrupts = (
            agent_state.tasks[0].interrupts
            if agent_state.tasks and len(agent_state.tasks) > 0
            else []
        )
        has_active_interrupts = len(interrupts) > 0
        resume_input = forwarded_props.get("command", {}).get("resume", None)

        self.active_run["schema_keys"] = self.get_schema_keys(config)

        # PATCH: Skip the regeneration logic that causes "Message ID not found"
        # The original code checks if agent has more messages than request
        # and tries to time-travel, but message IDs don't match.
        # Instead, we just continue with the merged state.
        non_system_messages = [
            msg for msg in langchain_messages if not isinstance(msg, SystemMessage)
        ]
        stored_message_count = len(agent_state.values.get("messages", []))
        incoming_message_count = len(non_system_messages)

        if stored_message_count > incoming_message_count:
            logger.debug(
                "Thread continuation: stored=%d, incoming=%d messages - using merged state",
                stored_message_count,
                incoming_message_count,
            )
            # Don't call prepare_regenerate_stream - just continue with merged state

        # Handle interrupts (unchanged from original)
        events_to_dispatch: list = []
        if has_active_interrupts and not resume_input:
            events_to_dispatch.append(
                RunStartedEvent(
                    type=EventType.RUN_STARTED,
                    thread_id=thread_id,
                    run_id=self.active_run["id"],
                )
            )
            for interrupt in interrupts:
                events_to_dispatch.append(
                    CustomEvent(
                        type=EventType.CUSTOM,
                        name=LangGraphEventTypes.OnInterrupt.value,
                        value=dump_json_safe(interrupt.value),
                        raw_event=interrupt,
                    )
                )
            events_to_dispatch.append(
                RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=thread_id,
                    run_id=self.active_run["id"],
                )
            )
            return {
                "stream": None,
                "state": None,
                "config": None,
                "events_to_dispatch": events_to_dispatch,
            }

        if self.active_run["mode"] == "continue":
            await self.graph.aupdate_state(
                config, state, as_node=self.active_run.get("node_name")
            )

        if resume_input:
            stream_input = Command(resume=resume_input)
        else:
            payload_input = get_stream_payload_input(
                mode=self.active_run["mode"],
                state=state,
                schema_keys=self.active_run["schema_keys"],
            )
            stream_input = {**forwarded_props, **payload_input} if payload_input else None

        subgraphs_stream_enabled = (
            input.forwarded_props.get("stream_subgraphs") if input.forwarded_props else False
        )

        kwargs = self.get_stream_kwargs(
            input=stream_input,
            config=config,
            subgraphs=bool(subgraphs_stream_enabled),
            version="v2",
        )

        stream = self.graph.astream_events(**kwargs)

        return {
            "stream": stream,
            "state": state,
            "config": config,
        }

    LangGraphAgent.prepare_stream = patched_prepare_stream
    logger.info("Applied ag-ui-langgraph thread continuation patch")
