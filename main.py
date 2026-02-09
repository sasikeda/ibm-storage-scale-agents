"""Main entry point for IBM Storage Scale Provisioning Agent."""

import asyncio
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from src.provisioning_agent.agent import ProvisioningAgent


def get_interrupt_value(interrupt_list):
    """Extract interrupt value from interrupt list."""
    if interrupt_list and len(interrupt_list) > 0:
        interrupt_obj = interrupt_list[0]
        return interrupt_obj.value if hasattr(interrupt_obj, "value") else {}
    return {}


def get_user_approval(tool_name, arguments):
    """Display confirmation request and get user approval."""
    print(f"\n{'=' * 70}")
    print(f"⚠️  CONFIRMATION REQUIRED: {tool_name}")
    print(f"{'=' * 70}")
    print("Arguments:")
    print(json.dumps(arguments, indent=2))
    print(f"{'=' * 70}")

    while True:
        approval_input = input("Approve? (yes/no): ").strip().lower()
        if approval_input in ["yes", "y"]:
            return True
        elif approval_input in ["no", "n"]:
            return False
        print("Please enter 'yes' or 'no'")


async def handle_interrupt(agent, event, config):
    """Handle interrupt and resume execution based on user approval."""
    interrupt_value = get_interrupt_value(event["__interrupt__"])
    tool_name = interrupt_value.get("tool_name", "unknown")
    arguments = interrupt_value.get("arguments", {})

    approved = get_user_approval(tool_name, arguments)

    if approved:
        print(f"✓ Approved. Executing {tool_name}...\n")
    else:
        print(f"✗ Cancelled. Operation {tool_name} not executed.\n")

    final_event = None
    async for resume_event in agent.agent_executor.astream(
        Command(resume={"approved": approved}), config=config, stream_mode="values"
    ):
        final_event = resume_event

    return final_event


def display_results(event):
    """Display tool results and final agent response."""
    if not event:
        return

    messages = event.get("messages", [])
    final_response_shown = False

    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__

        if msg_type == "ToolMessage":
            print("\n[Tool Result]:")
            try:
                content = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                print(json.dumps(content, indent=2))
            except (json.JSONDecodeError, TypeError):
                print(msg.content)

        elif msg_type == "AIMessage" and hasattr(msg, "content") and msg.content:
            is_last_ai = all(type(messages[j]).__name__ != "AIMessage" for j in range(i + 1, len(messages)))

            if is_last_ai and not final_response_shown:
                print(f"\nAgent: {msg.content}\n")
                final_response_shown = True


async def main():
    """Run interactive agent session with human-in-the-loop confirmation."""
    print("IBM Storage Scale Provisioning Agent")
    print("=" * 70)

    async with ProvisioningAgent() as agent:
        print("Ready. Type 'quit' to exit.\n")

        config = {"configurable": {"thread_id": "main"}, "recursion_limit": 50}

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    break

                if not user_input:
                    continue

                interrupted = False
                event = None

                async for event in agent.agent_executor.astream(
                    {"messages": [SystemMessage(content=agent.system_prompt), HumanMessage(content=user_input)]},
                    config=config,
                    stream_mode="values",
                ):
                    if "__interrupt__" in event:
                        interrupted = True
                        event = await handle_interrupt(agent, event, config)
                        display_results(event)

                if not interrupted:
                    display_results(event)

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
