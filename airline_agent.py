from openai import OpenAI
from tau2.domains.airline.environment import get_environment, get_tasks
from tau2.data_model.tasks import Task
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.evaluator.evaluator import evaluate_simulation, EvaluationType
from tau2.utils.utils import get_now
import json
import uuid

# Use pure OpenAI message format with error handling
TaskOutput = dict  # {"messages": list[dict], "errors": any}

AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()


def criteria(task_input: Task, task_output: TaskOutput) -> float:
    """Evaluate the agent's performance using the existing evaluation system."""
    # Convert OpenAI messages to tau2 Message objects
    from tau2.data_model.message import Message, AssistantMessage, UserMessage, ToolMessage, SystemMessage, ToolCall

    messages = task_output.get("messages", [])
    errors = task_output.get("errors", None)

    tau2_messages = []
    for msg in messages:
        if msg["role"] == "system":
            tau2_messages.append(SystemMessage(role="system", content=msg["content"]))
        elif msg["role"] == "user":
            tau2_messages.append(UserMessage(role="user", content=msg["content"]))
        elif msg["role"] == "assistant":
            tool_calls = None
            if "tool_calls" in msg and msg["tool_calls"]:
                tool_calls = [
                    ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"]),
                        requestor="assistant",
                    )
                    for tc in msg["tool_calls"]
                ]
            tau2_messages.append(AssistantMessage(role="assistant", content=msg.get("content"), tool_calls=tool_calls))
        elif msg["role"] == "tool":
            tau2_messages.append(
                ToolMessage(
                    role="tool", content=msg["content"], tool_call_id=msg["tool_call_id"], requestor="assistant"
                )
            )

    # Determine termination reason based on errors or message content
    termination_reason = TerminationReason.AGENT_STOP
    if errors:
        termination_reason = TerminationReason.TOO_MANY_ERRORS

    # Create a mock SimulationRun for evaluation
    simulation = SimulationRun(
        id=str(uuid.uuid4()),
        task_id=task_input.id,
        start_time=get_now(),
        end_time=get_now(),
        duration=0.0,
        termination_reason=termination_reason,
        messages=tau2_messages,
    )

    # Use the existing evaluation system
    reward_info = evaluate_simulation(
        simulation=simulation, task=task_input, evaluation_type=EvaluationType.ALL, solo_mode=False, domain="airline"
    )

    print("Evaluation info:", reward_info)

    return reward_info.reward


def agent(task_input: Task, openai: OpenAI, model: str, max_turns: int = 20) -> TaskOutput:
    """Minimal OpenAI SDK-based agent implementation."""
    messages: list[dict] = []
    try:
        # Get environment and tools
        env = get_environment()
        tools = env.get_tools()
        policy = env.get_policy()

        # Create system prompt
        system_prompt = f"""<instructions>
{AGENT_INSTRUCTION}
</instructions>
<policy>
{policy}
</policy>"""

        # Convert tools to OpenAI format
        openai_tools = [tool.openai_schema for tool in tools]

        # Initialize conversation with pure OpenAI format
        messages = [{"role": "system", "content": system_prompt}]

        # Initialize with user scenario if available
        if task_input.user_scenario and task_input.user_scenario.instructions:
            user_msg = f"Hello, {task_input.user_scenario.instructions.reason_for_call}"
            messages.append({"role": "user", "content": user_msg})
        else:
            # Fallback user message
            messages.append({"role": "user", "content": "Hello, I need help with my airline reservation."})

        # Conversation loop
        for _ in range(max_turns):
            # Get agent response
            response = openai.chat.completions.create(
                model=model, messages=messages, tools=openai_tools, temperature=0.0
            )

            choice = response.choices[0]
            assistant_msg = choice.message

            # Handle tool calls
            if assistant_msg.tool_calls:
                # Add assistant message with tool calls
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_msg.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                            }
                            for tc in assistant_msg.tool_calls
                        ],
                    }
                )

                # Execute tool calls
                for tc in assistant_msg.tool_calls:
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments)

                    # Find and execute the tool
                    tool_result = "Tool execution not implemented"
                    for tool in tools:
                        if tool.name == tool_name:
                            try:
                                tool_result = str(tool(**tool_args))
                            except Exception as e:
                                tool_result = f"Error: {str(e)}"
                            break

                    # Add tool result to conversation
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_result})

            else:
                # Regular text response
                messages.append({"role": "assistant", "content": assistant_msg.content})
                # End conversation after text response
                break

        return {"messages": messages, "errors": None}

    except Exception as e:
        # Return error information
        return {
            "messages": messages,
            "errors": str(e),
        }


if __name__ == "__main__":
    # Test the implementation without OpenAI API calls
    env = get_environment()
    tasks = get_tasks()

    print(f"Found {len(tasks)} tasks")
    print(f"Environment has {len(env.get_tools())} tools")
    print("\nFirst task:")
    print(f"ID: {tasks[0].id}")
    print(f"Purpose: {tasks[0].description.purpose}")

    if tasks[0].user_scenario and tasks[0].user_scenario.instructions:
        print(f"User scenario: {tasks[0].user_scenario.instructions.reason_for_call}")

    print(f"Evaluation criteria actions: {len(tasks[0].evaluation_criteria.actions or [])}")

    # Uncomment to test with actual OpenAI API
    openai = OpenAI()
    for task in tasks[:1]:  # Test with first task only
        print(f"\nTesting task {task.id}")
        result = agent(task, openai, "gpt-4.1-mini")
        print(f"Agent result:", result)
        reward = criteria(task, result)
        print(f"Reward: {reward}")
