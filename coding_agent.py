import json
from subprocess import run as run_subprocess

from litellm import completion

_MODEL = "ollama_chat/qwen3.6"
_API_BASE = "http://localhost:11434"
_MAX_TOOL_CALLS = 20


def read_file(filename: str) -> str:
    result = run_subprocess(["cat", filename], capture_output=True, text=True)
    return result.stdout


def edit_file(filename: str, content: str) -> str:
    with open(filename, "w") as file:
        file.write(content)
    return f"Wrote {len(content)} characters to {filename}"


def bash(command: str) -> str:
    result = run_subprocess(command.split(), capture_output=True, text=True)
    return result.stdout


def code_search(params: str) -> str:
    """Using ripgrep"""
    result = run_subprocess(["rg", params], capture_output=True, text=True)
    return result.stdout


tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a file. "
                "Returns the file contents as a string. "
                "Do not use this function to read a directory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The name of the file to read",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace the contents of a file with the provided content. "
                "Returns a short confirmation message."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The name of the file to edit",
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete new contents of the file",
                    },
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a shell command and return stdout. "
                "Use this for commands that inspect or operate on the local project."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_search",
            "description": (
                "Search the codebase using ripgrep and return matching lines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "params": {
                        "type": "string",
                        "description": "The ripgrep search pattern",
                    }
                },
                "required": ["params"],
            },
        },
    },
]


def assistant_message_to_dict(message):
    msg = {
        "role": "assistant",
        "content": message.content,  # can be empty if tool_calls are present
    }
    if message.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in message.tool_calls
        ]
        print(f"Anticipated tool calls: {msg['tool_calls']}")
    return msg


messages = [{"role": "system", "content": "You are a helpful coding assistant."}]

# Outer loop for the conversation
while True:
    user_input = input("You: ")
    if user_input.lower() == "/exit":
        print("Exiting.. bye-bye 👋")
        break

    messages.append({"role": "user", "content": user_input})

    # Inner agent loop: keep going until assistant gives a final text answer
    for _ in range(_MAX_TOOL_CALLS):  # safety limit to avoid infinite tool loops
        response = completion(
            model=_MODEL,
            messages=messages,
            api_base=_API_BASE,
            tools=tools,
        )

        assistant_message = response.choices[0].message
        # Always append the assistant message first, including its tool_calls if present.
        messages.append(assistant_message_to_dict(assistant_message))

        # Process tool calls if present.
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                print(
                    "Tool call:",
                    tool_call.function.name,
                    tool_call.function.arguments,
                )
                try:
                    func_kwargs = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    tool_result = f"Invalid JSON arguments: {e}"
                else:  # Executes ONLY if try block succeeded
                    if tool_call.function.name == "read_file":
                        tool_result = read_file(**func_kwargs)
                    elif tool_call.function.name == "edit_file":
                        tool_result = edit_file(**func_kwargs)
                    elif tool_call.function.name == "bash":
                        tool_result = bash(**func_kwargs)
                    elif tool_call.function.name == "code_search":
                        tool_result = code_search(**func_kwargs)
                    else:
                        tool_result = f"Unknown tool: {tool_call.function.name}"
                tool_call_result = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": tool_result,
                }
                print(f"Tool call result: {tool_call_result}")
                messages.append(tool_call_result)
            # Continue inner loop: now call the LLM again,
            # this time with the tool results in context.
            continue

        # No tool calls means this is the final assistant answer.
        print("Assistant:", assistant_message.content)
        break
    else:  # else block only if the loop completes all iterations without encountering a break statement
        print("⚠️ Assistant: I stopped because the tool loop hit the safety limit.")
