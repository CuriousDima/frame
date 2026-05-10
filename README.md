# Frame Agent Harness

__Frame__ is a small agent harness for research purposes.

# Coding Agent

*Note: This section was written entirely by the agent itself, describing its own capabilities without human intervention.*

`coding_agent.py` is a local, autonomous coding assistant that connects a Large Language Model (via Ollama) to your local environment using the LiteLLM library. 

### How it works
1. **Tool Definitions**: The script defines four callable tools (`read_file`, `edit_file`, `bash`, `code_search`) using JSON schemas that the LLM understands.
2. **Execution Loop**: The agent runs a persistent `while` loop that waits for user input. When prompted, it calls the LLM (`completion` endpoint) passing the current conversation history and available tools.
3. **Agent Reasoning**: If the LLM decides an action is needed, it returns a tool call. The harness executes the corresponding Python function, captures the output (usually `stdout`), and appends the result back into the message history.
4. **Iterative Context**: The loop continues until the LLM provides a plain-text final answer (meaning the task is complete) or hits a safety limit of 20 consecutive tool calls to prevent infinite loops.

By default, the agent connects to a local Ollama instance at `http://localhost:11434` using the `ollama_chat/qwen3.6` model.
