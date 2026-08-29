import json
import os
import uuid
from types import SimpleNamespace
from typing import List, Dict, Any, Optional
from clients import groq_client, DEFAULT_MODEL
from groq import BadRequestError
import ast
import re
import hal9 as h9

TOOL_USE_FAILED_RETRIES = 2

def _tool_use_failed_generation(error: Exception) -> Optional[str]:
    """Extract Groq's failed_generation text from a tool_use_failed 400."""
    body = getattr(error, "body", None)
    if body is None:
        response = getattr(error, "response", None)
        if response is not None:
            try:
                body = response.json()
            except Exception:
                body = None
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            body = None
    if not isinstance(body, dict):
        return None
    err = body.get("error", body)
    if not isinstance(err, dict):
        return None
    message = str(err.get("message") or "")
    if err.get("code") != "tool_use_failed" and "Failed to call a function" not in message:
        return None
    text = err.get("failed_generation")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None

def _has_tool_named(tools: Optional[List], name: str) -> bool:
    if not tools:
        return False
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name") == name:
            return True
    return False

def _looks_like_malformed_tool_call(text: str) -> bool:
    lowered = text.lower()
    return (
        "<tool_call" in lowered
        or "<function" in lowered
        or '"name":' in lowered
        or text.lstrip().startswith("{")
        or text.lstrip().startswith("[")
    )

def _parse_qwen_tool_call(text: str) -> Optional[tuple[str, dict]]:
    """Parse Qwen XML-style tool calls that Groq rejects as tool_use_failed."""
    if not text:
        return None
    func = re.search(r"<function=([A-Za-z0-9_]+)>", text)
    if not func:
        return None
    name = func.group(1)
    args: dict[str, str] = {}
    for match in re.finditer(
        r"<parameter=([A-Za-z0-9_]+)>\s*(.*?)\s*</parameter>",
        text,
        flags=re.DOTALL,
    ):
        args[match.group(1)] = match.group(2).strip()
    return name, args

def _synthetic_tool_call_response(name: str, arguments: dict) -> SimpleNamespace:
    """Build a completion-shaped object so existing tool-call handlers still work."""
    tool_call = SimpleNamespace(
        id=f"call_fallback_{uuid.uuid4().hex[:12]}",
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        ),
    )
    message = SimpleNamespace(role="assistant", content=None, tool_calls=[tool_call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])

def generate_response(
    messages: List[Dict[str, Any]],
    tools: Optional[List] = None,
    tool_choice: Optional[str] = None,
    seed: Optional[int] = None,
    model: str = DEFAULT_MODEL,
    reasoning_effort: Optional[str] = None,
    response_format: Optional[Dict[str, Any]] = None,
    stream: bool = False,
    temperature: Optional[float] = None,
) -> Any:
    """
    Generates a Groq chat completion using qwen/qwen3.6-27b by default.

    Groq returns HTTP 400 tool_use_failed when tool_choice is required but the
    model replies in plain text instead of a function call. Retry, then salvage
    that text as final_response when that tool is available.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if seed is not None:
        payload["seed"] = seed
    if response_format is not None:
        payload["response_format"] = response_format
    if stream:
        payload["stream"] = True
    if temperature is not None:
        payload["temperature"] = temperature

    uses_tools_or_json = tools is not None or response_format is not None
    if uses_tools_or_json:
        payload["reasoning_format"] = "hidden"
        payload["reasoning_effort"] = reasoning_effort if reasoning_effort is not None else "none"
    elif reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
        payload["reasoning_format"] = "hidden"

    attempts = 1 + (TOOL_USE_FAILED_RETRIES if tools and tool_choice == "required" and not stream else 0)
    last_error = None
    last_failed_generation = None

    for attempt in range(attempts):
        call_payload = dict(payload)
        if attempt > 0:
            call_payload["messages"] = list(messages) + [{
                "role": "system",
                "content": (
                    "You must call one of the provided functions. "
                    "For greetings, small talk, or simple questions, call final_response "
                    "with your reply. Do not answer in plain text."
                ),
            }]
            h9.event("Tool call retry", f"attempt {attempt + 1}/{attempts}")
        try:
            return groq_client.chat.completions.create(**call_payload)
        except BadRequestError as error:
            failed_generation = _tool_use_failed_generation(error)
            if failed_generation is None:
                raise
            last_error = error
            last_failed_generation = failed_generation
            continue

    if last_failed_generation:
        parsed = _parse_qwen_tool_call(last_failed_generation)
        if parsed and _has_tool_named(tools, parsed[0]):
            name, arguments = parsed
            h9.event("Tool call XML fallback", name)
            return _synthetic_tool_call_response(name, arguments)
        if (
            not _looks_like_malformed_tool_call(last_failed_generation)
            and _has_tool_named(tools, "final_response")
        ):
            h9.event("Tool call fallback", last_failed_generation)
            return _synthetic_tool_call_response(
                "final_response",
                {"final_message": last_failed_generation},
            )

    raise last_error

def load_messages(file_path="./.storage/.messages.json") -> List[Dict[str, Any]]:
    """
    Loads messages from a JSON file located in the './.storage' directory.

    Returns:
        List[Dict[str, Any]]: A list of messages if the file exists and is valid.
    """
    if not os.path.exists(file_path):
        return []
    else :
        with open(file_path, "r", encoding="utf-8") as file:
            messages = json.load(file)

        return messages

def save_messages(messages: List[Dict[str, Any]], file_path="./.storage/.messages.json") -> None:
    """
    Saves messages to a JSON file located in the './.storage' directory.

    Args:
        messages (List[Dict[str, Any]]): A list of messages to be saved.
    """

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(messages, file, ensure_ascii=False, indent=4)

def insert_message(messages , role, content, tool_call_id=None):
    if tool_call_id:
        return None
    else:
        messages.append({"role": role, "content": content})
    return messages

def execute_function(model_response, functions, debug_mode=True):
    if debug_mode:
        h9.event("Executing Tool", model_response.choices[0].message.tool_calls[0].function.name)
    # Extract the message from the response.
    try:
        response_message = model_response.choices[0].message
    except (IndexError, AttributeError) as e:
        print(f"Error extracting message from model response: {e}")
        return

    # Access the tool calls (if any) from the message.
    tool_calls = getattr(response_message, 'tool_calls', None)

    if not tool_calls:
        print("No tool calls found.")
        return

    # Iterate over the tool calls and extract relevant information.
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        raw_args = getattr(tool_call.function, "arguments", None)
        try:
            arguments = json.loads(raw_args)
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be an object")
        except (TypeError, json.JSONDecodeError, ValueError):
            try:
                arguments = ast.literal_eval(raw_args)
            except Exception as e:
                print(f"Error parsing arguments: {e}")
                continue
        # Convert arguments into a string format for logging or execution.
        args_str = ', '.join(f"{k}={repr(v)}" for k, v in arguments.items())

        # Add all the functions into the exec context
        context = {}
        for func in functions:
            context[func.__name__] = func

        # Prepare the code string to execute
        code_to_exec = f"result = {function_name}({args_str})"

        # Execute the code with exec(), but ensure proper error handling.
        try:
            exec(code_to_exec, context)
            return context['result']
        except Exception as e:
            print(f"Error executing function '{function_name}': {e}")
            raise

def insert_tool_message(messages, model_response, tool_result, debug_mode=True):
    tool_calls = model_response.choices[0].message.tool_calls

    if tool_calls:
      for tool_call in tool_calls:
        messages.append({
          "role": "assistant",
          "tool_calls": [{
            "id": tool_call.id,
            "type": "function",
            "function": {
              "arguments": tool_call.function.arguments,
              "name": tool_call.function.name,
            },
          }]
        })
        function_args = json.loads(tool_call.function.arguments, strict=False)

        tool_content = json.dumps({**function_args, "response": str(tool_result)})

        messages.append({
            "role": "tool",
            "content": tool_content,
            "tool_call_id": tool_call.id
        })

        if debug_mode:
            h9.event("Tool Result", tool_content)
