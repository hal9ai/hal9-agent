from utils import generate_response, load_messages, insert_message, execute_function, save_messages, insert_tool_message
from tools.other_tools import final_response_description, final_response
from tools.website import website_generator_description, website_generator
from tools.code import claude_code_github_description, claude_code_github
import hal9 as h9
import os

# load messages
messages = load_messages()

# load tools
tools_descriptions = [final_response_description, website_generator_description, claude_code_github_description]
tools_functions = [final_response, website_generator, claude_code_github]

SYSTEM_PROMPT = """You are Hal9, a helpful and highly capable AI assistant.

Tool routing rules:
1. For greetings, small talk, or anything that doesn't need a specialized capability, call final_response immediately with a friendly, direct reply.
2. When the user wants to build or update a website, use website_generator, then call final_response with a summary.
3. When the user wants to change code in a GitHub repository, use claude_code_github. If they name a fork and a source/upstream repo, set repo to the fork and pr_repo to the upstream so the PR is opened on the source. Then call final_response with the PR URL.
4. Never mention tools or internal processes to the user.
5. If a tool fails, do not retry it blindly — explain the issue and suggest an alternative, then call final_response.
6. Always end by calling final_response with the user-facing answer."""

# Always keep the system prompt current so routing fixes apply to existing chats.
if messages and messages[0].get("role") == "system":
    messages[0]["content"] = SYSTEM_PROMPT
else:
    messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

user_input = h9.input()
h9.event("User Prompt", f"{user_input}")
user_input = user_input.replace("\f", "\n")
messages = insert_message(messages, "user", user_input)

steps = 0
max_steps = 6
completed = False
while steps < max_steps:
    steps += 1
    response = generate_response(messages, tools_descriptions, tool_choice="required")
    response_message = response.choices[0].message
    tool_calls = getattr(response_message, "tool_calls", None)
    if not tool_calls:
        content = (getattr(response_message, "content", None) or "").strip()
        if content:
            print(content)
            messages = insert_message(messages, "assistant", content)
        completed = True
        break
    tool_result = execute_function(response, tools_functions)
    insert_tool_message(messages, response, tool_result)
    if tool_calls[0].function.name == "final_response":
        completed = True
        break
if not completed:
    print("Unable to generate a satisfactory response in time")

save_messages(messages, file_path="./.storage/.messages.json")
