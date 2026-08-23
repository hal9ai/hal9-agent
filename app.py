from utils import generate_response, load_messages, insert_message, execute_function, save_messages, insert_tool_message, is_url, is_url_list, process_url
from tools.calculator import solve_math_problem_description, solve_math_problem
from tools.generic import answer_generic_question_description, answer_generic_question
from tools.csv_agent import analyze_csv_description, analyze_csv
from tools.image_agent import images_management_system, images_management_system_description
from tools.hal9 import answer_hal9_questions_description, answer_hal9_questions
from tools.text_agent import analyze_text_file_description, analyze_text_file
from tools.streamlit import streamlit_generator, streamlit_generator_description
from tools.shiny import shiny_generator, shiny_generator_description
from tools.fastapi import fastapi_generator, fastapi_generator_description
from tools.other_tools import final_response_description, final_response
from tools.python_execution import python_execution_description ,python_execution
from tools.website import website_generator_description, website_generator
from tools.pdf_to_csv import generate_csv_based_pdf_description, generate_csv_based_pdf
from tools.claude_code import claude_code_github_description, claude_code_github
import hal9 as h9
import os

# load messages
messages = load_messages()

# load tools
tools_descriptions = [generate_csv_based_pdf_description, python_execution_description, final_response_description, solve_math_problem_description, answer_generic_question_description, analyze_csv_description, images_management_system_description, answer_hal9_questions_description, analyze_text_file_description, fastapi_generator_description, streamlit_generator_description, shiny_generator_description, website_generator_description, claude_code_github_description]
tools_functions = [generate_csv_based_pdf, python_execution, final_response, solve_math_problem, answer_generic_question, analyze_csv, images_management_system, answer_hal9_questions, analyze_text_file, fastapi_generator, streamlit_generator, shiny_generator, website_generator, claude_code_github]

SYSTEM_PROMPT = """You are Hal9, a helpful and highly capable AI assistant. Your job is to help the user with their request.

Tool routing rules:
1. For greetings, small talk, simple chat, or anything that does not need a specialized capability, call final_response immediately with a friendly, direct reply. Do not use other tools.
2. Only use specialized tools (files, code, math, images, apps, etc.) when the request clearly needs them.
3. Available files are optional context. An empty file list is normal and is not an error — never refuse a request just because no files are present.
4. Never mention tools or internal processes to the user.
5. When a specialized tool is needed: pick the best one, use its result, then call final_response. If a tool fails, do not retry the same tool; explain the issue and suggest an alternative.
6. Always end by calling final_response with the user-facing answer."""

# Always keep the system prompt current so routing fixes apply to existing chats.
if messages and messages[0].get("role") == "system":
    messages[0]["content"] = SYSTEM_PROMPT
else:
    messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

user_input = input()

if is_url(user_input) or is_url_list(user_input):
    if is_url_list(user_input):
        for url in user_input.split(","):
            url = url.strip()
            messages = process_url(url, messages)
    else:
        messages = process_url(user_input.strip(), messages)
else:
    h9.event("User Prompt", f"{user_input}")
    user_input = user_input.replace("\f", "\n")
    available_files = os.listdir("./.storage/")
    filtered_available_files = [f for f in available_files if f != ".events" and not f.startswith(".messages")]
    # Keep the user message natural so simple chat is not biased toward tools/files.
    user_message = user_input
    if filtered_available_files:
        user_message += f"\n\n[Optional context — available files, use only if relevant: {filtered_available_files}]"
    messages = insert_message(messages, "user", user_message)
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
        print("Unable to generate a satisfactory response on time")

save_messages(messages, file_path="./.storage/.messages.json")