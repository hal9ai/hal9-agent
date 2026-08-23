import json
import os
import urllib.parse
import requests
from typing import List, Dict, Any, Optional
from clients import groq_client, DEFAULT_MODEL
import pymupdf
from io import BytesIO
import pandas as pd
import ast
import re
import hal9 as h9
from replicate import Client

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

    return groq_client.chat.completions.create(**payload)

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
        try:
            arguments = ast.literal_eval(tool_call.function.arguments)
        except AttributeError as e:
            print(f"Error accessing arguments: {e}")
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

def stream_print(stream, show = True):
    content = ""
    for chunk in stream:
      if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None: 
        if show:
            print(chunk.choices[0].delta.content, end="")
        content += chunk.choices[0].delta.content
    return content

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

def is_url(prompt):
  result = urllib.parse.urlparse(prompt)
  return all([result.scheme, result.netloc])

def download_file(url):
    filename = url.split("/")[-1]
    modified_filename = f"./.storage/.{filename}"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(modified_filename, 'wb') as file:
            file.write(response.content)
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

def split_text(text, n_words=300, overlap=0):
    """
    Splits a text into chunks of `n_words` words with an overlap of `overlap` words.

    Args:
        text (str): The input text to be split.
        n_words (int): Number of words per chunk.
        overlap (int): Number of overlapping words between consecutive chunks.

    Returns:
        list: A list of text chunks.
    """
    # Validate inputs
    if overlap >= n_words:
        raise ValueError("Overlap must be smaller than the number of words per chunk.")

    # Split the text into words
    words = text.split()
    chunks = []

    # Generate the chunks
    start = 0
    while start < len(words):
        end = start + n_words
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        
        # Move the start point forward, with overlap
        start += n_words - overlap

    return chunks

def generate_text_embeddings_parquet(
    url,
    n_words=300,
    overlap=0,
    storage_path="./.storage/.text_files.parquet"
):
    # Download PDF and split pages into text chunks for later retrieval.
    resp = requests.get(url)
    doc = pymupdf.open(stream=BytesIO(resp.content))

    rows = []
    for i in range(len(doc)):
        text = doc[i].get_text()
        for chunk in split_text(text, n_words=n_words, overlap=overlap):
            rows.append({
                "text": chunk,
                "page": i + 1,
            })
    doc.close()

    df_new = pd.DataFrame(rows)
    df_new['chunk_id'] = range(len(df_new))
    df_new['filename'] = os.path.basename(url)

    os.makedirs(os.path.dirname(storage_path), exist_ok=True)

    # Load existing and append
    if os.path.exists(storage_path):
        df_old = pd.read_parquet(storage_path, engine="pyarrow")
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    # Save all
    df.to_parquet(storage_path, engine="pyarrow", index=False)

def load_json_file(json_path):
    if os.path.exists(json_path):
        with open(json_path, 'r') as file:
            return json.load(file)
    return []

def extract_code_block(code: str, language: str) -> str:
    pattern = rf"```{language}\n(.*?)```"
    match = re.search(pattern, code, re.DOTALL)
    return match.group(1) if match else ""


def is_url_list(prompt):
    urls_list = prompt.split(",")
    for url in urls_list:
        result = urllib.parse.urlparse(url.strip())
        if not all([result.scheme, result.netloc]):
            return False
    return True

def add_images_descriptions(image_path):
    description = generate_description(image_path)

    file_name = './.storage/.images_description.json'

    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            data = json.load(file)
    else:
        data = []

    new_record = {
        "image_path": image_path,
        "image_description": description
    }

    data.append(new_record)

    with open(file_name, 'w') as file:
        json.dump(data, file, indent=4)

    return description

replicate = Client(api_token=os.environ['HAL9_TOKEN'], base_url="https://api.hal9.com/proxy/server=https://api.replicate.com")

def generate_description(image_path):
    try:
        file_input = open(image_path, 'rb')
        input = {
            "image": file_input,
            "prompt": """Generate a detailed image prompt that includes all specific visual details in the image. This should include precise descriptions of colors, textures, lighting, positions of all elements, proportions, background details, 
            foreground details, and any unique stylistic choices. Ensure the description is exhaustive enough to allow an artist or AI to recreate the image accurately without visual reference."""
        }

        description = ""
        for event in replicate.stream(
            "yorickvp/llava-13b:80537f9eead1a5bfa72d5ac6ea6414379be41d4d4f6679fd776e9535d1eb58bb",
            input=input
        ):
          description+=event.data
        file_input.close()
    except Exception as e: 
        return (f"Couldn't describe that image. -> Error: {e}")
    
    return description.replace("{", "").replace("}", "")

def process_url(url, messages):
    h9.event("Uploaded File", f"{url}")
    filename = url.split("/")[-1]
    file_extension = filename.split(".")[-1] if "." in filename else "No extension"

    download_file(url)
    messages = insert_message(messages, "system", f"Consider use the file available at path: './.storage/.{filename}' for the following questions.")
    messages = insert_message(messages, "assistant", f"I'm ready to answer questions about your file: {filename}")

    if file_extension.lower() == "pdf":
        generate_text_embeddings_parquet(url)
    elif file_extension.lower() in ['jpg', 'jpeg', 'png', 'webp']:
        add_images_descriptions(f"./.storage/.{filename}")

    print(f"I'm ready to answer questions about your file: {filename}")
    return messages