import json
import os
import re

import hal9 as h9

from utils import generate_response, load_messages, save_messages, insert_message

STORAGE_DIR = "./.storage/"
MESSAGES_PATH = os.path.join(STORAGE_DIR, ".website_messages.json")
FILES_STATE_PATH = os.path.join(STORAGE_DIR, ".website_files.json")
WEBSITE_DIR = "website"

SYSTEM_PROMPT = """You can build html applications for user requests. Your replies can include markdown code blocks but they must include a filename parameter after the language. For example,
```javascript filename=code.js
```

The main html file must be named index.html. You can generate other web files like javascript, css, svg that are referenced from index.html. Prefer a single self-contained index.html with embedded <style> and <script> tags unless the user's request benefits from separate files.
"""

DEFAULT_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Website</title>
</head>
<body>
</body>
</html>
"""

# Matches fenced code blocks tagged with a `filename=` parameter, e.g.
# ```html filename=index.html
# ...content...
# ```
FILENAME_BLOCK_RE = re.compile(
    r"```[ \t]*[\w+-]*[ \t]+filename=(?P<filename>[^\s`]+)[ \t]*\r?\n(?P<content>.*?)```",
    re.DOTALL,
)


def load_website_files(path=FILES_STATE_PATH):
    """Loads the previously generated website files (if any) from disk."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as file:
        try:
            return json.load(file)
        except json.JSONDecodeError:
            return {}


def save_website_files_state(files, path=FILES_STATE_PATH):
    """Persists the current set of generated website files as JSON state."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(files, file, ensure_ascii=False, indent=2)


def extract_files(response_content, default=None):
    """
    Parses fenced code blocks tagged with a `filename=` parameter out of a
    model response and returns a dict mapping filename -> file content. Files
    already present in `default` are preserved unless the response redefines
    them, so incremental change requests only touch the files that were
    actually regenerated.
    """
    files = dict(default) if default else {}

    for match in FILENAME_BLOCK_RE.finditer(response_content):
        filename = match.group("filename").strip().strip("`")
        content = match.group("content")
        if content.endswith("\n"):
            content = content[:-1]
        files[filename] = content

    return files


def write_website_files(files, directory=WEBSITE_DIR):
    """Writes every generated file directly to disk via plain file I/O."""
    os.makedirs(directory, exist_ok=True)
    for filename, content in files.items():
        file_path = os.path.join(directory, filename)
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)


def website_generator(prompt):
  """
  Builds or modifies a website based on user description or a change request
    'prompt' with user change or requirements
  """

  messages = load_messages(file_path=MESSAGES_PATH)
  files = load_website_files()

  if len(messages) < 1:
      messages = insert_message(messages, "system", SYSTEM_PROMPT)

  messages = insert_message(messages, "user", prompt)

  model_response = generate_response(messages, reasoning_effort="default")
  response_content = model_response.choices[0].message.content

  files = extract_files(response_content, default=files)

  if not files.get("index.html"):
      files["index.html"] = DEFAULT_INDEX_HTML

  messages = insert_message(messages, "assistant", response_content)

  save_messages(messages, file_path=MESSAGES_PATH)
  save_website_files_state(files)
  write_website_files(files)

  relative_path = h9.deploy(WEBSITE_DIR, target="hal9", url=os.environ.get("HAL9_URL", "https://api.hal9.com"))
  print(f"The website got deployed to: {relative_path}")

  messages = insert_message(messages, "user", "briefly describe what was accomplished")
  summary_response = generate_response(messages, reasoning_effort="none")
  summary = summary_response.choices[0].message.content

  return summary

website_generator_description = {
    "type": "function",
    "function": {
        "name": "website_generator",
        "description": "This function creates or modifies a website based on a user's description or change requests. It dynamically generates HTML, CSS, JavaScript, and other web assets as specified in the input prompt. The function maintains a stateful interaction, allowing for iterative website building and modification.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "A user-provided description of the website requirements or specific modification requests.",
                },
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
    },
}
