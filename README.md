# Hal9 Agent

A conversational AI agent built for the [Hal9](https://hal9.com) platform. Hal9 Agent
routes every user message through a tool-calling loop backed by an LLM (served over
Groq), automatically selecting and invoking the right specialized tool — math solving,
CSV/text/image analysis, app generation, website building, or even opening a pull
request against a GitHub repository — before replying with a single, user-facing
answer.

![Hal9 Agent](thumbnail.jpg)

## Overview

Hal9 Agent implements a classic **agentic tool-routing loop**:

1. The user's message (plus any optional uploaded files) is appended to the running
   conversation history.
2. The message history and the full list of available tools are sent to the LLM with
   `tool_choice="required"`, forcing the model to always pick a tool rather than
   free-form reply.
3. The chosen tool is executed, and its result is fed back into the conversation as a
   `tool` message.
4. The loop repeats (up to a small step budget) until the model calls the special
   `final_response` tool, which is what actually gets shown to the user.

This design keeps routing decisions, tool execution, and the final user-facing message
cleanly separated, and makes it easy to add new capabilities by simply registering a
new tool.

### Core components

| File | Purpose |
| --- | --- |
| `app.py` | Entry point — loads conversation state, registers tools, and runs the tool-calling loop for each incoming message. |
| `clients.py` | Configures the Groq API client (via the Hal9 proxy) used to talk to the underlying LLM. |
| `utils.py` | Shared helpers: response generation (with retry/fallback logic for malformed tool calls), message persistence, tool execution/dispatch, URL/file ingestion, and image description generation. |
| `data/hal9.txt` | Reference knowledge base used by the `answer_hal9_questions` tool to answer questions about the Hal9 product itself. |
| `coworker.yaml` | Welcome message configuration shown when the chatbot starts. |

### Available tools

Each tool in `tools/` is a `(description, function)` pair registered in `app.py`. The
model can invoke any of the following:

| Tool | Module | Description |
| --- | --- | --- |
| `final_response` | `tools/other_tools.py` | Delivers the final, user-facing reply. Always called last. |
| `solve_math_problem` | `tools/calculator.py` | Solves math problems with a step-by-step explanation plus executable Python code. |
| `answer_generic_question` | `tools/generic.py` | Answers general-knowledge questions and small talk. |
| `analyze_csv` | `tools/csv_agent.py` | Explores and analyzes CSV files (overview, stats, charts). |
| `analyze_text_file` | `tools/text_agent.py` | Answers questions about uploaded text/PDF content using chunked retrieval. |
| `images_management_system` | `tools/image_agent.py` | Generates, edits, and describes images. |
| `answer_hal9_questions` | `tools/hal9.py` | Answers questions about the Hal9 platform itself. |
| `streamlit_generator` | `tools/streamlit.py` | Generates and iteratively fixes interactive Streamlit apps. |
| `shiny_generator` | `tools/shiny.py` | Generates and iteratively fixes interactive Shiny (R) apps. |
| `fastapi_generator` | `tools/fastapi.py` | Generates FastAPI backend code/apps. |
| `website_generator` | `tools/website.py` | Builds or modifies static HTML/CSS/JS websites. |
| `generate_csv_based_pdf` | `tools/pdf_to_csv.py` | Extracts tabular data from PDFs into CSV. |
| `python_execution` | `tools/python_execution.py` | Writes and safely executes Python code in an isolated virtual environment, fixing errors automatically. |
| `claude_code_github` | `tools/claude_code.py` | Clones a GitHub repository, applies a requested code change using [Claude Code](https://docs.claude.com/en/docs/claude-code), and opens a pull request into `main`. |

## Installation

### Prerequisites

- Python 3.10+
- A [Hal9](https://hal9.com) account/token, used to proxy requests to Groq and
  Replicate
- (Optional) A GitHub personal access token and Anthropic API key if you want to use
  the `claude_code_github` tool

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-org>/hal9-agent.git
   cd hal9-agent
   ```

2. Install the Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Install the Hal9 CLI/runtime (used for local execution, deployment, and the `hal9`
   Python module imported as `h9`):

   ```bash
   pip install hal9
   ```

4. Configure the required environment variables:

   ```bash
   export HAL9_URL="https://api.hal9.com"
   export HAL9_TOKEN="<your-hal9-token>"

   # Only needed for the claude_code_github tool
   export GITHUB_PAT="<your-github-personal-access-token>"
   export ANTHROPIC_API_KEY="<your-anthropic-api-key>"
   ```

## Usage

### Running locally

Hal9 apps read a single line of user input from stdin and print the response to
stdout, persisting conversation state under `./.storage/`:

```bash
mkdir -p .storage
echo "What can you help me with?" | python app.py
```

Because conversation history is persisted to `./.storage/.messages.json`, running the
command again continues the same conversation.

### Example interactions

```bash
# Small talk — routed straight to final_response
echo "Hey, how's it going?" | python app.py

# Math — routed to solve_math_problem
echo "What is the derivative of x^3 + 2x?" | python app.py

# Ask about Hal9 itself — routed to answer_hal9_questions
echo "What does Hal9 do?" | python app.py

# Share a file for analysis — the agent detects the URL and ingests it
echo "https://example.com/data.csv" | python app.py
echo "What are the top 5 rows by revenue?" | python app.py

# Request a GitHub code change and get a pull request back
echo "In repo acme/api, add input validation to the /users endpoint" | python app.py
```

### Deploying to Hal9

This repository is wired up to auto-deploy via GitHub Actions
(`.github/workflows/main.yaml`) whenever changes are pushed to `main`:

```bash
hal9 deploy . --name hal9 --access public --url "$HAL9_URL" \
  --title Hal9 --description "Conversations and content creation"
```

## Contributing

Contributions are welcome! To propose a change:

1. Fork the repository and create a feature branch:

   ```bash
   git checkout -b feature/my-improvement
   ```

2. Make your changes. When adding a new tool:
   - Create a new module in `tools/` that exports a function and a matching
     `*_description` OpenAI-style function-calling schema (see `tools/calculator.py`
     for a minimal example).
   - Register both in `app.py` by importing them and adding them to
     `tools_descriptions` and `tools_functions`.
   - Keep tool functions side-effect-scoped to `./.storage/` for any files they read
     or write, so state stays isolated per conversation.
3. Test your changes locally (see [Usage](#usage) above).
4. Commit your changes with a clear, descriptive message.
5. Push to your fork and open a pull request describing the motivation and behavior
   of your change.

Please keep pull requests focused — prefer several small PRs over one large one — and
avoid introducing new required environment variables unless necessary.

## License

This project is licensed under the [MIT License](LICENSE).
