# Agentic Search Demo

Multi-module agentic search demo showcasing quality-aware reasoning with MongoDB Atlas and Google Gemini AI. This demo illustrates the core loop of an agentic system: **Sense → Plan → Act → Evaluate → Learn**.

Unlike traditional search that simply retrieves documents, this agent pursues a goal through continuous reasoning, dynamically selecting tools, reflecting on results, and deciding when to halt based on quality metrics.

## What is Agentic Search?

Agentic systems differ from traditional retrieval pipelines. They're autonomous and adaptive:

1. **Sense** the environment and user intent
2. **Plan** a multi-step course of action
3. **Act** by executing tools (MongoDB searches)
4. **Evaluate** results with quality metrics
5. **Learn** and adapt the strategy

This demo showcases these concepts with a **Brain** (the LLM agent) and a **Toolbox** (MongoDB search capabilities).

### Architecture: Brain + Toolbox

The demo follows a clean architectural separation inspired by biological systems:

**🧠 The Brain ([agent.py](agent.py))**
- **Structured Output First**: Pydantic schemas define the agent's response format (`AgenticSearchResponse`, `MongoToolCall`, `LoopQualityMetrics`)
- **System Prompt**: Implements the ReACT pattern (Reason → Act → Reflect) with quality-aware halting criteria
- **Gemini Integration**: Uses Google's Gemini 2.0 Flash model with structured JSON responses
- **Quality Assessment**: 5-dimensional evaluation (relevance, coverage, diversity, confidence, improvement potential)
- **Smart Halting**: Multi-factor decision logic to avoid infinite loops and minimize cost

**🛠️ The Toolbox ([mongodb.py](mongodb.py))**
- **mongo.find.keyword** - Text search with MongoDB's `$text` index
- **mongo.aggregate.faceted** - Faceted search with bucket aggregation
- **mongo.aggregate.pipeline** - Custom aggregation pipelines
- **mongo.aggregate.vector** - Simulated vector search (using pre-computed embeddings)

The agent decides which tools to use, in what order, and when it has gathered enough information to halt.

## Prerequisites

- macOS or Linux (tested on macOS)
- Python 3.11+
- **MongoDB Atlas** cluster (free tier works fine - [Create one here](https://www.mongodb.com/cloud/atlas/register))
- **Google Gemini API** key ([Get one free here](https://aistudio.google.com/app/apikey))

## Project Structure

The demo is organized into four main modules:

- **`app.py`** - Main application entrypoint and CLI orchestration
  - Handles command-line arguments and demo execution flow
  - Manages the agent loop lifecycle and interactive pauses
  - Formats and displays agent responses, quality metrics, and tool execution results

- **`agent.py`** - Agent orchestration and Gemini AI integration
  - Defines the agentic search protocol with Pydantic schemas
  - Manages LLM interactions with structured responses
  - Implements quality-aware reasoning and early halting logic
  - Handles reranking operations

- **`mongodb.py`** - MongoDB access, tools, and sample data
  - Provides MongoDB tool implementations (keyword, faceted, pipeline, vector search)
  - Seeds and manages the demo product catalog
  - Formats tool execution transcripts for agent memory

- **`utils.py`** - Utility functions for formatting and visualization
  - ASCII panel rendering for CLI output
  - Quality metrics visualization with progress bars
  - Text wrapping and ANSI color styling

## Setup

### 1. Install Dependencies

```bash
# From repo root
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set Up MongoDB Atlas (Free Tier)

1. **Create a free cluster** at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register)
2. **Create a database user**:
   - Go to "Database Access" → "Add New Database User"
   - Choose "Password" authentication and save credentials
3. **Whitelist your IP**:
   - Go to "Network Access" → "Add IP Address"
   - Choose "Allow Access from Anywhere" (or add your specific IP)
4. **Get your connection string**:
   - Click "Connect" on your cluster → "Connect your application"
   - Copy the connection string and replace `<password>` with your user password

### 3. Get Your Gemini API Key (Free)

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

### 4. Configure Environment Variables

**Option A: Using .env file (Recommended)**

Copy the example file and fill in your credentials:
```bash
cp .env.example .env
# Then edit .env with your actual keys
```

Your `.env` file should look like:
```bash
GEMINI_API_KEY=your-actual-gemini-api-key
MONGODB_URL=mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority
```

**Option B: Export manually**

```bash
export GEMINI_API_KEY="your-actual-gemini-api-key"
export MONGODB_URL="mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority"
```

**Pro Tip:** Add to your shell profile for persistence:
```bash
echo 'export GEMINI_API_KEY="your-key"' >> ~/.zshrc
echo 'export MONGODB_URL="your-url"' >> ~/.zshrc
source ~/.zshrc
```

## Run the demo

Quick unattended run (2 loops):

```bash
source .venv/bin/activate
python3 app.py --query "Find sustainable coffee equipment for office" --non-stop --max-loops 2
```

Full default run (4 loops):

```bash
source .venv/bin/activate
python3 app.py --query "I need eco-friendly barista equipment and office coffee solutions" --non-stop
```

Interactive run (pauses enabled):

```bash
source .venv/bin/activate
python3 app.py --query "Show me sustainable espresso machines" --max-loops 3
```

Non-stop mode details:

- `--non-stop` is an alias for `--non-interactive` and skips pause prompts between phases.

## Flags you may want

- `--query`: Natural language goal for the agent to pursue
- `--max-loops`: Maximum reasoning loops (default 4)
- `--min-results`: Minimum relevant results required before halting (default 6)
- `--max-results`: Maximum documents to retain in memory (default 12)
- `--non-interactive` / `--non-stop`: Run unattended (no press-enter pauses)

## What You'll See

The demo visualizes the agent's reasoning loop with colorful terminal output:

### Quality Metrics Dashboard
```
Quality Metrics

  Relevance:   ██████████████░░░░░░ 0.70
  Coverage:    ████████████░░░░░░░░ 0.60
  Diversity:   ██████████░░░░░░░░░░ 0.50
  Confidence:  ████████████░░░░░░░░ 0.60
  Next Loop:   ████████████░░░░░░░░ 0.60

  Trend:       ▲ +20.00%
  Overall:     ████████████░░░░░░░░ 0.62
```

### Agent Plan
Each loop, the agent plans multiple tool calls:
- **Step 1**: `mongo.aggregate.vector` - Find espresso equipment semantically
  - Expected Yield: `███████████████░░░░░ 0.75`
- **Step 2**: `mongo.find.keyword` - Search for sustainability certifications
  - Expected Yield: `██████████░░░░░░░░░░ 0.50`

### Execution Results
See real-time MongoDB query results and document previews as the agent gathers information.

### Smart Halting
The agent autonomously decides when to stop based on:
- High confidence (≥0.85) + Low improvement potential (<0.2)
- Excellent relevance (≥0.9)
- Quality plateau (diminishing returns)

## Key Features

- **Quality-Aware Reasoning**: 5-dimensional self-evaluation (relevance, coverage, diversity, confidence, improvement potential)
- **Visual Progress Bars**: See quality metrics evolve across loops
- **Smart Halting Logic**: Avoid infinite loops and minimize LLM costs
- **Tool Yield Prediction**: Agent estimates usefulness of each tool call
- **Efficient Reranking**: Only reranks when truly needed (~50% cost reduction)
- **Trend Detection**: Automatically identifies when additional loops provide diminishing returns

## Troubleshooting

- **Missing environment variables:** Ensure both `GEMINI_API_KEY` and `MONGODB_URL` are set
- **MongoDB connection errors:** Verify your connection string format and that your IP is whitelisted in Atlas
- **Gemini API errors:** Confirm your API key is valid and has sufficient quota
- **Schema or index issues:** If you modify schemas or index logic, re-run and watch the CLI panels for validation errors or empty result sets
