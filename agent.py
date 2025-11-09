"""Agent orchestration and Gemini AI integration for agentic search."""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from google import genai
from pydantic import BaseModel, Field, ValidationError

# Global Gemini client
ganai_client: genai.Client | None = None

# Model configuration
gemini_model = "gemini-2.0-flash"
low_safety_settings = [
    genai.types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="BLOCK_ONLY_HIGH",
    ),
    genai.types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
    genai.types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="BLOCK_ONLY_HIGH",
    ),
    genai.types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
]


class OutputLanguage(str, Enum):
    ENGLISH = "English"
    TURKISH = "Turkish"


class MongoTool(str, Enum):
    KEYWORD = "mongo.find.keyword"
    FACETED = "mongo.aggregate.faceted"
    PIPELINE = "mongo.aggregate.pipeline"
    VECTOR = "mongo.aggregate.vector"


class MongoToolCall(BaseModel):
    step: int = Field(
        ..., ge=1, description="Sequential step number within the current loop."
    )
    goal: str = Field(
        ..., description="Short natural-language justification for the tool call."
    )
    tool: MongoTool
    inputs: str = Field(
        ...,
        description="JSON string payload executed against MongoDB; must parse to an object.",
    )
    expected_yield: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Expected usefulness of this tool call (0-1)",
    )


class LoopQualityMetrics(BaseModel):
    """Evaluate results quality after each loop iteration."""

    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How well results match user intent (0-1)",
    )
    coverage_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How well we've covered the search space (0-1)",
    )
    diversity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How diverse the result set is (0-1)",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in results (0-1)",
    )
    improvement_potential: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated value of another loop (0-1)",
    )


class LoopMemoryUpdate(BaseModel):
    condensed_history_entry: str = Field(
        ...,
        description="Bullet-style recap to append to loop_memory.condensed_history.",
    )
    outstanding_questions: list[str] = Field(default_factory=list)
    next_focus: str = Field(
        ..., description="Most actionable intent for the next loop."
    )


class AgenticSearchResponse(BaseModel):
    plan: list[MongoToolCall] = Field(
        ...,
        description="Ordered tool calls the controller must execute in this loop (>=2 entries unless halting).",
    )
    loop_summary: str = Field(
        ..., description="Narrative recap of current understanding in OutputLanguage."
    )
    quality_metrics: LoopQualityMetrics = Field(
        ..., description="Self-evaluation of current results quality"
    )
    rerank_instruction: str | None = Field(
        default=None,
        description="How to score/merge tool outputs before finalizing results.",
    )
    rerank_needed: bool = Field(
        default=False,
        description="Set true if reranking would significantly improve results",
    )
    decision: Literal["continue", "halt", "abort"]
    loop_memory: LoopMemoryUpdate
    final_answer: str | None = Field(
        default=None,
        description="Only populated when decision is 'halt'; contains the narrated answer.",
    )
    final_references: list[str] = Field(
        default_factory=list,
        description="List of product ids the agent believes satisfy the user once halting.",
    )


class RerankResponse(BaseModel):
    ranked_ids: list[str] = Field(
        ..., description="Ordered list of document ids to prioritize."
    )
    rationale: str = Field(..., description="Short explanation of the ranking logic.")


output_language = OutputLanguage.ENGLISH

system_prompt: list[str] = [
    "You are Atlas, an agent orchestrator for an agentic search demo.",
    "Follow a ReACT cadence: Observe context, Plan >=2 tool calls, Act, Reflect, and request another loop until you can confidently halt.",
    "Always emit JSON that conforms to the AgenticSearchResponse schema; do not include extra keys.",
    "Keep loop_summary, plan.goal, and final_answer in the requested output language.",
    "Use the allowed toolbox identifiers only: mongo.find.keyword, mongo.aggregate.faceted, mongo.aggregate.pipeline, mongo.aggregate.vector.",
    "Represent each plan entry's inputs as a JSON string matching the selected tool's contract.",
    "Respect constraints: mongo.aggregate.pipeline payloads should include a $limit stage (<= 20), though one will be auto-injected if missing; mongo.aggregate.vector requires query_embedding_label from ['espresso-barista','travel-kit','eco-office'] and top_k <= 10.",
    "QUALITY ASSESSMENT: After each loop, honestly evaluate quality_metrics: relevance (how well results match intent), coverage (search space explored), diversity (variety in results), confidence (certainty in recommendations), and improvement_potential (expected value of another loop).",
    "TOOL YIELD: For each planned tool call, estimate expected_yield (0-1) based on how useful you expect it to be given current state.",
    "RERANKING EFFICIENCY: Only set rerank_needed=true when reranking would significantly improve results (e.g., many candidates with mixed quality, final loop preparation, or unclear ranking).",
    "HALTING CRITERIA: Consider halting when confidence >= 0.85 AND improvement_potential < 0.2, OR when relevance >= 0.9 with sufficient results. Agent should self-assess and halt early if quality has plateaued.",
    "Only set decision to 'halt' when you have high confidence in results and list at least min_required_results distinct documents in final_references.",
    "When you abort, explain the blocking issue and populate next_focus with recovery advice.",
]

rerank_system_prompt: list[str] = [
    "You are Atlas' reranker module.",
    "Given an instruction and candidate documents, return ranked_ids in best-first order.",
    "Only use ids that appear in the candidate list. Keep rationale concise.",
]


async def init_gemini_client() -> None:
    """Initialize the Gemini client."""
    global ganai_client

    if ganai_client is not None:
        return

    genai_api_key = os.environ.get("GEMINI_API_KEY")
    if not genai_api_key:
        import sys
        print("\n❌ Error: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        print("\nExport the variable:", file=sys.stderr)
        print("  export GEMINI_API_KEY='your-gemini-api-key'", file=sys.stderr)
        print("\nSee README.md for detailed setup instructions.\n", file=sys.stderr)
        sys.exit(1)
    ganai_client = genai.Client(api_key=genai_api_key)
    print("Initialized Gemini client.")


async def shutdown_gemini_client() -> None:
    """Close the Gemini client."""
    global ganai_client

    if ganai_client is not None:
        aio_iface = getattr(ganai_client, "aio", None)
        close_coro = getattr(aio_iface, "aclose", None)
        if callable(close_coro):
            await close_coro()
        ganai_client = None


async def get_pydantic_response(
    system_instructions: list[str],
    user_contents: list[str | genai.types.Content],
    safety_settings: list[genai.types.SafetySetting] = low_safety_settings,
    response_model: type[BaseModel] | None = None,
    model_name: str = gemini_model,
    temperature: float = 0.8,
) -> BaseModel | dict | str | None:
    """Call Gemini until we obtain a valid response, with structured retries.

    This helper provides the behaviour for the script can request structured
    output and receive either a validated Pydantic instance or plain text.
    Steps performed:
    1. Build a ``GenerateContentConfig`` that enforces the response schema.
    2. Attempt the request up to ten times, backing off on 429/5xx errors.
    3. If validation fails, adjust temperature to encourage a different shape.
    """
    if ganai_client is None:
        raise RuntimeError(
            "Gemini client is not initialised. Call init_gemini_client() first."
        )

    config = genai.types.GenerateContentConfig(
        safety_settings=safety_settings,
        system_instruction=system_instructions,
        response_mime_type="application/json" if response_model else "text/plain",
        response_schema=response_model.model_json_schema() if response_model else None,
        temperature=temperature,
        max_output_tokens=8192,
    )

    response_text = None
    validated_response = None
    last_error: Exception | None = None
    last_response_text: str | None = None

    for attempt in range(1, 11):
        try:
            response = await ganai_client.aio.models.generate_content(
                model=model_name,
                contents=user_contents,
                config=config,
            )
            response_text = response.text
            last_response_text = response_text

            if not response_text:
                continue

            response_dict = None
            if response_model:
                response_dict = json.loads(response_text)
                validated_response = response_model.model_validate(response_dict)
                if validated_response:
                    break
            elif response_text:
                break

        except genai.errors.APIError as exc:
            last_error = exc
            if exc.code == 429:
                await asyncio.sleep(5 * attempt)
            if 500 <= exc.code < 600:
                await asyncio.sleep(1)
            else:
                print(f"get_pydantic_response: APIError: {exc}")
                return None
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            config.temperature = 0.5 + 0.05 * attempt
            print(f"get_pydantic_response: PARSING ERROR: {exc}")
            await asyncio.sleep(1)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"get_pydantic_response: unhandled error: {exc} at {attempt}")
            return None

    if not response_text:
        if response_model:
            preview = ""
            if last_error:
                preview = f" Last error: {last_error!r}"
            raise RuntimeError(
                "get_pydantic_response: no response text received." + preview
            )
        return None

    if response_model:
        if validated_response:
            return validated_response
        debug_message = (
            "get_pydantic_response: failed to validate response_model after retries."
        )
        if last_response_text:
            preview = last_response_text.strip()
            if len(preview) > 500:
                preview = preview[:500] + " …"
            debug_message += f" Last response preview: {preview}"
        if last_error:
            debug_message += f" Last error: {last_error!r}"
        raise RuntimeError(debug_message)

    return response_text


@dataclass
class AgentConstraints:
    """Immutable parameters describing the user's request and loop bounds."""

    search_query: str
    min_results: int
    max_results: int
    max_loops: int
    output_language: OutputLanguage = output_language


@dataclass
class AgentState:
    """Mutable scratchpad that the controller updates between reasoning loops."""

    condensed_history: list[str] = field(default_factory=list)
    outstanding_questions: list[str] = field(default_factory=list)
    tool_transcripts: deque[str] = field(default_factory=lambda: deque(maxlen=6))
    collected_documents: dict[str, dict[str, Any]] = field(default_factory=dict)
    reranked_ids: list[str] = field(default_factory=list)
    next_focus: str = "Interpret the user query and draft an opening plan."
    quality_history: list[float] = field(default_factory=list)
    tool_calls_used: int = 0
    loop_count: int = 0

    def add_documents(
        self, docs: list[dict[str, Any]], source: str, max_results: int
    ) -> None:
        """Merge tool output into ``collected_documents`` with light scoring."""
        for rank, doc in enumerate(docs):
            doc_id = doc.get("_id")
            if not doc_id:
                continue
            base_score = (
                doc.get("score")
                or doc.get("vector_score")
                or doc.get("sustainability_index")
                or doc.get("popularity")
                or 0
            )
            jitter = (rank + 1) * 0.01
            self.collected_documents[doc_id] = {
                "_id": doc_id,
                "name": doc.get("name"),
                "category": doc.get("category"),
                "price": doc.get("price"),
                "price_bucket": doc.get("price_bucket"),
                "sustainability_index": doc.get("sustainability_index"),
                "source": source,
                "score": float(base_score) - jitter,
            }

        if len(self.collected_documents) > max_results:
            sorted_items = sorted(
                self.collected_documents.values(),
                key=lambda item: item.get("score", 0),
                reverse=True,
            )[:max_results]
            self.collected_documents = {item["_id"]: item for item in sorted_items}

    def snapshot_candidates(self, limit: int) -> list[dict[str, Any]]:
        """Return a sorted list of top candidates for prompt rendering."""
        ranked = sorted(
            self.collected_documents.values(),
            key=lambda item: item.get("score", 0),
            reverse=True,
        )
        return ranked[:limit]

    def transcripts_text(self) -> str:
        """Render tool transcripts as newline-delimited text snippets."""
        if not self.tool_transcripts:
            return "(no tool transcripts yet)"
        return "\n\n".join(self.tool_transcripts)

    def condensed_history_text(self) -> str:
        """Render the condensed history bullet list for the agent prompt."""
        if not self.condensed_history:
            return "(empty)"
        return "\n".join(f"- {entry}" for entry in self.condensed_history)

    def outstanding_questions_text(self) -> str:
        """Render outstanding questions for the agent to address next."""
        if not self.outstanding_questions:
            return "(none)"
        return "\n".join(f"- {item}" for item in self.outstanding_questions)

    def is_plateauing(self, current_score: float, threshold: float = 0.05) -> bool:
        """Check if quality improvement has stalled."""
        if len(self.quality_history) < 2:
            return False
        recent_improvement = current_score - self.quality_history[-1]
        return recent_improvement < threshold

    def record_quality(self, metrics: Any) -> None:
        """Record overall quality score for this loop."""
        overall = (
            metrics.relevance_score * 0.35
            + metrics.confidence * 0.35
            + metrics.coverage_score * 0.15
            + metrics.diversity_score * 0.15
        )
        self.quality_history.append(overall)


def should_halt_early(
    metrics: Any, state: AgentState, constraints: AgentConstraints
) -> tuple[bool, str]:
    """Multi-factor halting decision with reasoning.

    Returns (should_halt, reason)
    """
    has_enough_results = len(state.collected_documents) >= constraints.min_results
    high_confidence = metrics.confidence >= 0.85
    low_improvement = metrics.improvement_potential < 0.2
    strong_relevance = metrics.relevance_score >= 0.8
    excellent_relevance = metrics.relevance_score >= 0.9

    if has_enough_results and high_confidence and low_improvement:
        return True, "High confidence with low improvement potential"

    if has_enough_results and excellent_relevance and metrics.confidence >= 0.75:
        return True, "Excellent relevance achieved"

    if state.quality_history:
        current_overall = state.quality_history[-1]
        if state.is_plateauing(current_overall):
            return True, "Quality has plateaued (diminishing returns)"

    if has_enough_results and strong_relevance and metrics.improvement_potential > 0.3:
        return False, "Good results but room for improvement"

    return False, "Continue exploring"


def _render_candidates_for_prompt(candidates: list[dict[str, Any]]) -> str:
    """Format collected documents so Gemini can understand current evidence."""
    if not candidates:
        return "(no candidates collected)"
    lines: list[str] = []
    for item in candidates:
        lines.append(
            f"- id={item['_id']} | name={item['name']} | category={item['category']} | price={item['price']} | sustainability={item.get('sustainability_index')} | source={item['source']} | score={round(item.get('score', 0), 2)}"
        )
    return "\n".join(lines)


def build_user_payload(
    loop_index: int, constraints: AgentConstraints, state: AgentState
) -> str:
    """Construct the ReACT-style prompt that feeds context back to Gemini."""
    quality_trend = ""
    if len(state.quality_history) >= 2:
        delta = state.quality_history[-1] - state.quality_history[-2]
        quality_trend = f"\nquality_trend: {'+' if delta >= 0 else ''}{delta:.2%}"

    return textwrap.dedent(
        f"""
        # User Query
        {constraints.search_query}

        # Loop Context
        loop_number: {loop_index}
        max_loops: {constraints.max_loops}
        min_required_results: {constraints.min_results}
        max_results: {constraints.max_results}
        tool_calls_used: {state.tool_calls_used}
        next_focus_hint: {state.next_focus}{quality_trend}

        # Loop Memory
        condensed_history:\n{state.condensed_history_text()}

        outstanding_questions:\n{state.outstanding_questions_text()}

        # Tool Transcripts (latest first)
{state.transcripts_text()}

        # Collected Candidates (trimmed)
{_render_candidates_for_prompt(state.snapshot_candidates(limit=constraints.max_results))}

        # Guidance
        Respond with a JSON object that validates against AgenticSearchResponse.
        - Include quality_metrics with honest self-assessment of relevance, coverage, diversity, confidence, and improvement_potential
        - Each tool call must include expected_yield (0-1) indicating how useful you expect it to be
        - Set rerank_needed=true only if reranking would significantly improve results (e.g., many candidates, final loop, or mixed quality)
        - Propose at least two tool calls unless you plan to halt
        - Consider halting if: confidence >= 0.85 AND improvement_potential < 0.2, OR relevance >= 0.9
        - If results look weak, explain gaps and set decision to "continue" with focused next_focus
        """
    ).strip()


def build_rerank_payload(instruction: str, candidates: list[dict[str, Any]]) -> str:
    """Format rerank instructions plus candidate JSON for Gemini."""
    return textwrap.dedent(
        f"""
        # Rerank Instruction
        {instruction}

        # Candidates
        {json.dumps(candidates, indent=2, ensure_ascii=False)}
        """
    ).strip()


async def await_user(prompt: str, interactive: bool) -> None:
    """Pause execution when interactive mode is enabled."""
    if not interactive:
        return
    await asyncio.to_thread(input, prompt)


async def apply_rerank(
    instruction: str | None,
    rerank_needed: bool,
    state: AgentState,
) -> tuple[list[str], str | None, RerankResponse | None]:
    """Invoke Gemini reranker only when agent signals it's needed."""
    if not rerank_needed or not instruction or not state.collected_documents:
        return state.reranked_ids, None, None

    candidates = state.snapshot_candidates(limit=len(state.collected_documents))
    payload = build_rerank_payload(instruction, candidates)
    response = await get_pydantic_response(
        rerank_system_prompt,
        [payload],
        response_model=RerankResponse,
        temperature=0.2,
    )
    ranked_ids = response.ranked_ids if isinstance(response, RerankResponse) else []
    if not ranked_ids:
        return state.reranked_ids, None, None

    state.reranked_ids = ranked_ids
    rationale = response.rationale if isinstance(response, RerankResponse) else None
    response_model = response if isinstance(response, RerankResponse) else None
    return ranked_ids, rationale, response_model


def decode_tool_inputs(raw: str) -> dict[str, Any]:
    """Parse a JSON string emitted by the agent into a Python dictionary."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Tool inputs must be valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Tool inputs must decode to a JSON object.")
    return data
