"""Main application entrypoint for the agentic search demo.

Before running, ensure the following environment variables are set:
  export GEMINI_API_KEY='your-gemini-api-key'
  export MONGODB_URL='mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority'

Or create a .env file (see .env.example for template).
See README.md for detailed setup instructions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import textwrap
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agent import (
    AgentConstraints,
    AgenticSearchResponse,
    AgentState,
    RerankResponse,
    apply_rerank,
    await_user,
    build_user_payload,
    decode_tool_inputs,
    get_pydantic_response,
    init_gemini_client,
    should_halt_early,
    shutdown_gemini_client,
    system_prompt,
)
from mongodb import (
    format_tool_transcript,
    init_mongodb_client,
    seed_demo_products,
    shutdown_mongodb_client,
    tool_dispatch,
)
from utils import (
    Ansi,
    ascii_panel,
    render_quality_metrics,
    stylize,
    wrap_text,
)

# Load environment variables from .env file if it exists
load_dotenv(Path(__file__).parent / ".env")

DEFAULT_QUERY = "I am looking for sustainable coffee equipment for the team and energy-efficient office hospitality solutions"
DEFAULT_MIN_RESULTS = 6
DEFAULT_MAX_RESULTS = 12
DEFAULT_MAX_LOOPS = 4


async def init_clients() -> None:
    """Initialize MongoDB and Gemini clients before the agent loop runs."""
    await init_mongodb_client()
    await init_gemini_client()


async def shutdown_clients() -> None:
    """Close external clients when the demo finishes."""
    await shutdown_gemini_client()
    await shutdown_mongodb_client()


async def run_agentic_demo(
    search_query: str,
    *,
    min_required_results: int = 6,
    max_results: int = 12,
    max_loops: int = 4,
    interactive: bool = True,
) -> dict[str, Any]:
    """Drive the full agent loop with formatted CLI output."""
    constraints = AgentConstraints(
        search_query=search_query,
        min_results=min_required_results,
        max_loops=max_loops,
        max_results=max_results,
    )
    state = AgentState()
    loop_logs: list[dict[str, Any]] = []

    for loop_index in range(1, constraints.max_loops + 1):
        user_payload = build_user_payload(loop_index, constraints, state)
        agent_response = await get_pydantic_response(
            system_prompt,
            [user_payload],
            response_model=AgenticSearchResponse,
            temperature=0.6,
        )

        if not isinstance(agent_response, AgenticSearchResponse):
            preview = str(agent_response)
            if len(preview) > 200:
                preview = preview[:200] + " …"
            raise RuntimeError(
                "AgenticSearchResponse validation failed; "
                f"received type {type(agent_response)} with preview: {preview}"
            )

        agent_payload = json.dumps(
            agent_response.model_dump(), indent=2, ensure_ascii=False
        )
        print(
            ascii_panel(
                "AgenticSearchResponse",
                agent_payload.splitlines() or [agent_payload],
                color=Ansi.MAGENTA,
            )
        )

        print()
        print(
            ascii_panel(
                f"Loop {loop_index}",
                [agent_response.loop_summary],
                color=Ansi.CYAN,
            )
        )

        # Record quality for trend tracking
        state.record_quality(agent_response.quality_metrics)
        state.loop_count = loop_index

        # Display quality metrics visually
        print()
        print(render_quality_metrics(agent_response.quality_metrics, state))
        print()

        # Check if we should recommend early halt
        should_halt, halt_reason = should_halt_early(
            agent_response.quality_metrics, state, constraints
        )
        if should_halt and agent_response.decision == "continue":
            print(
                stylize(
                    f"  ⚠️  Halting recommendation: {halt_reason}",
                    Ansi.YELLOW,
                    Ansi.BOLD,
                )
            )
            print()

        print(stylize("Plan", Ansi.BOLD, Ansi.MAGENTA))
        decoded_plan: list[tuple[Any, dict[str, Any]]] = []
        for call in agent_response.plan:
            header = stylize(
                f"Step {call.step}: {call.tool.value}", Ansi.BOLD, Ansi.YELLOW
            )
            print(f"  {header}")
            try:
                inputs_payload = decode_tool_inputs(call.inputs)
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid tool inputs for {call.tool.value}: {exc}"
                ) from exc
            print(wrap_text(f"Goal: {call.goal}", indent=4))
            from utils import render_quality_bar

            print(
                wrap_text(
                    f"Expected Yield: {render_quality_bar(call.expected_yield, width=10)} {call.expected_yield:.2f}",
                    indent=4,
                )
            )
            print(stylize("    Inputs:", Ansi.DIM))
            from mongodb import _format_inputs

            print(
                stylize(
                    textwrap.indent(_format_inputs(inputs_payload), "    "), Ansi.GRAY
                )
            )
            decoded_plan.append((call, inputs_payload))
            state.tool_calls_used += 1

        await await_user("Press Enter to execute the plan...", interactive)

        executed: list[dict[str, Any]] = []
        for call, inputs_payload in decoded_plan:
            tool_fn = tool_dispatch.get(call.tool)
            if not tool_fn:
                print(f"Skipping unsupported tool: {call.tool.value}")
                continue
            try:
                outcome = await tool_fn(inputs_payload)
            except Exception as exc:  # noqa: BLE001
                outcome = {"documents": [], "summary": f"Error executing tool: {exc}"}
            transcript = format_tool_transcript(call.tool, inputs_payload, outcome)
            state.tool_transcripts.appendleft(transcript)
            documents = outcome.get("documents") or []
            state.add_documents(
                documents, source=call.tool.value, max_results=constraints.max_results
            )
            executed.append(
                {"call": call, "inputs": inputs_payload, "outcome": outcome}
            )
            summary_line = outcome.get("summary", "n/a")
            print(stylize("    Outcome:", Ansi.GREEN))
            print(wrap_text(summary_line, indent=6))
            if documents:
                preview = ", ".join(
                    doc.get("_id") or doc.get("name", "unknown")
                    for doc in documents[:3]
                )
                print(stylize("    Top docs:", Ansi.BLUE))
                print(wrap_text(preview, indent=6))

        ranked_ids = state.reranked_ids
        rerank_rationale: str | None = None
        rerank_response: RerankResponse | None = None
        ranked_ids, rerank_rationale, rerank_response = await apply_rerank(
            agent_response.rerank_instruction,
            agent_response.rerank_needed,
            state,
        )

        if rerank_response is not None:
            rerank_payload = json.dumps(
                rerank_response.model_dump(), indent=2, ensure_ascii=False
            )
            print(
                ascii_panel(
                    "RerankResponse (Executed)",
                    rerank_payload.splitlines() or [rerank_payload],
                    color=Ansi.BLUE,
                )
            )
        elif agent_response.rerank_instruction and not agent_response.rerank_needed:
            print(
                stylize(
                    "  ℹ️  Rerank instruction provided but agent marked rerank_needed=false (skipped for efficiency)",
                    Ansi.GRAY,
                )
            )
        elif agent_response.rerank_instruction:
            print(
                ascii_panel(
                    "RerankResponse",
                    ["Instruction issued but no rerank results were returned."],
                    color=Ansi.BLUE,
                )
            )

        if rerank_response is not None:
            await await_user(
                "Press Enter to continue after reviewing rerank...",
                interactive,
            )

        # Feed the agent's reflections back into loop memory for the next turn.
        state.condensed_history.append(
            agent_response.loop_memory.condensed_history_entry
        )
        state.outstanding_questions = agent_response.loop_memory.outstanding_questions
        state.next_focus = agent_response.loop_memory.next_focus

        loop_logs.append(
            {
                "loop": loop_index,
                "summary": agent_response.loop_summary,
                "decision": agent_response.decision,
                "plan": agent_response.plan,
                "executed": executed,
                "rerank": {
                    "instruction": agent_response.rerank_instruction,
                    "ranked_ids": ranked_ids,
                    "rationale": rerank_rationale,
                    "response": rerank_response.model_dump()
                    if rerank_response
                    else None,
                },
            }
        )

        if agent_response.decision != "continue":
            # Once the agent halts or aborts we surface the final narrative and exit.
            color = Ansi.GREEN if agent_response.decision == "halt" else Ansi.RED
            lines = [
                f"Decision: {agent_response.decision.upper()}",
            ]
            if agent_response.final_answer:
                lines.append(f"Answer: {agent_response.final_answer}")
            if agent_response.final_references:
                lines.append(
                    "References: " + ", ".join(agent_response.final_references)
                )
            print()
            print(ascii_panel("Agent Outcome", lines, color=color))
            break

    else:
        message = (
            "Max loops reached; consider increasing max_loops or relaxing constraints."
        )
        print()
        print(ascii_panel("Loop Limit", [message], color=Ansi.YELLOW))

    return {
        "state": state,
        "logs": loop_logs,
    }


def parse_args() -> argparse.Namespace:
    """Translate CLI flags into runtime configuration for the demo."""
    parser = argparse.ArgumentParser(
        description="Run the Atlas agentic search demo loop."
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Search query to provide to the agent.",
    )
    parser.add_argument(
        "--min-results",
        type=int,
        default=DEFAULT_MIN_RESULTS,
        help="Minimum relevant results required before halting.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_MAX_RESULTS,
        help="Maximum documents to retain in memory.",
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        default=DEFAULT_MAX_LOOPS,
        help="Maximum reasoning loops to execute.",
    )
    parser.add_argument(
        "--non-interactive",
        dest="interactive",
        action="store_false",
        help="Disable interactive pauses between tool executions.",
    )
    parser.add_argument(
        "--non-stop",
        dest="non_stop",
        action="store_true",
        help="Alias for --non-interactive; runs the demo without pause prompts.",
    )
    parser.set_defaults(interactive=True)
    return parser.parse_args()


async def run_cli(args: argparse.Namespace) -> None:
    """Run the demo from the command line."""
    await init_clients()
    inserted = await seed_demo_products()
    print(f"Demo dataset ready with {inserted} documents.")

    try:
        await run_agentic_demo(
            args.query,
            min_required_results=args.min_results,
            max_results=args.max_results,
            max_loops=args.max_loops,
            interactive=args.interactive,
        )
    finally:
        await shutdown_clients()


def main() -> None:
    """Entry point for the CLI demo."""
    args = parse_args()
    if getattr(args, "non_stop", False):
        args.interactive = False

    try:
        asyncio.run(run_cli(args))
    except KeyboardInterrupt:
        print("\nInterrupted by user.")


if __name__ == "__main__":
    main()
