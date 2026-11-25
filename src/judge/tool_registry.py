"""Tool catalog + dispatcher for the supervisory agent (observable, real calls)."""

from __future__ import annotations

from typing import Any, Callable

from src.eval.behavioral import analyze_behavior
from src.judge.tools import JudgeTools


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "get_rubric",
        "description": "Read the evaluation rubric dimensions and scoring rules.",
        "args": {},
    },
    {
        "name": "read_target_system_prompt",
        "description": "Read the target assistant's system instructions (read-only).",
        "args": {},
    },
    {
        "name": "retrieve_safety_principles",
        "description": (
            "Semantic retrieval from Chroma of safety principles/examples relevant "
            "to the user prompt and target response."
        ),
        "args": {"query": "optional override query string; default uses prompt+response"},
    },
    {
        "name": "inspect_conversation_history",
        "description": "Read recent conversation turns for this session.",
        "args": {"limit": "optional int, default 6"},
    },
    {
        "name": "retrieve_prior_evaluations",
        "description": "Fetch recent prior evaluation summaries from memory.",
        "args": {"limit": "optional int, default 3"},
    },
    {
        "name": "analyze_behavior",
        "description": (
            "Deterministic behavioral cue analysis: intent class, refusal, "
            "over/under-refusal, fabrication flags, and whether explicit format "
            "constraints in the prompt (word/sentence/bullet counts) are met."
        ),
        "args": {},
    },
]


def tool_catalog_text() -> str:
    lines = []
    for t in TOOL_SPECS:
        lines.append(f"- {t['name']}: {t['description']} args={t['args']}")
    return "\n".join(lines)


class ToolExecutor:
    """Executes real tools via JudgeTools; never fabricates results."""

    def __init__(
        self,
        tools: JudgeTools,
        prompt: str,
        response: str,
        hints: dict[str, Any] | None = None,
    ):
        self.tools = tools
        self.prompt = prompt
        self.response = response
        self.hints = hints or {}
        self.observations: list[dict[str, Any]] = []

    def available_names(self) -> list[str]:
        return [t["name"] for t in TOOL_SPECS]

    def run(
        self, name: str, args: dict | None = None, selected_by: str = "agent"
    ) -> dict[str, Any]:
        """Execute one real tool call. `selected_by` records who chose it (provenance)."""
        obs = self._run(name, args)
        obs["selected_by"] = selected_by
        obs["call_index"] = len(self.observations) + 1
        self.observations.append(obs)
        return obs

    def _run(self, name: str, args: dict | None = None) -> dict[str, Any]:
        args = args or {}
        name = str(name).strip()
        if name not in self.available_names():
            return {
                "tool": name,
                "ok": False,
                "args": args,
                "error": f"Unknown tool: {name}",
                "result": None,
            }

        try:
            if name == "get_rubric":
                result = self.tools.get_rubric()
            elif name == "read_target_system_prompt":
                result = {"system_prompt": self.tools.read_target_system_prompt()}
            elif name == "retrieve_safety_principles":
                q = str(args.get("query") or "").strip()
                if q:
                    hits = self.tools.store.retrieve(q, k=args.get("k"))
                    self.tools._log(
                        "retrieve_safety_principles",
                        {"query": q[:200], "backend": self.tools.store.backend},
                        [{"id": h.get("id"), "title": h.get("title")} for h in hits],
                    )
                else:
                    hits = self.tools.retrieve_safety_principles(self.prompt, self.response)
                result = {
                    "backend": self.tools.store.backend,
                    "hits": [
                        {
                            "id": h.get("id"),
                            "title": h.get("title"),
                            "category": h.get("category"),
                            "doc_type": h.get("doc_type"),
                            "score": h.get("score"),
                            "excerpt": (h.get("text") or h.get("principle") or "")[:240],
                        }
                        for h in hits
                    ],
                }
            elif name == "inspect_conversation_history":
                limit = int(args.get("limit") or 6)
                hist = self.tools.inspect_conversation_history(limit=limit)
                result = {"turns": hist}
            elif name == "retrieve_prior_evaluations":
                limit = int(args.get("limit") or 3)
                evals = self.tools.retrieve_prior_evaluations(limit=limit)
                result = {
                    "evaluations": [
                        {
                            "id": e.get("id"),
                            "overall_score": e.get("overall_score"),
                            "prompt": (e.get("prompt") or "")[:120],
                        }
                        for e in evals
                    ]
                }
            elif name == "analyze_behavior":
                full = analyze_behavior(self.prompt, self.response, self.hints)
                fmt = full.get("format_check") or {}
                # Compact view for the agent's context (same facts, less JSON)
                result = {
                    "intent_class": full.get("intent_class"),
                    "intent_source": full.get("intent_source"),
                    "declared_expectations": full.get("declared_hints"),
                    "flags": full.get("flags"),
                    "declines_request": full.get("declines_request"),
                    "acknowledges_uncertainty": full.get("acknowledges_uncertainty"),
                    "over_refusal": full.get("over_refusal"),
                    "under_refusal": full.get("under_refusal"),
                    "possible_fabrication": full.get("possible_fabrication"),
                    "format_constraints_in_prompt": [
                        {"requirement": r.get("label"), "observed": r.get("observed"), "passed": r.get("passed")}
                        for r in (fmt.get("results") or [])
                    ],
                    "format_violations": full.get("format_violations_text"),
                    "response_stats": fmt.get("stats"),
                }
                self.tools._log("analyze_behavior", {}, result.get("flags"))
            else:
                raise RuntimeError(f"Unhandled tool {name}")
            return {"tool": name, "ok": True, "args": args, "result": result}
        except Exception as e:
            return {"tool": name, "ok": False, "args": args, "error": str(e), "result": None}
