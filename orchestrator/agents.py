# =============================================================================
# orchestrator/agents.py
#
# LLM-backed agent classes that form the reasoning pipeline.
# Each agent receives a structured context dict, calls the LLM service via
# an injected client, and returns either a typed object or plain text.
#
# Pipeline order
# --------------
#   1. PlannerAgent   — reads VLM output + tool registry, selects a tool and
#                       target, returns a step-by-step reasoning trace.
#   2. CriticAgent    — validates the Planner's decision, corrects modality /
#                       target mismatches, returns a ToolDecision + raw dict.
#   3. ExplainerAgent — turns a need_clarification result into natural-language
#                       text shown to the user.
#   4. ReporterAgent  — writes a concise final report after tool execution.
#   5. VerifierAgent  — deterministic (no LLM): checks that the mask file
#                       exists, is non-empty, and matches the plan.
#
# All LLM agents accept an llm_client (LLMClient) at construction time so
# they can be tested with any compatible stub.
# =============================================================================

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import json
from contracts import ToolDecision, decision_from_dict


def _extract_first_json_obj(text: str) -> Optional[dict]:
    """Parse the first well-formed JSON object out of an arbitrary string.

    Scans character-by-character using a brace counter so it tolerates
    surrounding prose, markdown fences, or partial content before/after the
    object.  Returns None if no valid object is found.
    """
    s = str(text) if text else ""
    stack, start = 0, None
    for i, ch in enumerate(s):
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}" and stack > 0:
            stack -= 1
            if stack == 0 and start is not None:
                try:
                    d = json.loads(s[start:i+1])
                    if isinstance(d, dict):
                        return d
                except Exception:
                    pass
                start = None
    return None


def _safe_list(x) -> List[Any]:
    """Return x as a list; wraps scalars and converts None to []."""
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


# ---------------------------------------------------------------------------
# PlannerAgent
# ---------------------------------------------------------------------------

class PlannerAgent:
    """Selects the best segmentation tool and target from VLM output + user instruction.

    Calls LLMClient.generate_json with a structured prompt that encodes the
    available tools' decision logic.  Returns a dict with keys:
      selected_tool, target, modality, reasoning, steps.
    """

    SYS = """You are the Planner in an agentic medical image segmentation system.

You receive a JSON object containing:
- "vlm_output": what the Vision-Language Model extracted from the medical image (modality, anatomy, confidence, task intent)
- "user_instruction": the raw text instruction from the user
- "available_tools": the list of tools available with their capabilities, modality support, and decision logic

Your job is to produce a segmentation plan: decide WHICH tool to use and WHAT to pass to it.

=== HOW TO REASON ===

Step 1 - Understand the image:
  Read "vlm_output" carefully. What modality was detected? What anatomy region? How confident?

Step 2 - Understand the user's intent:
  Read "user_instruction". What exactly does the user want to segment?
  Is it a healthy anatomical structure (liver, spleen, kidney...) or a pathological finding (tumor, lesion, cancer, infection...)?

Step 3 - Read the tool notes:
  Each tool in "available_tools" has "notes" and "modalities" fields. Read them carefully.
  The notes explain exactly when to use each tool and when NOT to.

Step 4 - Select the best tool:
  Match modality + target type to the tool whose notes best fit.
  Do not guess. Use the tool notes as your decision guide.

Step 5 - Build your output:
  State which tool you chose, why, and what the exact target string is.

=== OUTPUT FORMAT ===

Return EXACTLY this JSON and nothing else. No markdown, no extra text:
{
  "selected_tool": "<totalseg|voxtell|biomedparse>",
  "target": "<exact segmentation target extracted from user instruction>",
  "modality": "<normalized modality: CT|MRI|X-Ray|Ultrasound|PET|Pathology|OCT|Fundus|Dermoscopy|Endoscopy>",
  "reasoning": "<2-3 sentence explanation: what modality, what target type, which tool note matched>",
  "steps": [
    {"step": 1, "action": "read_modality", "detail": "<modality and confidence from VLM>"},
    {"step": 2, "action": "read_target", "detail": "<what user wants: anatomical or pathological>"},
    {"step": 3, "action": "match_tool", "detail": "<which tool note matched and why>"},
    {"step": 4, "action": "confirm", "detail": "<final confirmation of tool and target>"}
  ]
}
"""

    def __init__(self, llm_client):
        self.llm = llm_client

    def run(
        self,
        vlm_json: Dict[str, Any],
        tools_compact: List[Dict[str, Any]] = None,
        user_instruction: str = "",
    ) -> Dict[str, Any]:
        """Run the Planner and return its output dict (or {} on parse failure)."""
        payload = {
            "vlm_output": vlm_json,
            "user_instruction": user_instruction,
            "available_tools": tools_compact or [],
        }
        raw = self.llm.generate_json(
            system=self.SYS,
            user=json.dumps(payload, ensure_ascii=False),
            max_new_tokens=512,
            temperature=0.1,
        )
        return _extract_first_json_obj(raw.get("text", "")) or {}


# ---------------------------------------------------------------------------
# CriticAgent
# ---------------------------------------------------------------------------

class CriticAgent:
    """Validates and optionally corrects the Planner's tool selection.

    Checks modality compatibility, target/tool mismatch, and tool availability.
    Always returns one of the three known tool IDs — never null.
    Returns (ToolDecision, raw_dict).
    """

    SYS = """You are the Critic in an agentic medical image segmentation system.

You receive a JSON object containing:
- "planner_output": the Planner's proposed plan including selected_tool, target, modality, and reasoning
- "vlm_output": original VLM analysis of the image
- "user_instruction": the raw text instruction from the user
- "available_tools": the list of available tools with their capabilities and notes

Your job is to critically review the Planner's decision and either CONFIRM or CORRECT it.

CRITICAL RULE: You MUST ALWAYS select one of the three tools (totalseg, voxtell, biomedparse). 
You are NEVER allowed to return null or skip tool selection. If unsure, confirm the Planner's choice.
The only time you set need_clarification is when the image modality and user instruction are physically contradictory (e.g., user asks for liver segmentation but image is clearly a brain scan).

=== HOW TO REVIEW ===

Step 1 - Check modality compatibility:
  Is the Planner's selected tool compatible with the detected modality?
  Example: totalseg only works on CT. If modality is MRI and Planner selected totalseg → this is WRONG, change to voxtell.

Step 2 - Check target compatibility:
  Does the target type match the tool's purpose?
  Example: If target contains tumor/lesion/cancer/infection/nodule/mass and Planner selected totalseg → this is WRONG, totalseg cannot segment pathology, change to voxtell.
  Example: If modality is X-Ray/Ultrasound/PET/Pathology and Planner did NOT select biomedparse → this is WRONG, change to biomedparse.

Step 3 - Check tool availability:
  Is the selected tool in available_tools? If not → set status to need_clarification.

Step 4 - Make final decision:
  If Planner is correct → confirm it (output same tool).
  If Planner has an error → correct it and explain why.
  If something is fundamentally unclear or missing → set need_clarification.

=== WHEN TO SET need_clarification ===

Only in these cases:
- The selected tool is not in available_tools
- Modality is completely undetectable and cannot be inferred at all
- The user instruction is not a segmentation request (e.g., "diagnose this" or "what is in this image")

Do NOT set need_clarification for:
- Low confidence modality detection (just proceed)
- Unusual anatomical targets (let the tool try)
- Minor ambiguity in phrasing

=== OUTPUT FORMAT ===

Return EXACTLY this JSON and nothing else. No markdown, no extra text:
{
  "tool": "<totalseg|voxtell|biomedparse>",   // MUST be one of these three, NEVER null
  "reason": "<clear explanation: did you confirm Planner or correct it, and why>",
  "required_inputs": ["nifti_path", "text_prompt"],
  "params": {},
  "questions": [],
  "status": "<ok|need_clarification>"
}
"""

    def __init__(self, llm_client):
        self.llm = llm_client

    def run(
        self,
        planner_output: Dict[str, Any],
        vlm_json: Dict[str, Any],
        tools_compact: List[Dict[str, Any]],
        user_instruction: str = "",
    ) -> Tuple[ToolDecision, Dict[str, Any]]:
        """Run the Critic and return (ToolDecision, raw_dict)."""
        payload = {
            "planner_output": planner_output,
            "vlm_output": vlm_json,
            "user_instruction": user_instruction,
            "available_tools": tools_compact,
        }
        raw = self.llm.generate_json(
            system=self.SYS,
            user=json.dumps(payload, ensure_ascii=False),
            max_new_tokens=384,
            temperature=0.1,
        )
        text = raw.get("text", "")
        parsed = _extract_first_json_obj(text) or {}
        td = decision_from_dict(parsed)
        return td, parsed


# ---------------------------------------------------------------------------
# ExplainerAgent
# ---------------------------------------------------------------------------

class ExplainerAgent:
    """Generates a plain-text clarification message for the user.

    When the Critic raises need_clarification, this agent writes a short
    natural-language explanation and — if the user has not yet decided to
    proceed — presents two numbered options (proceed / cancel).
    """

    SYS = """You are a medical imaging assistant explaining a situation to a user.

You receive information about what the system detected and what concern was raised.
Write plain text only. No JSON, no markdown, no bullet points.

IF user has NOT yet decided to proceed (user_override_requested = false):
- Write 2-3 sentences: what was understood, what concern was detected
- Write ONE question
- Write exactly two numbered options:
  1) Proceed anyway — I understand the limitation
  2) Cancel — I will provide corrected input

IF user HAS decided to proceed (user_override_requested = true):
- Write 1-2 sentences confirming the system will proceed and what will be attempted
- No questions, no options
"""

    def __init__(self, llm_client):
        self.llm = llm_client

    def run(
        self,
        planner_output: Dict[str, Any],
        critic_raw: Dict[str, Any],
        user_override_requested: bool = False,
    ) -> str:
        """Return a plain-text clarification string."""
        payload = {
            "planner_output": planner_output,
            "critic_questions": _safe_list(critic_raw.get("questions")),
            "user_override_requested": user_override_requested,
        }
        raw = self.llm.generate_text(
            system=self.SYS,
            user=json.dumps(payload, ensure_ascii=False),
            max_new_tokens=250,
            temperature=0.1,
        )
        return str(raw.get("text", "")).strip()


# ---------------------------------------------------------------------------
# ReporterAgent
# ---------------------------------------------------------------------------

class ReporterAgent:
    """Writes a concise final report after tool execution.

    Covers: what was segmented, which tool was used, whether it succeeded,
    any important caveats, and — on failure — actionable suggestions.
    Output is plain text, max 4 sentences, no internal system details.
    """

    SYS = """You are a medical imaging assistant writing a final report after segmentation.

Write plain text only. No markdown, no bullet points, no headers. Maximum 4 sentences.

Cover:
1. What was segmented, in which modality and anatomical region
2. Which tool was used and whether it succeeded
3. If succeeded: note important limitations
4. If failed: explain what went wrong plainly, suggest what user could try
5. If mask is empty/zeros: say no segmentation was found and result should not be used

Tone: professional, honest, concise. Do not mention JSON, agents, prompts, or internal system details.
"""

    def __init__(self, llm_client):
        self.llm = llm_client

    def run(self, planner_output: Dict[str, Any], tool_result: Dict[str, Any]) -> str:
        """Return a plain-text report string."""
        payload = {
            "planner_output": planner_output,
            "tool_result": tool_result,
        }
        raw = self.llm.generate_text(
            system=self.SYS,
            user=json.dumps(payload, ensure_ascii=False),
            max_new_tokens=250,
            temperature=0.1,
        )
        return str(raw.get("text", "")).strip()


# ---------------------------------------------------------------------------
# VerifierAgent  (deterministic — no LLM)
# ---------------------------------------------------------------------------

class VerifierAgent:
    """Deterministic post-execution check — no LLM involved.

    Verifies that:
      - The tool reported success (ok=True).
      - The mask path exists on disk.
      - If a directory: at least one .nii.gz file is present.
      - If a single file: nibabel can load it and the mask is non-zero.

    Returns a dict with keys: ok (bool), issues (list[str]), mask_path (str|None).
    """

    @staticmethod
    def run(tool_result: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """Check tool_result and return a verification summary dict."""
        import os
        mask_path = tool_result.get("mask_path")
        issues = []
        if not tool_result.get("ok"):
            issues.append(f"Tool reported failure: {tool_result.get('message')}")
        if mask_path:
            if not os.path.exists(mask_path):
                issues.append("Mask path does not exist.")
            elif os.path.isdir(mask_path):
                nii_files = [f for f in os.listdir(mask_path) if f.endswith(".nii.gz")]
                if not nii_files:
                    issues.append("Output directory is empty.")
            else:
                try:
                    import nibabel as nib
                    mask = nib.load(mask_path).get_fdata()
                    if mask.sum() == 0:
                        issues.append("Mask is all zeros — segmentation produced empty result.")
                except Exception as e:
                    issues.append(f"Could not load mask: {e}")
        else:
            issues.append("No mask path returned by tool.")
        return {"ok": len(issues) == 0, "issues": issues, "mask_path": mask_path}