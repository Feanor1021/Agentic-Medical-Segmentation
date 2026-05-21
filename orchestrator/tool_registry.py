# =============================================================================
# orchestrator/tool_registry.py
#
# Central registry of all segmentation tool services available to the
# orchestrator pipeline.  The Planner and Critic agents read this registry
# at runtime to decide which tool best fits the incoming request.
#
# Structure
# ---------
#   TOTALSEG_STRUCTURES  — exhaustive list of anatomical labels supported by
#                          TotalSegmentator (used for fuzzy target matching).
#   list_tools()         — returns the full registry as a list of dicts.
#                          Each entry includes id, modalities, supported targets,
#                          API port, and detailed decision-logic notes for the LLM.
#   tool_summary_for_llm() — strips each entry down to only the fields the LLM
#                            needs, keeping the prompt concise.
#   get_totalseg_structures() — convenience accessor for the structure list.
# =============================================================================

from __future__ import annotations
from typing import Any, Dict, List


# Full list of anatomical structures recognised by TotalSegmentator.
# Used by app.py's fuzzy keyword matcher to map free-text targets to exact names.
TOTALSEG_STRUCTURES = [
    'adrenal_gland_left', 'adrenal_gland_right', 'aorta', 'atrial_appendage_left',
    'autochthon_left', 'autochthon_right', 'brachiocephalic_trunk', 'brachiocephalic_vein_left',
    'brachiocephalic_vein_right', 'brain', 'clavicula_left', 'clavicula_right', 'colon',
    'common_carotid_artery_left', 'common_carotid_artery_right', 'costal_cartilages',
    'duodenum', 'esophagus', 'femur_left', 'femur_right', 'gallbladder',
    'gluteus_maximus_left', 'gluteus_maximus_right', 'gluteus_medius_left', 'gluteus_medius_right',
    'gluteus_minimus_left', 'gluteus_minimus_right', 'heart', 'hip_left', 'hip_right',
    'humerus_left', 'humerus_right', 'iliac_artery_left', 'iliac_artery_right',
    'iliac_vena_left', 'iliac_vena_right', 'iliopsoas_left', 'iliopsoas_right',
    'inferior_vena_cava', 'kidney_cyst_left', 'kidney_cyst_right', 'kidney_left', 'kidney_right',
    'liver', 'lung_lower_lobe_left', 'lung_lower_lobe_right', 'lung_middle_lobe_right',
    'lung_upper_lobe_left', 'lung_upper_lobe_right', 'pancreas', 'portal_vein_and_splenic_vein',
    'prostate', 'pulmonary_vein', 'rib_left_1', 'rib_left_2', 'rib_left_3', 'rib_left_4',
    'rib_left_5', 'rib_left_6', 'rib_left_7', 'rib_left_8', 'rib_left_9', 'rib_left_10',
    'rib_left_11', 'rib_left_12', 'rib_right_1', 'rib_right_2', 'rib_right_3', 'rib_right_4',
    'rib_right_5', 'rib_right_6', 'rib_right_7', 'rib_right_8', 'rib_right_9', 'rib_right_10',
    'rib_right_11', 'rib_right_12', 'sacrum', 'scapula_left', 'scapula_right', 'skull',
    'small_bowel', 'spinal_cord', 'spleen', 'sternum', 'stomach', 'subclavian_artery_left',
    'subclavian_artery_right', 'superior_vena_cava', 'thyroid_gland', 'trachea',
    'urinary_bladder', 'vertebrae_C1', 'vertebrae_C2', 'vertebrae_C3', 'vertebrae_C4',
    'vertebrae_C5', 'vertebrae_C6', 'vertebrae_C7', 'vertebrae_L1', 'vertebrae_L2',
    'vertebrae_L3', 'vertebrae_L4', 'vertebrae_L5', 'vertebrae_S1', 'vertebrae_T1',
    'vertebrae_T2', 'vertebrae_T3', 'vertebrae_T4', 'vertebrae_T5', 'vertebrae_T6',
    'vertebrae_T7', 'vertebrae_T8', 'vertebrae_T9', 'vertebrae_T10', 'vertebrae_T11',
    'vertebrae_T12',
]


def list_tools() -> List[Dict[str, Any]]:
    """Return the full tool registry.

    Each entry contains:
      id, name, task_types, modalities, supports_3d, targets,
      supported_structures, api_port, notes (LLM decision-logic hints).
    """
    return [
        {
            "id": "totalseg",
            "name": "TotalSegmentator",
            "task_types": ["segmentation"],
            "modalities": ["CT"],
            "supports_3d": True,
            "targets": ["anatomy_organs", "anatomy_structures", "bones", "muscles", "vessels"],
            "supported_structures": TOTALSEG_STRUCTURES,
            "api_port": 8011,
            "notes": [
                "PRIMARY ROLE: The gold standard for whole-organ anatomical segmentation in CT scans.",
                "MANDATORY USE CASE: Use ALWAYS and ONLY for healthy, major organs (liver, spleen, kidneys, pancreas, heart, lungs) in CT.",
                "STRENGTH: Uses fixed anatomical priors; it knows exactly where an organ should be. High Dice scores for standard anatomy.",
                "HARD LIMITATION: It is pathology-blind. It will NOT see or segment tumors, lesions, or nodules. If the user asks for a 'tumor', TotalSeg is the WRONG choice.",
                "MODALITY RESTRICTION: Strictly CT. If the image is MRI or anything else, choosing TotalSeg is a critical system error.",
                "DECISION LOGIC: If (modality == CT) AND (target is in supported_structures) AND (no 'tumor/lesion' in prompt) -> TotalSeg is the EXCLUSIVE choice.",
            ],
        },
        {
            "id": "voxtell",
            "name": "VoxTell",
            "task_types": ["segmentation"],
            "modalities": ["CT", "MRI"],
            "supports_3d": True,
            "targets": ["instruction_based_3d", "lesion_like", "tumor", "pathology", "text_prompt_any_structure"],
            "supported_structures": [],
            "api_port": 8012,
            "notes": [
                "PRIMARY ROLE: Specialized for irregular, unpredictable structures like tumors, lesions, and metastases.",
                "MANDATORY USE CASE: Use for any 3D segmentation task in CT or MRI that involves pathology (cancer, mass, nodule, lesion).",
                "STRENGTH: Flexible shape handling. Unlike TotalSeg, it doesn't expect a 'standard' organ shape; it follows the visual evidence of the lesion.",
                "HARD LIMITATION: Inefficient for whole-organ segmentation in CT. For a simple 'Segment liver' request in CT, it is inferior to TotalSeg.",
                "TEXT-DRIVEN: It is highly responsive to descriptive text prompts for 3D volumes.",
                "DECISION LOGIC: If (target involves 'tumor' or 'lesion') OR (modality == MRI and target is 3D structure) -> VoxTell is the primary candidate.",
            ],
        },
        {
            "id": "biomedparse",
            "name": "BiomedParse",
            "task_types": ["segmentation"],
            "modalities": ["CT", "MRI", "X-Ray", "Ultrasound", "PET", "Pathology"],
            "supports_3d": False,
            "targets": ["text_prompt_any_structure", "lesion", "organ", "tumor", "pathology", "2d_slice"],
            "supported_structures": [],
            "api_port": 8013,
            "notes": [
                "PRIMARY ROLE: The universal adapter for non-standard modalities and 2D-centric tasks.",
                "MANDATORY USE CASE: The EXCLUSIVE choice for X-Ray, Ultrasound, PET, Pathology slides, Dermoscopy, and Endoscopy.",
                "STRENGTH: Can segment almost anything from a text prompt, but operates primarily on a slice-by-slice or 2D basis.",
                "CRITICAL WARNING: Never use for CT organ segmentation if TotalSeg is available. Never use for 3D CT/MRI lesion segmentation if VoxTell is available.",
                "LAST RESORT: Only use for CT/MRI if the target structure is extremely niche and not supported by other 3D-native tools.",
                "DECISION LOGIC: If (modality is NOT CT/MRI) -> BiomedParse is the ONLY choice. If (target is a 2D specific slice) -> BiomedParse is the best choice.",
            ],
        },
    ]


def tool_summary_for_llm(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a condensed view of the registry for inclusion in LLM prompts.

    Keeps only the fields the agents need (id, task_types, modalities,
    supports_3d, targets, notes) so the prompt stays concise.
    """
    return [
        {
            "id": t["id"],
            "task_types": t.get("task_types", []),
            "modalities": t.get("modalities", []),
            "supports_3d": bool(t.get("supports_3d", False)),
            "targets": t.get("targets", []),
            "notes": t.get("notes", []),
        }
        for t in tools
    ]


def get_totalseg_structures() -> List[str]:
    """Return the full list of TotalSegmentator-supported structure names."""
    return TOTALSEG_STRUCTURES