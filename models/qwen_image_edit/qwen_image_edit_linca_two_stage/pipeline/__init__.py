# Pipeline for qwen_image_edit_linca_two_stage
# QwenImageEditPipeline loaded from freqca_qwen (LinCA base pipeline)
import importlib.util
import sys
from pathlib import Path
# pipeline -> qwen_image_edit_linca_two_stage -> qwen_edit -> Linca -> LinCA
_pipeline_base = Path(__file__).resolve().parent.parent.parent.parent.parent / "freqca_qwen"
_spec = importlib.util.spec_from_file_location(
    "pipeline_qwenimage_edit",
    _pipeline_base / "pipeline" / "pipeline_qwenimage_edit.py",
)
_mod = importlib.util.module_from_spec(_spec)
_orig_path = sys.path.copy()
sys.path.insert(0, str(_pipeline_base))
_spec.loader.exec_module(_mod)
sys.path[:] = _orig_path
QwenImageEditPipeline = _mod.QwenImageEditPipeline
