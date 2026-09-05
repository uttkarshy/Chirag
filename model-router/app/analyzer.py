import re
from typing import Any
from .schema import ExecutionPlan, Modality, ModelCapability, ModelRequirements
from .validation import validate_execution_plan_for_routing

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".svg", ".gif", ".bmp", ".tiff", ".ico"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma", ".opus"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv"}

CODE_KEYWORDS = {
    "code", "script", "program", "function", "implementation", "repository",
    "unit test", "test suite", "tests", "api", "openapi", "sdk", "library",
    "python", "javascript", "typescript", "html", "css", "sql", "bash", "shell",
    "rust", "golang", "go", "java", "c++", "c#", "react", "vue", "fastapi", "django"
}
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".sql", ".sh", ".bash",
    ".go", ".rs", ".java", ".cpp", ".c", ".cs", ".php", ".rb"
}

IMAGE_GEN_KEYWORDS = {
    "image", "diagram", "chart", "illustration", "drawing", "photo", "photograph",
    "graphic", "logo", "mockup", "banner", "poster", "infographic"
}

VIDEO_GEN_KEYWORDS = {
    "video", "animation", "motion graphic", "video clip", "movie"
}

AUDIO_GEN_KEYWORDS = {
    "audio", "speech", "voice", "narration", "spoken", "podcast", "text-to-speech", "tts"
}

AUDIO_INPUT_KEYWORDS = {
    "audio", "speech", "voice", "recording", "podcast", "dictation", "transcription", "transcribe"
}

VISUAL_INPUT_KEYWORDS = {
    "image", "screenshot", "photo", "photograph", "diagram", "picture", "frame"
}

TEXT_OUTPUT_KEYWORDS = {
    "text", "explanation", "analysis", "summary", "report",
    "documentation", "article", "notes", "markdown", "description",
    "guide", "review", "comparison", "specification"
}

STRUCTURED_KEYWORDS = {
    "json", "yaml", "xml", "csv", "schema", "table", "structured output", "structured format", "openapi"
}

STREAMING_KEYWORDS = {
    "streaming", "streamed", "stream", "real-time", "realtime"
}


def _matches_any(text: str, keywords: set[str], extensions: set[str] | None = None) -> bool:
    """Check if text contains any keyword as a distinct token or matches any extension."""
    lower = text.lower()
    if extensions:
        for ext in extensions:
            if ext in lower:
                return True
    for kw in keywords:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, lower):
            return True
    return False


def analyze_execution_plan(plan: ExecutionPlan | dict[str, Any]) -> ModelRequirements:
    """Analyze an ExecutionPlan deterministically and produce ModelRequirements."""
    if isinstance(plan, dict):
        validated_plan = ExecutionPlan.model_validate(plan)
    elif hasattr(plan, "model_dump"):
        validated_plan = ExecutionPlan.model_validate(plan.model_dump())
    else:
        validated_plan = plan

    validate_execution_plan_for_routing(validated_plan)

    capabilities: set[str] = set()
    input_modalities: set[str] = set()
    output_modalities: set[str] = set()
    structured_output: bool = False
    streaming: bool = False

    # 1. Inspect steps for inputs and step-specific requirements
    for step in validated_plan.steps:
        # A. Inspect structured PlanStep metadata FIRST
        source_type = (step.metadata.get("source_type") or "").strip().lower()
        if source_type:
            if source_type in ("image", "visual"):
                input_modalities.add(Modality.IMAGE.value)
                capabilities.add(ModelCapability.VISION.value)
            elif source_type == "audio":
                input_modalities.add(Modality.AUDIO.value)
                capabilities.add(ModelCapability.SPEECH_TO_TEXT.value)
            elif source_type == "video":
                input_modalities.add(Modality.VIDEO.value)
                capabilities.add(ModelCapability.VISION.value)
            elif source_type in ("text", "file", "repository", "url", "connected_source"):
                input_modalities.add(Modality.TEXT.value)

        # Check explicit metadata overrides if provided
        if "capability" in step.metadata:
            capabilities.add(step.metadata["capability"])
        if "input_modality" in step.metadata:
            input_modalities.add(step.metadata["input_modality"])
        if "output_modality" in step.metadata:
            output_modalities.add(step.metadata["output_modality"])

        # B. Inspect deterministic explicit fields (e.g. file extensions in inputs)
        for inp in step.inputs:
            lower_inp = inp.lower()
            if any(lower_inp.endswith(ext) for ext in IMAGE_EXTENSIONS):
                input_modalities.add(Modality.IMAGE.value)
                capabilities.add(ModelCapability.VISION.value)
            elif any(lower_inp.endswith(ext) for ext in VIDEO_EXTENSIONS):
                input_modalities.add(Modality.VIDEO.value)
                capabilities.add(ModelCapability.VISION.value)
            elif any(lower_inp.endswith(ext) for ext in AUDIO_EXTENSIONS):
                input_modalities.add(Modality.AUDIO.value)
                capabilities.add(ModelCapability.SPEECH_TO_TEXT.value)

        # C. Textual fallback for inspect_source steps
        if step.step_type == "inspect_source":
            combined_inspect = f"{' '.join(step.inputs)} {' '.join(step.requirements)} {step.title}".lower()
            if _matches_any(combined_inspect, VISUAL_INPUT_KEYWORDS, IMAGE_EXTENSIONS):
                input_modalities.add(Modality.IMAGE.value)
                capabilities.add(ModelCapability.VISION.value)
            if _matches_any(combined_inspect, {"video"}, VIDEO_EXTENSIONS):
                input_modalities.add(Modality.VIDEO.value)
                capabilities.add(ModelCapability.VISION.value)
            if _matches_any(combined_inspect, AUDIO_INPUT_KEYWORDS, AUDIO_EXTENSIONS):
                input_modalities.add(Modality.AUDIO.value)
                capabilities.add(ModelCapability.SPEECH_TO_TEXT.value)


        # Check structured output requirement
        step_reqs_text = " ".join(step.requirements)
        if _matches_any(f"{step.expected_output} {step_reqs_text}", STRUCTURED_KEYWORDS):
            structured_output = True

        # Check streaming requirement
        if _matches_any(step_reqs_text, STREAMING_KEYWORDS):
            streaming = True

    # 2. Inspect generation steps for capabilities and output modalities
    generation_steps = [s for s in validated_plan.steps if s.step_type == "generation"]

    for gen_step in generation_steps:
        gen_output_text = f"{gen_step.expected_output} {' '.join(gen_step.requirements)}"

        is_text_focused = _matches_any(gen_step.expected_output, TEXT_OUTPUT_KEYWORDS)
        is_code = _matches_any(gen_output_text, CODE_KEYWORDS, CODE_EXTENSIONS)
        is_video_gen = _matches_any(gen_output_text, VIDEO_GEN_KEYWORDS, VIDEO_EXTENSIONS) and not is_text_focused
        is_img_gen = _matches_any(gen_output_text, IMAGE_GEN_KEYWORDS, IMAGE_EXTENSIONS) and not is_text_focused
        is_tts = _matches_any(gen_output_text, AUDIO_GEN_KEYWORDS, AUDIO_EXTENSIONS) and not is_text_focused

        if is_video_gen:
            output_modalities.add(Modality.VIDEO.value)
            capabilities.add(ModelCapability.VIDEO_GENERATION.value)

        if is_img_gen:
            output_modalities.add(Modality.IMAGE.value)
            capabilities.add(ModelCapability.IMAGE_GENERATION.value)

        if is_tts:
            output_modalities.add(Modality.AUDIO.value)
            capabilities.add(ModelCapability.TEXT_TO_SPEECH.value)

        if is_code:
            output_modalities.add(Modality.TEXT.value)
            capabilities.add(ModelCapability.CODE_GENERATION.value)

        # Text generation: if produces text/explanation/analysis/code, or no non-text media was produced
        has_media_output = is_video_gen or is_img_gen or is_tts
        if not has_media_output and not is_code:
            output_modalities.add(Modality.TEXT.value)
            capabilities.add(ModelCapability.TEXT_GENERATION.value)
        elif is_text_focused:
            output_modalities.add(Modality.TEXT.value)
            capabilities.add(ModelCapability.TEXT_GENERATION.value)

    # If speech-to-text was required and no output modality set yet, output modality is text
    if ModelCapability.SPEECH_TO_TEXT.value in capabilities and not output_modalities:
        output_modalities.add(Modality.TEXT.value)

    # 3. Fallbacks and normalization
    if not input_modalities or any(s.step_type in ("generation", "reasoning", "verification") for s in validated_plan.steps):
        input_modalities.add(Modality.TEXT.value)

    if not output_modalities:
        output_modalities.add(Modality.TEXT.value)

    if not capabilities:
        capabilities.add(ModelCapability.TEXT_GENERATION.value)

    all_plan_text = f"{validated_plan.desired_output} {' '.join(validated_plan.verification_criteria)}"
    if _matches_any(all_plan_text, STRUCTURED_KEYWORDS):
        structured_output = True
    if _matches_any(all_plan_text, STREAMING_KEYWORDS):
        streaming = True

    return ModelRequirements(
        capabilities=sorted(list(capabilities)),
        input_modalities=sorted(list(input_modalities)),
        output_modalities=sorted(list(output_modalities)),
        minimum_context_tokens=None,
        structured_output=structured_output,
        streaming=streaming,
    )
