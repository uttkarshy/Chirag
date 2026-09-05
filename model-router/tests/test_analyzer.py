import pytest
from app.analyzer import analyze_execution_plan
from app.schema import ExecutionPlan, ModelCapability, Modality, ModelRequirements, PlanStep


def test_text_generation():
    plan = ExecutionPlan(
        plan_id="plan_text",
        goal="Summarize annual report",
        desired_output="Executive summary report in markdown",
        steps=[
            PlanStep(
                id="step_1",
                title="Generate Executive summary report",
                step_type="generation",
                expected_output="Executive summary report in markdown",
            )
        ],
    )
    req = analyze_execution_plan(plan)
    assert req.capabilities == [ModelCapability.TEXT_GENERATION.value]
    assert req.input_modalities == [Modality.TEXT.value]
    assert req.output_modalities == [Modality.TEXT.value]


def test_vision():
    plan = ExecutionPlan(
        plan_id="plan_vision",
        goal="Explain architecture diagram",
        desired_output="Text explanation of system diagram",
        steps=[
            PlanStep(
                id="step_1",
                title="Inspect source: architecture_diagram.png",
                step_type="inspect_source",
                inputs=["architecture_diagram.png"],
                expected_output="Extracted diagram elements",
            ),
            PlanStep(
                id="step_2",
                title="Generate architecture explanation",
                step_type="generation",
                inputs=["step_1"],
                depends_on=["step_1"],
                expected_output="Text explanation of system diagram",
            ),
        ],
    )
    req = analyze_execution_plan(plan)
    assert ModelCapability.VISION.value in req.capabilities
    assert ModelCapability.TEXT_GENERATION.value in req.capabilities
    assert Modality.IMAGE.value in req.input_modalities
    assert Modality.TEXT.value in req.input_modalities
    assert req.output_modalities == [Modality.TEXT.value]


def test_image_generation():
    plan = ExecutionPlan(
        plan_id="plan_img_gen",
        goal="Create brand logo",
        desired_output="Modern minimalist tech logo image in PNG",
        steps=[
            PlanStep(
                id="step_1",
                title="Generate tech logo image",
                step_type="generation",
                expected_output="Modern minimalist tech logo image in PNG format",
            )
        ],
    )
    req = analyze_execution_plan(plan)
    assert req.capabilities == [ModelCapability.IMAGE_GENERATION.value]
    assert req.input_modalities == [Modality.TEXT.value]
    assert req.output_modalities == [Modality.IMAGE.value]


def test_video_generation():
    plan = ExecutionPlan(
        plan_id="plan_vid_gen",
        goal="Generate promotional teaser",
        desired_output="30-second promotional teaser animation MP4",
        steps=[
            PlanStep(
                id="step_1",
                title="Generate promo animation",
                step_type="generation",
                expected_output="30-second promotional teaser animation video clip",
            )
        ],
    )
    req = analyze_execution_plan(plan)
    assert req.capabilities == [ModelCapability.VIDEO_GENERATION.value]
    assert req.input_modalities == [Modality.TEXT.value]
    assert req.output_modalities == [Modality.VIDEO.value]


def test_speech_to_text():
    plan = ExecutionPlan(
        plan_id="plan_stt",
        goal="Transcribe interview",
        desired_output="Full text interview transcription",
        steps=[
            PlanStep(
                id="step_1",
                title="Inspect source: interview_recording.mp3",
                step_type="inspect_source",
                inputs=["interview_recording.mp3"],
                expected_output="Extracted audio stream",
            ),
            PlanStep(
                id="step_2",
                title="Generate transcription",
                step_type="generation",
                inputs=["step_1"],
                depends_on=["step_1"],
                expected_output="Full text interview transcription",
            ),
        ],
    )
    req = analyze_execution_plan(plan)
    assert ModelCapability.SPEECH_TO_TEXT.value in req.capabilities
    assert Modality.AUDIO.value in req.input_modalities
    assert Modality.TEXT.value in req.output_modalities


def test_text_to_speech():
    plan = ExecutionPlan(
        plan_id="plan_tts",
        goal="Create audiobook chapter",
        desired_output="Spoken voice narration audio file in WAV",
        steps=[
            PlanStep(
                id="step_1",
                title="Synthesize narration voice",
                step_type="generation",
                expected_output="Spoken voice narration audio recording in WAV format",
            )
        ],
    )
    req = analyze_execution_plan(plan)
    assert req.capabilities == [ModelCapability.TEXT_TO_SPEECH.value]
    assert req.input_modalities == [Modality.TEXT.value]
    assert req.output_modalities == [Modality.AUDIO.value]


def test_code_generation():
    plan = ExecutionPlan(
        plan_id="plan_code",
        goal="Build authentication middleware",
        desired_output="FastAPI JWT authentication middleware in Python",
        steps=[
            PlanStep(
                id="step_1",
                title="Generate authentication code",
                step_type="generation",
                requirements=["Python 3.12", "FastAPI"],
                expected_output="FastAPI JWT authentication middleware in Python script",
            )
        ],
    )
    req = analyze_execution_plan(plan)
    assert req.capabilities == [ModelCapability.CODE_GENERATION.value]
    assert req.input_modalities == [Modality.TEXT.value]
    assert req.output_modalities == [Modality.TEXT.value]


def test_multiple_capabilities_vision_and_code():
    plan = ExecutionPlan(
        plan_id="plan_multi",
        goal="Convert mockup to code",
        desired_output="React component implementing UI design",
        steps=[
            PlanStep(
                id="step_1",
                title="Inspect source: dashboard_mockup.png",
                step_type="inspect_source",
                inputs=["dashboard_mockup.png"],
                expected_output="Extracted UI visual components",
            ),
            PlanStep(
                id="step_2",
                title="Generate React component",
                step_type="generation",
                inputs=["step_1"],
                depends_on=["step_1"],
                requirements=["React", "TypeScript", "TailwindCSS"],
                expected_output="React component code implementation",
            ),
        ],
    )
    req = analyze_execution_plan(plan)
    assert req.capabilities == [ModelCapability.CODE_GENERATION.value, ModelCapability.VISION.value]
    assert req.input_modalities == [Modality.IMAGE.value, Modality.TEXT.value]
    assert req.output_modalities == [Modality.TEXT.value]


def test_deterministic_ordering():
    # Pass capabilities in varying order to ModelRequirements and verify sorting
    req1 = ModelRequirements(
        capabilities=["vision", "code_generation", "text_generation"],
        input_modalities=["image", "text"],
        output_modalities=["text"],
    )
    req2 = ModelRequirements(
        capabilities=["text_generation", "vision", "code_generation"],
        input_modalities=["text", "image"],
        output_modalities=["text"],
    )
    assert req1.capabilities == ["code_generation", "text_generation", "vision"]
    assert req2.capabilities == ["code_generation", "text_generation", "vision"]
    assert req1.input_modalities == ["image", "text"]
    assert req2.input_modalities == ["image", "text"]
    assert req1.capabilities == req2.capabilities
    assert req1.input_modalities == req2.input_modalities


def test_structured_output_detection():
    plan = ExecutionPlan(
        plan_id="plan_json",
        goal="Extract product data",
        desired_output="JSON schema validated product list",
        steps=[
            PlanStep(
                id="step_1",
                title="Generate JSON output",
                step_type="generation",
                requirements=["Adhere to JSON schema format"],
                expected_output="Structured JSON array of products",
            )
        ],
    )
    req = analyze_execution_plan(plan)
    assert req.structured_output is True


def test_streaming_requirement_detection():
    plan = ExecutionPlan(
        plan_id="plan_stream",
        goal="Interactive chat",
        desired_output="Live streaming assistant response",
        steps=[
            PlanStep(
                id="step_1",
                title="Generate streaming response",
                step_type="generation",
                requirements=["real-time streaming response"],
                expected_output="Streaming markdown tokens",
            )
        ],
    )
    req = analyze_execution_plan(plan)
    assert req.streaming is True


def test_plan_with_provider_mentions_does_not_inject_provider_or_fail():
    # User's goal/output mentions AWS, OpenAI, Claude, NVIDIA NIM, but analyzer remains provider-agnostic
    plan = ExecutionPlan(
        plan_id="plan_provider_ref",
        goal="Write a technical comparison of AWS Bedrock, OpenAI, and NVIDIA NIM",
        desired_output="Comparative evaluation report",
        steps=[
            PlanStep(
                id="step_1",
                title="Synthesize comparison of OpenAI and Claude",
                step_type="generation",
                requirements=["Neutral tone", "Benchmark latency on AWS"],
                expected_output="Comparative evaluation report in markdown",
            )
        ],
    )
    req = analyze_execution_plan(plan)
    assert req.capabilities == [ModelCapability.TEXT_GENERATION.value]
    # Verify no provider fields exist on requirements
    assert not hasattr(req, "provider")
    assert not hasattr(req, "model")
    assert "provider" not in req.model_dump()


def test_goal_with_camera_does_not_blindly_infer_vision():
    # Goal mentions 'camera' in natural language, but no image inputs or visual outputs exist
    plan = ExecutionPlan(
        plan_id="plan_camera_story",
        goal="Write a fictional mystery story about an old antique camera found in an attic",
        desired_output="A short story in markdown",
        steps=[
            PlanStep(
                id="step_1",
                title="Generate mystery story",
                step_type="generation",
                expected_output="A short story in markdown",
            )
        ],
    )
    req = analyze_execution_plan(plan)
    # Must NOT infer vision just because goal mentions 'camera'
    assert req.capabilities == [ModelCapability.TEXT_GENERATION.value]
    assert req.input_modalities == [Modality.TEXT.value]
    assert req.output_modalities == [Modality.TEXT.value]


def test_dictionary_plan_input():
    raw_plan = {
        "plan_id": "plan_dict",
        "goal": "Write poetry",
        "desired_output": "A haiku poem",
        "steps": [
            {
                "id": "step_1",
                "title": "Generate poem",
                "step_type": "generation",
                "inputs": [],
                "depends_on": [],
                "requirements": [],
                "expected_output": "A haiku poem",
            }
        ],
    }
    req = analyze_execution_plan(raw_plan)
    assert req.capabilities == [ModelCapability.TEXT_GENERATION.value]
    assert req.input_modalities == [Modality.TEXT.value]
    assert req.output_modalities == [Modality.TEXT.value]


def test_image_modality_detected_from_structured_metadata_without_file_extension():
    # Source has no filename extension (e.g. id="user_upload_123", no ".png")
    plan = ExecutionPlan(
        plan_id="plan_struct_img",
        goal="Process image attachment",
        desired_output="Extracted text summary",
        steps=[
            PlanStep(
                id="step_1",
                title="Inspect source: user_upload_123",
                step_type="inspect_source",
                inputs=["user_upload_123"],
                expected_output="Extracted information from user_upload_123",
                metadata={"source_type": "image"},
            ),
            PlanStep(
                id="step_2",
                title="Generate summary",
                step_type="generation",
                inputs=["step_1"],
                depends_on=["step_1"],
                expected_output="Extracted text summary",
            ),
        ],
    )
    req = analyze_execution_plan(plan)
    assert ModelCapability.VISION.value in req.capabilities
    assert Modality.IMAGE.value in req.input_modalities
    assert req.output_modalities == [Modality.TEXT.value]


def test_audio_modality_detected_from_structured_metadata_without_file_extension():
    # Source has no filename extension (e.g. id="audio_stream_raw", no ".mp3")
    plan = ExecutionPlan(
        plan_id="plan_struct_audio",
        goal="Transcribe stream",
        desired_output="Transcription text",
        steps=[
            PlanStep(
                id="step_1",
                title="Inspect source: audio_stream_raw",
                step_type="inspect_source",
                inputs=["audio_stream_raw"],
                expected_output="Raw audio data",
                metadata={"source_type": "audio"},
            ),
            PlanStep(
                id="step_2",
                title="Generate transcription",
                step_type="generation",
                inputs=["step_1"],
                depends_on=["step_1"],
                expected_output="Transcription text",
            ),
        ],
    )
    req = analyze_execution_plan(plan)
    assert ModelCapability.SPEECH_TO_TEXT.value in req.capabilities
    assert Modality.AUDIO.value in req.input_modalities
    assert req.output_modalities == [Modality.TEXT.value]


def test_video_modality_detected_from_structured_metadata_without_file_extension():
    # Source has no filename extension (e.g. id="surveillance_feed", no ".mp4")
    plan = ExecutionPlan(
        plan_id="plan_struct_video",
        goal="Analyze surveillance feed",
        desired_output="Activity report",
        steps=[
            PlanStep(
                id="step_1",
                title="Inspect source: surveillance_feed",
                step_type="inspect_source",
                inputs=["surveillance_feed"],
                expected_output="Visual frames",
                metadata={"source_type": "video"},
            ),
            PlanStep(
                id="step_2",
                title="Generate activity report",
                step_type="generation",
                inputs=["step_1"],
                depends_on=["step_1"],
                expected_output="Activity report in markdown",
            ),
        ],
    )
    req = analyze_execution_plan(plan)
    assert ModelCapability.VISION.value in req.capabilities
    assert Modality.VIDEO.value in req.input_modalities
    assert req.output_modalities == [Modality.TEXT.value]
