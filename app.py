import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import tempfile
import os

# Optional dependencies — handled gracefully if not installed
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI-Assisted MEP Electrical Inspector",
    page_icon="⚡",
    layout="centered"
)

st.title("AI-Assisted Electrical Installation Inspector")
st.markdown(
    "Upload a site photograph or video walkthrough to receive a "
    "preliminary compliance assessment under SS 638: Code of Practice "
    "for Electrical Installations."
)

st.warning(
    "This tool is a preliminary screening aid only. It does not constitute "
    "a formal electrical inspection report and carries no legal or regulatory "
    "standing under EMA requirements.",
    icon="⚠️"
)

# ---------------------------------------------------------------------------
# Sidebar — API key and category selection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key. It is not stored after the session ends."
    )

    category = st.selectbox(
        "Inspection Category",
        options=[
            "Cable Tray Condition and Fill Level",
            "Distribution Panel Condition",
            "Cable Support, Identification and Loose Cables"
        ],
        help="Select the inspection category that matches your site photograph."
    )

    input_mode = st.radio(
        "Input Type",
        options=["Single Photograph", "Video Walkthrough"],
        help="Choose whether you are uploading a photo or a recorded site video."
    )

    if input_mode == "Video Walkthrough":
        frame_interval = st.slider(
            "Frame extraction interval (seconds)",
            min_value=1,
            max_value=30,
            value=5,
            help="One frame will be extracted and analysed for every N seconds of video."
        )
        max_frames = st.slider(
            "Maximum frames to analyse",
            min_value=1,
            max_value=20,
            value=5,
            help="Cap the number of frames sent to the API to control cost and time."
        )

# ---------------------------------------------------------------------------
# Inspection prompts — one per category, structured against SS 638 criteria
# ---------------------------------------------------------------------------
PROMPTS = {
    "Cable Tray Condition and Fill Level": """
You are an AI-assisted preliminary screening tool supporting electrical installation \
inspection in Singapore under SS 638: Code of Practice for Electrical Installations \
and EMA regulations.

Divide the image into four zones: Top-Left, Top-Right, Bottom-Left, Bottom-Right.
For each zone that contains a visible cable tray, evaluate the following criteria \
using only what is clearly visible. Do not guess if something is not visible.

Criterion 1 (SS 638 Clause 522 — Fill Level):
Cable tray fill level does not visually appear to exceed approximately 40% of the \
cross-sectional tray area.

Criterion 2 (SS 638 Clause 522 — Tray Condition):
Cable tray structure appears undamaged with no visible crushing, cracking, or deformation.

Criterion 3 (SS 638 Clause 543 — Earth Continuity Conductor):
Yellow and green earth continuity conductors are visible and appear to run continuously \
alongside power cables.

For each zone, respond in this exact format:
ZONE: [zone name]
CRITERION 1: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence explanation]
CRITERION 2: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence explanation]
CRITERION 3: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence explanation]
ZONE SEVERITY: [PASS / FLAG FOR REVIEW / FAIL / NOT APPLICABLE]

After all zones, provide:
OVERALL RESULT: [PASS / FLAG FOR REVIEW / FAIL]
SUMMARY: [Two to three sentences describing the most significant findings.]
""",

    "Distribution Panel Condition": """
You are an AI-assisted preliminary screening tool supporting electrical installation \
inspection in Singapore under SS 638: Code of Practice for Electrical Installations \
and EMA regulations.

Divide the image into four zones: Top-Left, Top-Right, Bottom-Left, Bottom-Right.
For each zone that contains a visible distribution panel or switchboard, evaluate \
the following criteria using only what is clearly visible. Do not guess if something \
is not visible.

Criterion 1 (SS 638 Clause 514 — Panel Condition):
Panel doors are present, closed, and not visibly damaged. No exposed live parts are visible.

Criterion 2 (SS 638 Clause 514 — Labelling):
Panel is visibly labelled. Circuit breakers and protection devices appear to carry \
legible identification markings.

Criterion 3 (SS 638 Clause 513 — Working Clearance):
There appears to be clear working space in front of the panel with no obstructions \
directly blocking access.

For each zone, respond in this exact format:
ZONE: [zone name]
CRITERION 1: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence explanation]
CRITERION 2: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence explanation]
CRITERION 3: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence explanation]
ZONE SEVERITY: [PASS / FLAG FOR REVIEW / FAIL / NOT APPLICABLE]

After all zones, provide:
OVERALL RESULT: [PASS / FLAG FOR REVIEW / FAIL]
SUMMARY: [Two to three sentences describing the most significant findings.]
""",

    "Cable Support, Identification and Loose Cables": """
You are an AI-assisted preliminary screening tool supporting electrical installation \
inspection in Singapore under SS 638: Code of Practice for Electrical Installations \
and EMA regulations.

Divide the image into four zones: Top-Left, Top-Right, Bottom-Left, Bottom-Right.
For each zone, evaluate the following criteria using only what is clearly visible. \
Do not guess if something is not visible.

Criterion 1 (SS 638 Clause 522 — Cable Support):
Cables are secured to supports at visible intervals with no unsupported spans or \
hanging loops evident.

Criterion 2 (SS 638 Clause 514 — Cable Identification):
Cables appear to be labelled or colour-coded in a consistent and identifiable manner.

Criterion 3 (SS 638 Clause 522 — Loose or Stray Cables):
No unsupported cables are visible lying on floors, above ceiling spaces, or creating \
potential trip hazards.

For each zone, respond in this exact format:
ZONE: [zone name]
CRITERION 1: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence explanation]
CRITERION 2: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence explanation]
CRITERION 3: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence explanation]
ZONE SEVERITY: [PASS / FLAG FOR REVIEW / FAIL / NOT APPLICABLE]

After all zones, provide:
OVERALL RESULT: [PASS / FLAG FOR REVIEW / FAIL]
SUMMARY: [Two to three sentences describing the most significant findings.]
"""
}

# ---------------------------------------------------------------------------
# Helper: encode image to base64
# ---------------------------------------------------------------------------
def encode_image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# ---------------------------------------------------------------------------
# Helper: call GPT-5.5 vision API
# ---------------------------------------------------------------------------
def call_gpt_vision(api_key: str, image_b64: str, prompt: str) -> str:
    if not OPENAI_AVAILABLE:
        return "ERROR: openai package is not installed. Run: pip install openai"

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-5.5-2026-04-23",   # <-- UPDATE THIS to latest model string
        max_completion_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    return response.choices[0].message.content

# ---------------------------------------------------------------------------
# Helper: parse zone findings from API response text
# ---------------------------------------------------------------------------
def parse_zone_findings(response_text: str) -> dict:
    """
    Extracts per-zone severity labels from the structured API response.
    Returns a dict: {"Top-Left": "PASS", "Top-Right": "FLAG FOR REVIEW", ...}
    """
    zones = {}
    severity_map = {
        "PASS": "PASS",
        "FLAG FOR REVIEW": "FLAG FOR REVIEW",
        "FAIL": "FAIL",
        "NOT APPLICABLE": "NOT APPLICABLE"
    }
    current_zone = None
    for line in response_text.splitlines():
        line = line.strip()
        if line.upper().startswith("ZONE:"):
            current_zone = line.split(":", 1)[1].strip()
        elif line.upper().startswith("ZONE SEVERITY:") and current_zone:
            severity_raw = line.split(":", 1)[1].strip().upper()
            for key in severity_map:
                if key in severity_raw:
                    zones[current_zone] = severity_map[key]
                    break
    return zones

# ---------------------------------------------------------------------------
# Helper: extract overall result
# ---------------------------------------------------------------------------
def extract_overall_result(response_text: str) -> str:
    for line in response_text.splitlines():
        if line.strip().upper().startswith("OVERALL RESULT:"):
            value = line.split(":", 1)[1].strip().upper()
            if "FAIL" in value:
                return "FAIL"
            elif "FLAG" in value:
                return "FLAG FOR REVIEW"
            elif "PASS" in value:
                return "PASS"
    return "CANNOT DETERMINE"

# ---------------------------------------------------------------------------
# Helper: draw zone annotation overlay on image
# ---------------------------------------------------------------------------
def annotate_image(image: Image.Image, zone_findings: dict) -> Image.Image:
    colour_map = {
        "PASS": (0, 180, 0, 80),           # green, semi-transparent
        "FLAG FOR REVIEW": (255, 165, 0, 80),  # amber
        "FAIL": (200, 0, 0, 80),           # red
        "NOT APPLICABLE": (150, 150, 150, 40)  # grey
    }
    label_colour = {
        "PASS": (0, 120, 0),
        "FLAG FOR REVIEW": (180, 100, 0),
        "FAIL": (160, 0, 0),
        "NOT APPLICABLE": (100, 100, 100)
    }

    w, h = image.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    zone_coords = {
        "Top-Left":     (0,     0,     w//2,  h//2),
        "Top-Right":    (w//2,  0,     w,     h//2),
        "Bottom-Left":  (0,     h//2,  w//2,  h),
        "Bottom-Right": (w//2,  h//2,  w,     h)
    }

    for zone_name, severity in zone_findings.items():
        coords = zone_coords.get(zone_name)
        if not coords:
            continue
        x0, y0, x1, y1 = coords
        fill = colour_map.get(severity, (150, 150, 150, 40))
        draw.rectangle([x0, y0, x1, y1], fill=fill, outline=fill[:3] + (220,), width=3)

        # Label text in zone centre
        label = severity
        tx = (x0 + x1) // 2
        ty = (y0 + y1) // 2
        tc = label_colour.get(severity, (80, 80, 80))
        draw.text((tx - 2, ty - 2), label, fill=(255, 255, 255), anchor="mm")
        draw.text((tx, ty), label, fill=tc, anchor="mm")

    base = image.convert("RGBA")
    combined = Image.alpha_composite(base, overlay)
    return combined.convert("RGB")

# ---------------------------------------------------------------------------
# Helper: extract frames from video
# ---------------------------------------------------------------------------
def extract_frames(video_path: str, interval_seconds: int, max_frames: int) -> list:
    if not CV2_AVAILABLE:
        return []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25
    frame_step = int(fps * interval_seconds)
    frames = []
    frame_index = 0
    while len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        timestamp = frame_index / fps
        frames.append((pil_image, timestamp))
        frame_index += frame_step
    cap.release()
    return frames

# ---------------------------------------------------------------------------
# Main application logic
# ---------------------------------------------------------------------------
prompt_text = PROMPTS[category]

if input_mode == "Single Photograph":
    uploaded_file = st.file_uploader(
        "Upload site photograph",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear site photograph of the electrical installation."
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded photograph", use_container_width=True)

        if st.button("Run Inspection", type="primary"):
            if not api_key:
                st.error("Please enter your OpenAI API key in the sidebar.")
            else:
                with st.spinner("Sending to GPT-5.5 for analysis..."):
                    try:
                        image_b64 = encode_image_to_base64(image)
                        response_text = call_gpt_vision(api_key, image_b64, prompt_text)

                        zone_findings = parse_zone_findings(response_text)
                        overall = extract_overall_result(response_text)

                        # Annotated image
                        if zone_findings:
                            annotated = annotate_image(image, zone_findings)
                            st.image(
                                annotated,
                                caption="Annotated output — colour-coded by zone",
                                use_container_width=True
                            )
                        else:
                            st.image(image, caption="No zone data parsed", use_container_width=True)

                        # Overall result banner
                        if overall == "PASS":
                            st.success(f"Overall Result: {overall}")
                        elif overall == "FLAG FOR REVIEW":
                            st.warning(f"Overall Result: {overall}")
                        elif overall == "FAIL":
                            st.error(f"Overall Result: {overall}")
                        else:
                            st.info(f"Overall Result: {overall}")

                        # Full written report
                        with st.expander("Full Assessment Report", expanded=True):
                            st.text(response_text)

                    except Exception as e:
                        st.error(f"API request failed: {e}")

elif input_mode == "Video Walkthrough":
    if not CV2_AVAILABLE:
        st.error(
            "OpenCV is required for video mode. "
            "Install it with: pip install opencv-python"
        )
    else:
        uploaded_video = st.file_uploader(
            "Upload video walkthrough",
            type=["mp4", "mov", "avi"],
            help="Upload a recorded site walkthrough video."
        )

        if uploaded_video:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(uploaded_video.name)[1]
            ) as tmp:
                tmp.write(uploaded_video.read())
                tmp_path = tmp.name

            st.success(f"Video uploaded: {uploaded_video.name}")

            if st.button("Extract Frames and Run Inspection", type="primary"):
                if not api_key:
                    st.error("Please enter your OpenAI API key in the sidebar.")
                else:
                    frames = extract_frames(tmp_path, frame_interval, max_frames)

                    if not frames:
                        st.error("No frames could be extracted from the video.")
                    else:
                        st.info(f"{len(frames)} frame(s) extracted. Sending to GPT-5.5...")
                        progress = st.progress(0)

                        results = []
                        for i, (frame_image, timestamp) in enumerate(frames):
                            with st.spinner(f"Analysing frame {i+1} of {len(frames)} "
                                            f"(at {timestamp:.1f}s)..."):
                                try:
                                    image_b64 = encode_image_to_base64(frame_image)
                                    response_text = call_gpt_vision(
                                        api_key, image_b64, prompt_text
                                    )
                                    zone_findings = parse_zone_findings(response_text)
                                    overall = extract_overall_result(response_text)
                                    results.append({
                                        "frame": i + 1,
                                        "timestamp": timestamp,
                                        "image": frame_image,
                                        "zone_findings": zone_findings,
                                        "overall": overall,
                                        "report": response_text
                                    })
                                except Exception as e:
                                    results.append({
                                        "frame": i + 1,
                                        "timestamp": timestamp,
                                        "image": frame_image,
                                        "zone_findings": {},
                                        "overall": "ERROR",
                                        "report": str(e)
                                    })
                            progress.progress((i + 1) / len(frames))

                        # Session summary
                        st.subheader("Session Summary")
                        total = len(results)
                        fails = sum(1 for r in results if r["overall"] == "FAIL")
                        flags = sum(1 for r in results if r["overall"] == "FLAG FOR REVIEW")
                        passes = sum(1 for r in results if r["overall"] == "PASS")

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Frames Analysed", total)
                        col2.metric("Pass", passes)
                        col3.metric("Flag for Review", flags)
                        col4.metric("Fail", fails)

                        # Worst-case overall
                        if fails > 0:
                            st.error("Worst-case result across session: FAIL")
                        elif flags > 0:
                            st.warning("Worst-case result across session: FLAG FOR REVIEW")
                        else:
                            st.success("Worst-case result across session: PASS")

                        # Per-frame expandable results
                        st.subheader("Frame-by-Frame Results")
                        for r in results:
                            label = (
                                f"Frame {r['frame']} | "
                                f"t={r['timestamp']:.1f}s | "
                                f"Result: {r['overall']}"
                            )
                            with st.expander(label):
                                if r["zone_findings"]:
                                    annotated = annotate_image(
                                        r["image"], r["zone_findings"]
                                    )
                                    st.image(
                                        annotated,
                                        caption="Annotated frame",
                                        use_container_width=True
                                    )
                                else:
                                    st.image(r["image"], use_container_width=True)
                                st.text(r["report"])

                    os.unlink(tmp_path)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "AI-Assisted MEP Installation Inspector | "
    "Preliminary screening tool only. "
    "This output does not constitute a formal electrical inspection report "
    "and carries no legal or regulatory standing under EMA requirements."
)
