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
# Sidebar — API key and settings
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
# Colour scheme for severity levels
# ---------------------------------------------------------------------------
SEVERITY_FILL = {
    "PASS":            (0,   200,   0,  60),
    "FLAG FOR REVIEW": (255, 165,   0,  60),
    "FAIL":            (220,   0,   0,  60),
    "CANNOT DETERMINE":(150, 150, 150,  40),
}
SEVERITY_BORDER = {
    "PASS":            (0,   160,   0, 230),
    "FLAG FOR REVIEW": (210, 130,   0, 230),
    "FAIL":            (180,   0,   0, 230),
    "CANNOT DETERMINE":(120, 120, 120, 200),
}
SEVERITY_TEXT = {
    "PASS":            (0,   110,   0),
    "FLAG FOR REVIEW": (150,  90,   0),
    "FAIL":            (150,   0,   0),
    "CANNOT DETERMINE":( 80,  80,  80),
}

# ---------------------------------------------------------------------------
# Inspection prompts — element-based bounding box detection
#
# GPT-5.5 is asked to:
#   1. Identify each relevant element in the image
#   2. Return a normalised bounding box (left, top, right, bottom) as
#      fractions of image width/height, values between 0.0 and 1.0
#   3. Assess the relevant SS 638 criteria for that element
#   4. Return a severity verdict per element
#
# The Python code then draws those boxes directly onto the image using Pillow.
# ---------------------------------------------------------------------------

PROMPTS = {
    "Cable Tray Condition and Fill Level": """
You are an AI-assisted preliminary screening tool supporting electrical installation \
inspection in Singapore under SS 638: Code of Practice for Electrical Installations \
and EMA regulations.

Your task has two parts:

PART 1 — ELEMENT DETECTION AND ASSESSMENT
Identify every distinct cable tray, trunking run, or cable ladder visible in the image. \
For each one, provide a bounding box as normalised coordinates (fractions of image \
width and height, between 0.0 and 1.0) in the format: left, top, right, bottom.

Assess each detected element against these criteria using only what is clearly visible:
Criterion 1 (SS 638 Clause 522 — Fill Level): Cable tray fill level does not visually \
appear to exceed approximately 40% of the cross-sectional tray area.
Criterion 2 (SS 638 Clause 522 — Tray Condition): Cable tray structure appears undamaged \
with no visible crushing, cracking, or deformation.
Criterion 3 (SS 638 Clause 543 — Earth Continuity Conductor): Yellow and green earth \
continuity conductors are visible and appear to run continuously alongside power cables.

For each element respond in this EXACT format — do not deviate:
ELEMENT: Cable Tray [number]
BBOX: [left], [top], [right], [bottom]
CRITERION 1: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence]
CRITERION 2: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence]
CRITERION 3: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence]
ELEMENT SEVERITY: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE]

Also identify any loose cables visible on the floor and report them as:
ELEMENT: Floor Cables
BBOX: [left], [top], [right], [bottom]
CRITERION 1: FAIL — Loose cables visible on floor
ELEMENT SEVERITY: FAIL

If no loose floor cables are visible, skip this element entirely.

PART 2 — OVERALL SUMMARY
After all elements, provide:
OVERALL RESULT: [PASS / FLAG FOR REVIEW / FAIL]
SUMMARY: [Two to three sentences on the most significant findings.]
""",

    "Distribution Panel Condition": """
You are an AI-assisted preliminary screening tool supporting electrical installation \
inspection in Singapore under SS 638: Code of Practice for Electrical Installations \
and EMA regulations.

Your task has two parts:

PART 1 — ELEMENT DETECTION AND ASSESSMENT
Identify every distinct distribution panel, switchboard, or electrical enclosure visible \
in the image. For each one, provide a bounding box as normalised coordinates (fractions \
of image width and height, between 0.0 and 1.0) in the format: left, top, right, bottom.

Assess each detected element against these criteria using only what is clearly visible:
Criterion 1 (SS 638 Clause 514 — Panel Condition): Panel doors are present, closed, \
and not visibly damaged. No exposed live parts are visible.
Criterion 2 (SS 638 Clause 514 — Labelling): Panel is visibly labelled. Warning signs \
and identification markings are visible on panel faces.
Criterion 3 (SS 638 Clause 514 — Panel Integrity): No visible signs of overheating, \
burn marks, corrosion, or physical damage on panel surfaces.

For each element respond in this EXACT format — do not deviate:
ELEMENT: Distribution Panel [number]
BBOX: [left], [top], [right], [bottom]
CRITERION 1: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence]
CRITERION 2: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence]
CRITERION 3: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence]
ELEMENT SEVERITY: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE]

Also identify the floor area in front of panels and assess working clearance:
ELEMENT: Floor / Working Clearance
BBOX: [left], [top], [right], [bottom]
CRITERION 1 (SS 638 Clause 513 — Working Clearance): [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence]
ELEMENT SEVERITY: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE]

PART 2 — OVERALL SUMMARY
After all elements, provide:
OVERALL RESULT: [PASS / FLAG FOR REVIEW / FAIL]
SUMMARY: [Two to three sentences on the most significant findings.]
""",

    "Cable Support, Identification and Loose Cables": """
You are an AI-assisted preliminary screening tool supporting electrical installation \
inspection in Singapore under SS 638: Code of Practice for Electrical Installations \
and EMA regulations.

Your task has two parts:

PART 1 — ELEMENT DETECTION AND ASSESSMENT
Identify every distinct cable run, cable tray, or group of cables visible in the image. \
For each one, provide a bounding box as normalised coordinates (fractions of image \
width and height, between 0.0 and 1.0) in the format: left, top, right, bottom.

Assess each detected element against these criteria using only what is clearly visible:
Criterion 1 (SS 638 Clause 522 — Cable Support): Cables appear secured to supports \
at visible intervals with no unsupported hanging loops evident.
Criterion 2 (SS 638 Clause 514 — Cable Identification): Cables appear colour-coded \
or labelled in a consistent and identifiable manner.

For each element respond in this EXACT format — do not deviate:
ELEMENT: Cable Run [number]
BBOX: [left], [top], [right], [bottom]
CRITERION 1: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence]
CRITERION 2: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence]
ELEMENT SEVERITY: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE]

Also identify any loose cables visible on the floor:
ELEMENT: Floor Cables
BBOX: [left], [top], [right], [bottom]
CRITERION 1 (SS 638 Clause 522 — Loose Cables): FAIL — Unsupported cables visible on floor
ELEMENT SEVERITY: FAIL

If no loose floor cables are visible, skip the Floor Cables element entirely.

PART 2 — OVERALL SUMMARY
After all elements, provide:
OVERALL RESULT: [PASS / FLAG FOR REVIEW / FAIL]
SUMMARY: [Two to three sentences on the most significant findings.]
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
        model="gpt-5.5-2026-04-23",
        max_completion_tokens=2000,
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
# Helper: parse element detections from API response
# Returns a list of dicts:
# [{"name": "Cable Tray 1", "bbox": (l,t,r,b), "severity": "PASS", "lines": [...]}, ...]
# ---------------------------------------------------------------------------
def parse_elements(response_text: str) -> list:
    elements = []
    current = None

    for line in response_text.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.upper().startswith("ELEMENT:"):
            if current:
                elements.append(current)
            current = {
                "name": line.split(":", 1)[1].strip(),
                "bbox": None,
                "severity": "CANNOT DETERMINE",
                "criteria_lines": []
            }

        elif line.upper().startswith("BBOX:") and current:
            try:
                raw = line.split(":", 1)[1].strip()
                parts = [float(x.strip()) for x in raw.split(",")]
                if len(parts) == 4:
                    # Clamp values to 0.0–1.0
                    parts = [max(0.0, min(1.0, p)) for p in parts]
                    current["bbox"] = tuple(parts)
            except Exception:
                pass

        elif line.upper().startswith("ELEMENT SEVERITY:") and current:
            sev_raw = line.split(":", 1)[1].strip().upper()
            if "NOT APPLICABLE" in sev_raw:
                current["severity"] = "NOT APPLICABLE"
            elif "FAIL" in sev_raw:
                current["severity"] = "FAIL"
            elif "FLAG" in sev_raw:
                current["severity"] = "FLAG FOR REVIEW"
            elif "PASS" in sev_raw:
                current["severity"] = "PASS"

        elif line.upper().startswith("CRITERION") and current:
            current["criteria_lines"].append(line)

    if current:
        elements.append(current)

    return elements

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
# Helper: draw element bounding boxes on image
# ---------------------------------------------------------------------------
def annotate_image(image: Image.Image, elements: list) -> Image.Image:
    w, h = image.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Font — try system font first, fall back to default
    font_size = max(16, w // 45)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            size=font_size
        )
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            size=max(13, w // 60)
        )
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    for elem in elements:
        bbox = elem.get("bbox")
        severity = elem.get("severity", "CANNOT DETERMINE")
        name = elem.get("name", "Element")

        if not bbox:
            continue

        # Convert normalised coords to pixels
        x0 = int(bbox[0] * w)
        y0 = int(bbox[1] * h)
        x1 = int(bbox[2] * w)
        y1 = int(bbox[3] * h)

        # Ensure box has minimum size for visibility
        if x1 - x0 < 20:
            x1 = x0 + 20
        if y1 - y0 < 20:
            y1 = y0 + 20

        fill   = SEVERITY_FILL.get(severity,   (150, 150, 150, 40))
        border = SEVERITY_BORDER.get(severity, (120, 120, 120, 200))
        tcolour = SEVERITY_TEXT.get(severity,  (80,  80,  80))

        # Draw filled rectangle with border
        draw.rectangle([x0, y0, x1, y1], fill=fill, outline=border, width=3)

        # Label background pill at top of box
        label_text = f"{name} | {severity}"
        try:
            bbox_text = draw.textbbox((0, 0), label_text, font=font_small)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]
        except Exception:
            text_w, text_h = len(label_text) * 7, 14

        pad = 4
        lx0 = x0
        ly0 = max(0, y0 - text_h - pad * 2)
        lx1 = min(w, x0 + text_w + pad * 2)
        ly1 = max(0, y0)

        # Draw label background
        label_bg = border[:3] + (200,)
        draw.rectangle([lx0, ly0, lx1, ly1], fill=label_bg)

        # Draw label text
        draw.text(
            (lx0 + pad, ly0 + pad),
            label_text,
            fill=(255, 255, 255),
            font=font_small
        )

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
# Helper: render inspection results to screen
# ---------------------------------------------------------------------------
def render_results(image, response_text):
    elements = parse_elements(response_text)
    overall  = extract_overall_result(response_text)

    # Annotated image
    if elements:
        annotated = annotate_image(image, elements)
        st.image(
            annotated,
            caption="Annotated output — bounding boxes colour-coded by severity",
            use_container_width=True
        )
    else:
        st.image(image, caption="No elements detected", use_container_width=True)

    # Overall result banner
    if overall == "PASS":
        st.success(f"Overall Result: {overall}")
    elif overall == "FLAG FOR REVIEW":
        st.warning(f"Overall Result: {overall}")
    elif overall == "FAIL":
        st.error(f"Overall Result: {overall}")
    else:
        st.info(f"Overall Result: {overall}")

    # Elements summary table
    if elements:
        st.subheader("Detected Elements")
        cols = st.columns([3, 2])
        cols[0].markdown("**Element**")
        cols[1].markdown("**Severity**")
        for elem in elements:
            c1, c2 = st.columns([3, 2])
            c1.write(elem["name"])
            sev = elem["severity"]
            if sev == "PASS":
                c2.success(sev)
            elif sev == "FLAG FOR REVIEW":
                c2.warning(sev)
            elif sev == "FAIL":
                c2.error(sev)
            else:
                c2.info(sev)

    # Full written report
    with st.expander("Full Assessment Report", expanded=True):
        st.text(response_text)

    return overall, elements

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
                        render_results(image, response_text)
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
                            with st.spinner(
                                f"Analysing frame {i+1} of {len(frames)} "
                                f"(at {timestamp:.1f}s)..."
                            ):
                                try:
                                    image_b64 = encode_image_to_base64(frame_image)
                                    response_text = call_gpt_vision(
                                        api_key, image_b64, prompt_text
                                    )
                                    overall, elements = render_results.__wrapped__ \
                                        if hasattr(render_results, '__wrapped__') \
                                        else (extract_overall_result(response_text),
                                              parse_elements(response_text))
                                    results.append({
                                        "frame": i + 1,
                                        "timestamp": timestamp,
                                        "image": frame_image,
                                        "elements": parse_elements(response_text),
                                        "overall": extract_overall_result(response_text),
                                        "report": response_text
                                    })
                                except Exception as e:
                                    results.append({
                                        "frame": i + 1,
                                        "timestamp": timestamp,
                                        "image": frame_image,
                                        "elements": [],
                                        "overall": "ERROR",
                                        "report": str(e)
                                    })
                            progress.progress((i + 1) / len(frames))

                        # Session summary
                        st.subheader("Session Summary")
                        total  = len(results)
                        fails  = sum(1 for r in results if r["overall"] == "FAIL")
                        flags  = sum(1 for r in results if r["overall"] == "FLAG FOR REVIEW")
                        passes = sum(1 for r in results if r["overall"] == "PASS")

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Frames Analysed", total)
                        col2.metric("Pass", passes)
                        col3.metric("Flag for Review", flags)
                        col4.metric("Fail", fails)

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
                                render_results(r["image"], r["report"])

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
