import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import tempfile
import os
import re
from collections import defaultdict

# Optional dependencies
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
# Sidebar
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
        ]
    )

    input_mode = st.radio(
        "Input Type",
        options=["Single Photograph", "Video Walkthrough"]
    )

    if input_mode == "Video Walkthrough":
        frame_interval = st.slider("Frame extraction interval (seconds)", 1, 30, 5)
        max_frames = st.slider("Maximum frames to analyse", 1, 20, 5)

# ---------------------------------------------------------------------------
# Colour scheme
# ---------------------------------------------------------------------------
SEVERITY_FILL = {
    "PASS":             (0,   200,   0,  55),
    "FLAG FOR REVIEW":  (255, 165,   0,  55),
    "FAIL":             (220,   0,   0,  55),
    "CANNOT DETERMINE": (150, 150, 150,  40),
}
SEVERITY_BORDER = {
    "PASS":             (0,   160,   0, 240),
    "FLAG FOR REVIEW":  (210, 130,   0, 240),
    "FAIL":             (180,   0,   0, 240),
    "CANNOT DETERMINE": (120, 120, 120, 200),
}

# ---------------------------------------------------------------------------
# Simplified prompts — shorter, outcome-focused, no rigid format demands
# GPT-5.5 is asked for structured JSON-like output but given freedom in phrasing
# ---------------------------------------------------------------------------
PROMPTS = {
    "Cable Tray Condition and Fill Level": """
You are assisting with a preliminary electrical installation inspection under \
SS 638 (Singapore Code of Practice for Electrical Installations).

Look at this construction site photograph. Identify all visible cable trays and \
trunking runs. For each one, assess the following and give a verdict of PASS, \
FLAG FOR REVIEW, FAIL, or CANNOT DETERMINE:

1. Fill level — does the tray appear to exceed 40% fill of its cross-section?
2. Tray condition — any visible crushing, cracking, or deformation?
3. Earth conductor — are yellow-green earth continuity conductors visible \
running alongside the power cables?

Also note whether any loose cables are lying on the floor.

Count how many cable trays you can see. Then for each one write:
- "Cable Tray [N]: [your findings and verdict]"

At the end write:
- "Floor cables: [YES/NO loose cables on floor]"
- "Overall: [PASS / FLAG FOR REVIEW / FAIL]"
- "Summary: [two sentences]"
""",

    "Distribution Panel Condition": """
You are assisting with a preliminary electrical installation inspection under \
SS 638 (Singapore Code of Practice for Electrical Installations).

Look at this construction site photograph. Identify all visible distribution \
panels and switchboards. For each one, assess the following and give a verdict \
of PASS, FLAG FOR REVIEW, FAIL, or CANNOT DETERMINE:

1. Panel condition — are doors closed, undamaged, with no exposed live parts?
2. Labelling — are warning signs and identification markings visible on the panel face?
3. Integrity — any visible burn marks, corrosion, or physical damage?

Also check the floor area in front of panels for working clearance and loose items.

Count how many panels you can see. Then for each one write:
- "Distribution Panel [N]: [your findings and verdict]"

At the end write:
- "Floor / Working Clearance: [your finding and verdict]"
- "Overall: [PASS / FLAG FOR REVIEW / FAIL]"
- "Summary: [two sentences]"
""",

    "Cable Support, Identification and Loose Cables": """
You are assisting with a preliminary electrical installation inspection under \
SS 638 (Singapore Code of Practice for Electrical Installations).

Look at this construction site photograph. Identify all visible cable runs and \
cable groups. For each one, assess the following and give a verdict of PASS, \
FLAG FOR REVIEW, FAIL, or CANNOT DETERMINE:

1. Cable support — are cables secured at intervals with no unsupported hanging loops?
2. Cable identification — are cables colour-coded or labelled consistently?

Also check whether any loose or stray cables are lying on the floor.

Count how many distinct cable runs you can see. Then for each one write:
- "Cable Run [N]: [your findings and verdict]"

At the end write:
- "Floor cables: [YES/NO loose cables on floor]"
- "Overall: [PASS / FLAG FOR REVIEW / FAIL]"
- "Summary: [two sentences]"
"""
}

# ---------------------------------------------------------------------------
# Rule-based layout for drawing boxes
# ---------------------------------------------------------------------------
ELEMENT_BANDS = {
    "cable_tray":         (0.00, 0.38),
    "distribution_panel": (0.28, 0.78),
    "cable_run":          (0.00, 0.75),
    "floor":              (0.73, 1.00),
}

def compute_boxes(element_type: str, count: int,
                  img_w: int, img_h: int) -> list:
    band = ELEMENT_BANDS.get(element_type, (0.10, 0.90))
    y0 = int(band[0] * img_h)
    y1 = int(band[1] * img_h)
    col_w = img_w / max(count, 1)
    margin = int(0.02 * img_w)
    boxes = []
    for i in range(count):
        bx0 = int(i * col_w) + margin
        bx1 = int((i + 1) * col_w) - margin
        boxes.append((bx0, y0, bx1, y1))
    return boxes

# ---------------------------------------------------------------------------
# Helper: encode image to base64
# ---------------------------------------------------------------------------
def encode_image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# ---------------------------------------------------------------------------
# Helper: call GPT-5.5
# ---------------------------------------------------------------------------
def call_gpt_vision(api_key: str, image_b64: str, prompt: str) -> str:
    if not OPENAI_AVAILABLE:
        return "ERROR: openai package is not installed."
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
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    )
    return response.choices[0].message.content

# ---------------------------------------------------------------------------
# Helper: flexible parser — scans response text for element mentions
# and verdict keywords regardless of exact formatting
# ---------------------------------------------------------------------------
def extract_severity(text: str) -> str:
    """Extract the strongest severity verdict from a block of text."""
    text_up = text.upper()
    if "FAIL" in text_up:
        return "FAIL"
    elif "FLAG FOR REVIEW" in text_up or "FLAG" in text_up:
        return "FLAG FOR REVIEW"
    elif "PASS" in text_up:
        return "PASS"
    return "CANNOT DETERMINE"

def parse_response(response_text: str, category: str) -> list:
    """
    Flexibly parses the response regardless of exact formatting.
    Looks for lines mentioning element types and extracts verdict.
    Returns list of element dicts for annotation.
    """
    if not response_text or not response_text.strip():
        return []

    elements = []
    lines = response_text.splitlines()

    # Determine what element type to look for
    if "Cable Tray" in category:
        primary_pattern = re.compile(r"cable\s*tray\s*(\d+)", re.IGNORECASE)
        primary_type    = "cable_tray"
        primary_label   = "Cable Tray"
    elif "Distribution Panel" in category:
        primary_pattern = re.compile(r"distribution\s*panel\s*(\d+)", re.IGNORECASE)
        primary_type    = "distribution_panel"
        primary_label   = "Distribution Panel"
    else:
        primary_pattern = re.compile(r"cable\s*run\s*(\d+)", re.IGNORECASE)
        primary_type    = "cable_run"
        primary_label   = "Cable Run"

    floor_pattern = re.compile(
        r"(floor|working\s*clearance|loose\s*cable)", re.IGNORECASE)
    overall_pattern = re.compile(r"overall\s*:?\s*(pass|flag for review|fail)",
                                 re.IGNORECASE)

    found_indices = set()

    for line in lines:
        # Primary elements
        m = primary_pattern.search(line)
        if m:
            idx = int(m.group(1))
            if idx not in found_indices:
                found_indices.add(idx)
                severity = extract_severity(line)
                elements.append({
                    "element_type": primary_type,
                    "label": f"{primary_label} {idx}",
                    "count_index": idx - 1,
                    "severity": severity,
                    "text": line.strip()
                })

        # Floor / clearance
        elif floor_pattern.search(line) and "overall" not in line.lower():
            severity = extract_severity(line)
            # Only add once
            if not any(e["element_type"] == "floor" for e in elements):
                elements.append({
                    "element_type": "floor",
                    "label": "Floor / Clearance",
                    "count_index": 0,
                    "severity": severity,
                    "text": line.strip()
                })

    # If no elements parsed but response is non-empty,
    # create a single generic element so the box still renders
    if not elements and response_text.strip():
        overall = "CANNOT DETERMINE"
        for line in lines:
            om = overall_pattern.search(line)
            if om:
                overall = extract_severity(om.group(1))
                break
        elements.append({
            "element_type": primary_type,
            "label": primary_label,
            "count_index": 0,
            "severity": overall,
            "text": response_text[:200]
        })

    return elements

# ---------------------------------------------------------------------------
# Helper: extract overall result
# ---------------------------------------------------------------------------
def extract_overall_result(response_text: str) -> str:
    if not response_text:
        return "CANNOT DETERMINE"
    for line in response_text.splitlines():
        if re.search(r"overall", line, re.IGNORECASE):
            val = line.upper()
            if "FAIL" in val:
                return "FAIL"
            elif "FLAG" in val:
                return "FLAG FOR REVIEW"
            elif "PASS" in val:
                return "PASS"
    # Fall back to scanning full text
    return extract_severity(response_text)

# ---------------------------------------------------------------------------
# Helper: draw rule-based boxes
# ---------------------------------------------------------------------------
def annotate_image(image: Image.Image, elements: list) -> Image.Image:
    w, h = image.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_size = max(16, w // 45)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            size=font_size)
        font_sm = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            size=max(13, w // 60))
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    # Group by element type for column layout
    type_groups = defaultdict(list)
    for elem in elements:
        type_groups[elem["element_type"]].append(elem)

    for etype, group in type_groups.items():
        count = len(group)
        boxes = compute_boxes(etype, count, w, h)

        for elem, box in zip(group, boxes):
            x0, y0, x1, y1 = box
            severity = elem["severity"]

            fill   = SEVERITY_FILL.get(severity,   (150, 150, 150, 40))
            border = SEVERITY_BORDER.get(severity, (120, 120, 120, 200))

            draw.rectangle([x0, y0, x1, y1],
                           fill=fill, outline=border, width=3)

            # Label tag
            label_text = f"{elem['label']}  |  {severity}"
            try:
                tb = draw.textbbox((0, 0), label_text, font=font_sm)
                tw = tb[2] - tb[0]
                th = tb[3] - tb[1]
            except Exception:
                tw, th = len(label_text) * 7, 14

            pad  = 5
            lx0  = x0
            ly0  = max(0, y0 - th - pad * 2)
            lx1  = min(w, x0 + tw + pad * 2)
            ly1  = y0

            tag_bg = border[:3] + (210,)
            draw.rectangle([lx0, ly0, lx1, ly1], fill=tag_bg)
            draw.text((lx0 + pad, ly0 + pad), label_text,
                      fill=(255, 255, 255), font=font_sm)

    base     = image.convert("RGBA")
    combined = Image.alpha_composite(base, overlay)
    return combined.convert("RGB")

# ---------------------------------------------------------------------------
# Helper: render results to screen
# ---------------------------------------------------------------------------
def render_results(image: Image.Image, response_text: str, cat: str):
    if not response_text or not response_text.strip():
        st.error("The model returned an empty response. Please try again.")
        st.image(image, use_container_width=True)
        return "CANNOT DETERMINE"

    elements = parse_response(response_text, cat)
    overall  = extract_overall_result(response_text)

    # Annotated image
    if elements:
        annotated = annotate_image(image, elements)
        st.image(annotated,
                 caption="Annotated output — colour-coded by element and severity",
                 use_container_width=True)
    else:
        st.image(image, caption="Uploaded photograph", use_container_width=True)

    # Overall result banner
    if overall == "PASS":
        st.success(f"Overall Result: {overall}")
    elif overall == "FLAG FOR REVIEW":
        st.warning(f"Overall Result: {overall}")
    elif overall == "FAIL":
        st.error(f"Overall Result: {overall}")
    else:
        st.info(f"Overall Result: {overall}")

    # Element summary
    if elements:
        st.subheader("Detected Elements")
        for elem in elements:
            c1, c2 = st.columns([3, 2])
            c1.write(elem["label"])
            sev = elem["severity"]
            if sev == "PASS":
                c2.success(sev)
            elif sev == "FLAG FOR REVIEW":
                c2.warning(sev)
            elif sev == "FAIL":
                c2.error(sev)
            else:
                c2.info(sev)

    # Full report
    with st.expander("Full Assessment Report", expanded=True):
        st.write(response_text)

    return overall

# ---------------------------------------------------------------------------
# Helper: extract frames from video
# ---------------------------------------------------------------------------
def extract_frames(video_path: str, interval_seconds: int,
                   max_frames: int) -> list:
    if not CV2_AVAILABLE:
        return []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_step = int(fps * interval_seconds)
    frames = []
    frame_index = 0
    while len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append((Image.fromarray(rgb), frame_index / fps))
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
        type=["jpg", "jpeg", "png"]
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
                        response_text = call_gpt_vision(
                            api_key, image_b64, prompt_text)
                        render_results(image, response_text, category)
                    except Exception as e:
                        st.error(f"API request failed: {e}")

elif input_mode == "Video Walkthrough":
    if not CV2_AVAILABLE:
        st.error("OpenCV is required for video mode.")
    else:
        uploaded_video = st.file_uploader(
            "Upload video walkthrough",
            type=["mp4", "mov", "avi"]
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
                        st.info(
                            f"{len(frames)} frame(s) extracted. "
                            "Sending to GPT-5.5...")
                        progress = st.progress(0)
                        results  = []

                        for i, (frame_image, timestamp) in enumerate(frames):
                            with st.spinner(
                                f"Analysing frame {i+1} of {len(frames)} "
                                f"(at {timestamp:.1f}s)..."
                            ):
                                try:
                                    image_b64 = encode_image_to_base64(
                                        frame_image)
                                    response_text = call_gpt_vision(
                                        api_key, image_b64, prompt_text)
                                    results.append({
                                        "frame":     i + 1,
                                        "timestamp": timestamp,
                                        "image":     frame_image,
                                        "overall":   extract_overall_result(
                                            response_text),
                                        "report":    response_text
                                    })
                                except Exception as e:
                                    results.append({
                                        "frame":     i + 1,
                                        "timestamp": timestamp,
                                        "image":     frame_image,
                                        "overall":   "ERROR",
                                        "report":    str(e)
                                    })
                            progress.progress((i + 1) / len(frames))

                        # Session summary
                        st.subheader("Session Summary")
                        total  = len(results)
                        fails  = sum(1 for r in results
                                     if r["overall"] == "FAIL")
                        flags  = sum(1 for r in results
                                     if r["overall"] == "FLAG FOR REVIEW")
                        passes = sum(1 for r in results
                                     if r["overall"] == "PASS")

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Frames Analysed", total)
                        col2.metric("Pass", passes)
                        col3.metric("Flag for Review", flags)
                        col4.metric("Fail", fails)

                        if fails > 0:
                            st.error(
                                "Worst-case result across session: FAIL")
                        elif flags > 0:
                            st.warning(
                                "Worst-case result across session: "
                                "FLAG FOR REVIEW")
                        else:
                            st.success(
                                "Worst-case result across session: PASS")

                        st.subheader("Frame-by-Frame Results")
                        for r in results:
                            with st.expander(
                                f"Frame {r['frame']} | "
                                f"t={r['timestamp']:.1f}s | "
                                f"Result: {r['overall']}"
                            ):
                                render_results(
                                    r["image"], r["report"], category)

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
