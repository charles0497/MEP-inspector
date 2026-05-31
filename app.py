import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import tempfile
import os
import re

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
SEVERITY_FILL   = {
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
SEVERITY_TEXT   = {
    "PASS":             (0,   100,   0),
    "FLAG FOR REVIEW":  (150,  90,   0),
    "FAIL":             (150,   0,   0),
    "CANNOT DETERMINE": ( 80,  80,  80),
}

# ---------------------------------------------------------------------------
# Rule-based layout engine
#
# Based on observed patterns across site photographs:
#   - Cable trays occupy the top ~35% of the frame
#   - Distribution panels occupy the middle ~40%
#   - Floor / working clearance occupies the bottom ~25%
#
# When the model reports N instances of an element type, this engine
# divides the relevant band into N equal vertical columns and draws
# one box per instance within that band.
# ---------------------------------------------------------------------------

ELEMENT_BANDS = {
    # element_type_key : (top_fraction, bottom_fraction)
    "cable_tray":        (0.00, 0.38),
    "distribution_panel":(0.30, 0.78),
    "floor_clearance":   (0.72, 1.00),
    "cable_run":         (0.00, 0.75),
    "loose_cables":      (0.72, 1.00),
}

def compute_boxes(element_type: str, count: int, img_w: int, img_h: int,
                  h_margin: float = 0.03) -> list:
    """
    Divide the element band into `count` equal vertical columns.
    Returns a list of pixel-coordinate tuples (x0, y0, x1, y1).
    """
    band = ELEMENT_BANDS.get(element_type)
    if not band:
        band = (0.10, 0.90)

    y0 = int(band[0] * img_h)
    y1 = int(band[1] * img_h)

    col_w = img_w / max(count, 1)
    margin = int(h_margin * img_w)

    boxes = []
    for i in range(count):
        bx0 = int(i * col_w) + margin
        bx1 = int((i + 1) * col_w) - margin
        boxes.append((bx0, y0, bx1, y1))
    return boxes

# ---------------------------------------------------------------------------
# Prompts — GPT-5.5 identifies element types and counts, assesses criteria
# ---------------------------------------------------------------------------
PROMPTS = {
    "Cable Tray Condition and Fill Level": """
You are an AI-assisted preliminary screening tool supporting electrical installation \
inspection in Singapore under SS 638: Code of Practice for Electrical Installations \
and EMA regulations.

Carefully examine the image and identify all visible cable trays, trunking runs, \
and cable ladders. Count how many distinct cable tray sections are visible.

For EACH cable tray section, assess the following criteria using only what is \
clearly visible. Do not guess if something is not visible.

Criterion 1 (SS 638 Clause 522 — Fill Level):
Cable tray fill level does not visually appear to exceed approximately 40% of the \
cross-sectional tray area.

Criterion 2 (SS 638 Clause 522 — Tray Condition):
Cable tray structure appears undamaged with no visible crushing, cracking, or deformation.

Criterion 3 (SS 638 Clause 543 — Earth Continuity Conductor):
Yellow and green earth continuity conductors are visible and appear to run continuously \
alongside power cables.

Also check the floor area and report whether loose cables are present on the floor.

Respond in this EXACT format:

CABLE TRAY COUNT: [number]

CABLE TRAY 1:
CRITERION 1: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence]
CRITERION 2: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence]
CRITERION 3: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence]
SEVERITY: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE]

CABLE TRAY 2:
CRITERION 1: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence]
CRITERION 2: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence]
CRITERION 3: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence]
SEVERITY: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE]

(repeat for each tray found)

FLOOR CABLES:
LOOSE CABLES PRESENT: [YES / NO]
SEVERITY: [FAIL / PASS]

OVERALL RESULT: [PASS / FLAG FOR REVIEW / FAIL]
SUMMARY: [Two to three sentences on the most significant findings.]
""",

    "Distribution Panel Condition": """
You are an AI-assisted preliminary screening tool supporting electrical installation \
inspection in Singapore under SS 638: Code of Practice for Electrical Installations \
and EMA regulations.

Carefully examine the image and identify all visible distribution panels, switchboards, \
and electrical enclosures. Count how many distinct panels are visible.

For EACH panel, assess the following criteria using only what is clearly visible. \
Do not guess if something is not visible.

Criterion 1 (SS 638 Clause 514 — Panel Condition):
Panel doors are present, closed, and not visibly damaged. No exposed live parts visible.

Criterion 2 (SS 638 Clause 514 — Labelling):
Panel is visibly labelled. Warning signs and identification markings are visible on \
panel faces.

Criterion 3 (SS 638 Clause 514 — Panel Integrity):
No visible signs of overheating, burn marks, corrosion, or physical damage on panel \
surfaces.

Also assess the floor area in front of the panels for working clearance.

Respond in this EXACT format:

DISTRIBUTION PANEL COUNT: [number]

DISTRIBUTION PANEL 1:
CRITERION 1: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence]
CRITERION 2: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence]
CRITERION 3: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence]
SEVERITY: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE]

DISTRIBUTION PANEL 2:
CRITERION 1: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence]
CRITERION 2: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence]
CRITERION 3: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence]
SEVERITY: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE]

(repeat for each panel found)

FLOOR / WORKING CLEARANCE:
CRITERION 1 (SS 638 Clause 513 — Working Clearance): [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence]
CRITERION 2 (SS 638 Clause 522 — Loose Items): [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence]
SEVERITY: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE]

OVERALL RESULT: [PASS / FLAG FOR REVIEW / FAIL]
SUMMARY: [Two to three sentences on the most significant findings.]
""",

    "Cable Support, Identification and Loose Cables": """
You are an AI-assisted preliminary screening tool supporting electrical installation \
inspection in Singapore under SS 638: Code of Practice for Electrical Installations \
and EMA regulations.

Carefully examine the image and identify all visible cable runs and cable groups. \
Count how many distinct cable runs or bundles are visible.

For EACH cable run, assess the following criteria using only what is clearly visible. \
Do not guess if something is not visible.

Criterion 1 (SS 638 Clause 522 — Cable Support):
Cables appear secured to supports at visible intervals with no unsupported hanging \
loops evident.

Criterion 2 (SS 638 Clause 514 — Cable Identification):
Cables appear colour-coded or labelled in a consistent and identifiable manner.

Also check the floor area for loose or stray cables.

Respond in this EXACT format:

CABLE RUN COUNT: [number]

CABLE RUN 1:
CRITERION 1: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence]
CRITERION 2: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence]
SEVERITY: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE]

CABLE RUN 2:
CRITERION 1: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence]
CRITERION 2: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE] — [one sentence]
SEVERITY: [PASS / FLAG FOR REVIEW / FAIL / CANNOT DETERMINE]

(repeat for each cable run found)

FLOOR CABLES:
LOOSE CABLES PRESENT: [YES / NO]
SEVERITY: [FAIL / PASS]

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
# Helper: parse GPT-5.5 response into structured element list
# ---------------------------------------------------------------------------
def parse_response(response_text: str, category: str) -> list:
    """
    Returns a list of dicts:
    [{"element_type": "distribution_panel", "label": "Distribution Panel 1",
      "count_index": 0, "total_count": 3, "severity": "FLAG FOR REVIEW",
      "criteria": ["C1: PASS...", ...]}, ...]
    """
    elements = []
    lines = response_text.splitlines()

    # Determine element type key from category
    if "Cable Tray" in category:
        primary_key  = "cable_tray"
        primary_label = "Cable Tray"
        count_pattern = r"CABLE TRAY COUNT\s*:\s*(\d+)"
        block_pattern = r"CABLE TRAY (\d+)\s*:"
    elif "Distribution Panel" in category:
        primary_key  = "distribution_panel"
        primary_label = "Distribution Panel"
        count_pattern = r"DISTRIBUTION PANEL COUNT\s*:\s*(\d+)"
        block_pattern = r"DISTRIBUTION PANEL (\d+)\s*:"
    else:
        primary_key  = "cable_run"
        primary_label = "Cable Run"
        count_pattern = r"CABLE RUN COUNT\s*:\s*(\d+)"
        block_pattern = r"CABLE RUN (\d+)\s*:"

    # Extract declared count
    total_count = 1
    for line in lines:
        m = re.search(count_pattern, line, re.IGNORECASE)
        if m:
            total_count = max(1, int(m.group(1)))
            break

    # Parse individual element blocks
    current_elem = None
    for line in lines:
        line_s = line.strip()

        # New primary element block
        bm = re.match(block_pattern, line_s, re.IGNORECASE)
        if bm:
            if current_elem:
                elements.append(current_elem)
            idx = int(bm.group(1)) - 1
            current_elem = {
                "element_type": primary_key,
                "label": f"{primary_label} {bm.group(1)}",
                "count_index": idx,
                "total_count": total_count,
                "severity": "CANNOT DETERMINE",
                "criteria": []
            }
            continue

        # Floor / clearance block
        if re.match(r"FLOOR\s*/?\s*(WORKING CLEARANCE|CABLES)\s*:", line_s, re.IGNORECASE):
            if current_elem:
                elements.append(current_elem)
            etype = "floor_clearance" if "CLEARANCE" in line_s.upper() else "loose_cables"
            current_elem = {
                "element_type": etype,
                "label": "Floor / Working Clearance" if etype == "floor_clearance" else "Floor Cables",
                "count_index": 0,
                "total_count": 1,
                "severity": "CANNOT DETERMINE",
                "criteria": []
            }
            continue

        if current_elem is None:
            continue

        # Severity line
        if re.match(r"SEVERITY\s*:", line_s, re.IGNORECASE):
            sev_raw = line_s.split(":", 1)[1].strip().upper()
            if "NOT APPLICABLE" in sev_raw:
                current_elem["severity"] = "NOT APPLICABLE"
            elif "FAIL" in sev_raw:
                current_elem["severity"] = "FAIL"
            elif "FLAG" in sev_raw:
                current_elem["severity"] = "FLAG FOR REVIEW"
            elif "PASS" in sev_raw:
                current_elem["severity"] = "PASS"
            continue

        # Loose cables present line
        if re.match(r"LOOSE CABLES PRESENT\s*:", line_s, re.IGNORECASE):
            val = line_s.split(":", 1)[1].strip().upper()
            if "YES" in val:
                current_elem["severity"] = "FAIL"
            elif "NO" in val:
                current_elem["severity"] = "PASS"
            continue

        # Criterion lines
        if re.match(r"CRITERION\s+\d", line_s, re.IGNORECASE):
            current_elem["criteria"].append(line_s)

    if current_elem:
        elements.append(current_elem)

    return elements

# ---------------------------------------------------------------------------
# Helper: extract overall result
# ---------------------------------------------------------------------------
def extract_overall_result(response_text: str) -> str:
    for line in response_text.splitlines():
        if line.strip().upper().startswith("OVERALL RESULT:"):
            val = line.split(":", 1)[1].strip().upper()
            if "FAIL" in val:
                return "FAIL"
            elif "FLAG" in val:
                return "FLAG FOR REVIEW"
            elif "PASS" in val:
                return "PASS"
    return "CANNOT DETERMINE"

# ---------------------------------------------------------------------------
# Helper: draw rule-based boxes on image
# ---------------------------------------------------------------------------
def annotate_image(image: Image.Image, elements: list) -> Image.Image:
    w, h = image.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_size = max(16, w // 45)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=font_size)
        font_sm = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            size=max(13, w // 60))
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    # Group elements by type to calculate column layouts
    from collections import defaultdict
    type_groups = defaultdict(list)
    for elem in elements:
        type_groups[elem["element_type"]].append(elem)

    for etype, group in type_groups.items():
        count = len(group)
        boxes = compute_boxes(etype, count, w, h)

        for i, (elem, box) in enumerate(zip(group, boxes)):
            x0, y0, x1, y1 = box
            severity = elem["severity"]

            fill   = SEVERITY_FILL.get(severity,   (150, 150, 150, 40))
            border = SEVERITY_BORDER.get(severity, (120, 120, 120, 200))
            tcolour = SEVERITY_TEXT.get(severity,  (80,  80,  80))

            # Main box
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline=border, width=3)

            # Label tag at top of box
            label_text = f"{elem['label']}  |  {severity}"
            try:
                tb = draw.textbbox((0, 0), label_text, font=font_sm)
                tw = tb[2] - tb[0]
                th = tb[3] - tb[1]
            except Exception:
                tw = len(label_text) * 7
                th = 14

            pad = 5
            lx0 = x0
            ly0 = max(0, y0 - th - pad * 2)
            lx1 = min(w, x0 + tw + pad * 2)
            ly1 = y0

            tag_bg = border[:3] + (210,)
            draw.rectangle([lx0, ly0, lx1, ly1], fill=tag_bg)
            draw.text((lx0 + pad, ly0 + pad), label_text,
                      fill=(255, 255, 255), font=font_sm)

    base = image.convert("RGBA")
    combined = Image.alpha_composite(base, overlay)
    return combined.convert("RGB")

# ---------------------------------------------------------------------------
# Helper: render full results to screen
# ---------------------------------------------------------------------------
def render_results(image: Image.Image, response_text: str, cat: str):
    elements = parse_response(response_text, cat)
    overall  = extract_overall_result(response_text)

    # Annotated image
    if elements:
        annotated = annotate_image(image, elements)
        st.image(annotated,
                 caption="Annotated output — colour-coded by element and severity",
                 use_container_width=True)
    else:
        st.image(image, caption="No elements parsed from response",
                 use_container_width=True)

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

            # Show criteria detail under each element
            if elem["criteria"]:
                with st.expander(f"Criteria detail — {elem['label']}"):
                    for cline in elem["criteria"]:
                        st.write(cline)

    # Full raw report
    with st.expander("Full Assessment Report", expanded=False):
        st.text(response_text)

    return overall

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
                        response_text = call_gpt_vision(api_key, image_b64, prompt_text)
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
                                        api_key, image_b64, prompt_text)
                                    results.append({
                                        "frame": i + 1,
                                        "timestamp": timestamp,
                                        "image": frame_image,
                                        "overall": extract_overall_result(response_text),
                                        "report": response_text
                                    })
                                except Exception as e:
                                    results.append({
                                        "frame": i + 1,
                                        "timestamp": timestamp,
                                        "image": frame_image,
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

                        st.subheader("Frame-by-Frame Results")
                        for r in results:
                            with st.expander(
                                f"Frame {r['frame']} | t={r['timestamp']:.1f}s | "
                                f"Result: {r['overall']}"
                            ):
                                render_results(r["image"], r["report"], category)

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
