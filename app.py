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
# Zone definitions — three horizontal content bands
#
# Based on observed site photograph layouts:
#   Cable Tray Zone    — top 35% of image (ceiling, trays, overhead cables)
#   Panel Zone         — middle 40% of image (distribution panels, switchboards)
#   Floor/Clear Zone   — bottom 25% of image (floor, working clearance, loose items)
#
# Each zone is assessed only against criteria relevant to what it contains.
# This eliminates CANNOT DETERMINE responses caused by asking the wrong
# criteria in the wrong area of the image.
# ---------------------------------------------------------------------------
ZONE_BANDS = {
    "Cable Tray Zone":         (0.00, 0.35),   # top 35%
    "Panel Zone":              (0.35, 0.75),   # middle 40%
    "Floor / Clearance Zone":  (0.75, 1.00),   # bottom 25%
}

# ---------------------------------------------------------------------------
# Inspection prompts — three-band content-based zoning
# ---------------------------------------------------------------------------
PROMPTS = {
    "Cable Tray Condition and Fill Level": """
You are an AI-assisted preliminary screening tool supporting electrical installation \
inspection in Singapore under SS 638: Code of Practice for Electrical Installations \
and EMA regulations.

The image is divided into three horizontal content zones based on what is typically \
visible in electrical plant room photographs:
- Cable Tray Zone: the upper portion of the image showing ceiling-level cable trays, \
trunking, and overhead cable runs.
- Panel Zone: the middle portion of the image showing distribution panels and switchboards.
- Floor / Clearance Zone: the lower portion of the image showing the floor area in front \
of panels and any items on the floor.

Assess each zone using ONLY the criteria relevant to that zone as follows:

CABLE TRAY ZONE — assess these criteria:
Criterion 1 (SS 638 Clause 522 — Fill Level): Cable tray fill level does not visually \
appear to exceed approximately 40% of the cross-sectional tray area.
Criterion 2 (SS 638 Clause 522 — Tray Condition): Cable tray structure appears undamaged \
with no visible crushing, cracking, or deformation.
Criterion 3 (SS 638 Clause 543 — Earth Continuity Conductor): Yellow and green earth \
continuity conductors are visible and appear to run continuously alongside power cables.

PANEL ZONE — assess this criterion only:
Criterion 1 (SS 638 Clause 522 — Cable Entry): Cables entering panels from trays appear \
supported and are not hanging freely under their own weight.

FLOOR / CLEARANCE ZONE — assess this criterion only:
Criterion 1 (SS 638 Clause 522 — Floor Cables): No unsupported cables or cable bundles \
are lying loose on the floor beneath the tray runs.

Use only what is clearly visible. Do not guess. \
Respond PASS, FLAG FOR REVIEW, FAIL, or CANNOT DETERMINE for each criterion.

Respond in this exact format:

ZONE: Cable Tray Zone
CRITERION 1: [result] — [one sentence]
CRITERION 2: [result] — [one sentence]
CRITERION 3: [result] — [one sentence]
ZONE SEVERITY: [PASS / FLAG FOR REVIEW / FAIL / NOT APPLICABLE]

ZONE: Panel Zone
CRITERION 1: [result] — [one sentence]
ZONE SEVERITY: [PASS / FLAG FOR REVIEW / FAIL / NOT APPLICABLE]

ZONE: Floor / Clearance Zone
CRITERION 1: [result] — [one sentence]
ZONE SEVERITY: [PASS / FLAG FOR REVIEW / FAIL / NOT APPLICABLE]

OVERALL RESULT: [PASS / FLAG FOR REVIEW / FAIL]
SUMMARY: [Two to three sentences on the most significant findings.]
""",

    "Distribution Panel Condition": """
You are an AI-assisted preliminary screening tool supporting electrical installation \
inspection in Singapore under SS 638: Code of Practice for Electrical Installations \
and EMA regulations.

The image is divided into three horizontal content zones based on what is typically \
visible in electrical plant room photographs:
- Cable Tray Zone: the upper portion of the image showing ceiling-level cable trays \
and overhead cable runs feeding into panels.
- Panel Zone: the middle portion of the image showing distribution panels and switchboards.
- Floor / Clearance Zone: the lower portion of the image showing the floor area and \
working clearance in front of panels.

Assess each zone using ONLY the criteria relevant to that zone as follows:

CABLE TRAY ZONE — assess this criterion only:
Criterion 1 (SS 638 Clause 522 — Cable Entry Condition): Cables descending from trays \
into panels appear organised, supported, and not in contact with sharp tray edges.

PANEL ZONE — assess these criteria:
Criterion 1 (SS 638 Clause 514 — Panel Condition): Panel doors are present, closed, \
and not visibly damaged. No exposed live parts are visible.
Criterion 2 (SS 638 Clause 514 — Labelling): Panel is visibly labelled. Warning signs \
and identification markings are visible on panel faces.
Criterion 3 (SS 638 Clause 514 — Panel Integrity): No visible signs of overheating, \
burn marks, corrosion, or physical damage on panel surfaces.

FLOOR / CLEARANCE ZONE — assess these criteria:
Criterion 1 (SS 638 Clause 513 — Working Clearance): The floor area in front of panels \
appears clear with no obstructions blocking access.
Criterion 2 (SS 638 Clause 522 — Loose Items): No loose cables, tools, or materials \
are visible on the floor directly in front of panels.

Use only what is clearly visible. Do not guess. \
Respond PASS, FLAG FOR REVIEW, FAIL, or CANNOT DETERMINE for each criterion.

Respond in this exact format:

ZONE: Cable Tray Zone
CRITERION 1: [result] — [one sentence]
ZONE SEVERITY: [PASS / FLAG FOR REVIEW / FAIL / NOT APPLICABLE]

ZONE: Panel Zone
CRITERION 1: [result] — [one sentence]
CRITERION 2: [result] — [one sentence]
CRITERION 3: [result] — [one sentence]
ZONE SEVERITY: [PASS / FLAG FOR REVIEW / FAIL / NOT APPLICABLE]

ZONE: Floor / Clearance Zone
CRITERION 1: [result] — [one sentence]
CRITERION 2: [result] — [one sentence]
ZONE SEVERITY: [PASS / FLAG FOR REVIEW / FAIL / NOT APPLICABLE]

OVERALL RESULT: [PASS / FLAG FOR REVIEW / FAIL]
SUMMARY: [Two to three sentences on the most significant findings.]
""",

    "Cable Support, Identification and Loose Cables": """
You are an AI-assisted preliminary screening tool supporting electrical installation \
inspection in Singapore under SS 638: Code of Practice for Electrical Installations \
and EMA regulations.

The image is divided into three horizontal content zones based on what is typically \
visible in electrical plant room photographs:
- Cable Tray Zone: the upper portion of the image showing ceiling-level cable trays \
and overhead cable runs.
- Panel Zone: the middle portion of the image showing distribution panels and the \
cables entering them.
- Floor / Clearance Zone: the lower portion of the image showing the floor area.

Assess each zone using ONLY the criteria relevant to that zone as follows:

CABLE TRAY ZONE — assess these criteria:
Criterion 1 (SS 638 Clause 522 — Cable Support): Cables inside trays appear secured \
at visible intervals with no unsupported hanging loops.
Criterion 2 (SS 638 Clause 514 — Cable Identification): Cables appear colour-coded or \
labelled in a consistent and identifiable manner.

PANEL ZONE — assess this criterion only:
Criterion 1 (SS 638 Clause 522 — Cable Entry Support): Cables entering panels appear \
supported and secured, not hanging freely under their own weight.

FLOOR / CLEARANCE ZONE — assess this criterion only:
Criterion 1 (SS 638 Clause 522 — Loose or Stray Cables): No unsupported cables are \
visible lying on the floor or creating potential trip hazards.

Use only what is clearly visible. Do not guess. \
Respond PASS, FLAG FOR REVIEW, FAIL, or CANNOT DETERMINE for each criterion.

Respond in this exact format:

ZONE: Cable Tray Zone
CRITERION 1: [result] — [one sentence]
CRITERION 2: [result] — [one sentence]
ZONE SEVERITY: [PASS / FLAG FOR REVIEW / FAIL / NOT APPLICABLE]

ZONE: Panel Zone
CRITERION 1: [result] — [one sentence]
ZONE SEVERITY: [PASS / FLAG FOR REVIEW / FAIL / NOT APPLICABLE]

ZONE: Floor / Clearance Zone
CRITERION 1: [result] — [one sentence]
ZONE SEVERITY: [PASS / FLAG FOR REVIEW / FAIL / NOT APPLICABLE]

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
    Returns a dict keyed by zone name: e.g.
    {"Cable Tray Zone": "PASS", "Panel Zone": "FLAG FOR REVIEW", ...}
    """
    zones = {}
    current_zone = None
    for line in response_text.splitlines():
        line_stripped = line.strip()
        if line_stripped.upper().startswith("ZONE:"):
            current_zone = line_stripped.split(":", 1)[1].strip()
        elif line_stripped.upper().startswith("ZONE SEVERITY:") and current_zone:
            severity_raw = line_stripped.split(":", 1)[1].strip().upper()
            if "NOT APPLICABLE" in severity_raw:
                zones[current_zone] = "NOT APPLICABLE"
            elif "FAIL" in severity_raw:
                zones[current_zone] = "FAIL"
            elif "FLAG" in severity_raw:
                zones[current_zone] = "FLAG FOR REVIEW"
            elif "PASS" in severity_raw:
                zones[current_zone] = "PASS"
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
# Helper: draw three-band zone annotation overlay on image
# ---------------------------------------------------------------------------
def annotate_image(image: Image.Image, zone_findings: dict) -> Image.Image:
    colour_map = {
        "PASS":            (0,   180,  0,   70),
        "FLAG FOR REVIEW": (255, 165,  0,   70),
        "FAIL":            (200,  0,   0,   70),
        "NOT APPLICABLE":  (150, 150, 150,  30),
    }
    border_map = {
        "PASS":            (0,   140,  0,  220),
        "FLAG FOR REVIEW": (200, 120,  0,  220),
        "FAIL":            (160,  0,   0,  220),
        "NOT APPLICABLE":  (120, 120, 120, 180),
    }
    label_colour = {
        "PASS":            (0,   100,  0),
        "FLAG FOR REVIEW": (140,  80,  0),
        "FAIL":            (140,   0,  0),
        "NOT APPLICABLE":  ( 80,  80, 80),
    }

    w, h = image.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Try to load a larger font; fall back to default if unavailable
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=max(18, w // 40))
    except Exception:
        font = ImageFont.load_default()

    # Map zone names to pixel bands
    zone_pixel_bands = {}
    for zone_name, (top_frac, bot_frac) in ZONE_BANDS.items():
        y0 = int(h * top_frac)
        y1 = int(h * bot_frac)
        zone_pixel_bands[zone_name] = (0, y0, w, y1)

    for zone_name, severity in zone_findings.items():
        coords = zone_pixel_bands.get(zone_name)
        if not coords:
            continue
        x0, y0, x1, y1 = coords
        fill   = colour_map.get(severity,  (150, 150, 150, 30))
        border = border_map.get(severity,  (120, 120, 120, 180))

        # Fill rectangle
        draw.rectangle([x0, y0, x1, y1], fill=fill, outline=border, width=4)

        # Zone name label — top left of band
        name_label = zone_name
        draw.text((x0 + 12, y0 + 10), name_label,
                  fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))

        # Severity label — centred in band
        sev_label = severity
        tx = (x0 + x1) // 2
        ty = (y0 + y1) // 2
        tc = label_colour.get(severity, (80, 80, 80))
        draw.text((tx, ty), sev_label,
                  fill=tc, font=font, anchor="mm",
                  stroke_width=2, stroke_fill=(255, 255, 255))

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
                            st.image(image, caption="No zone data parsed",
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
                            with st.spinner(
                                f"Analysing frame {i+1} of {len(frames)} "
                                f"(at {timestamp:.1f}s)..."
                            ):
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
