# AI-Assisted MEP Electrical Installation Inspector

A proof-of-concept Streamlit web application that uses Claude Opus 4 (via the Anthropic API) to analyse site photographs and video walkthroughs for electrical installation compliance against **SS 638: Code of Practice for Electrical Installations (Singapore)**.

> **Disclaimer:** This tool is a preliminary screening aid only. It does not constitute a formal electrical inspection report and carries no legal or regulatory standing under EMA requirements.

---

## Overview

This prototype was developed as a Final Year Project at Coventry University (Cohort 224, PT). It demonstrates the feasibility of applying a Vision Language Model (VLM) to construction site imagery to identify potential non-conformances across three SS 638 inspection categories, without replacing a licensed electrical engineer.

---

## Features

- **Single photograph mode** — upload a JPEG or PNG site photograph for immediate analysis
- **Video walkthrough mode** — upload an MP4, MOV, or AVI file; frames are extracted at a configurable interval using OpenCV and each frame is analysed independently
- **Three inspection categories:**
  - Cable Tray Condition and Fill Level
  - Distribution Panel Condition
  - Cable Support, Identification and Loose Cables
- **Colour-coded annotated image output** — detected elements are overlaid with bounding zones coloured green (Pass), amber (Flag for Review), red (Fail), or grey (Cannot Determine)
- **Structured assessment report** — element-by-element findings with an overall verdict
- **Session summary** — for video mode, aggregated Pass / Flag / Fail counts across all extracted frames

---

## Requirements

- Python 3.9 or later
- An Anthropic API key with access to `claude-opus-4-8`

### Python dependencies

```
streamlit
anthropic
Pillow
opencv-python
```

Install with:

```bash
pip install streamlit anthropic Pillow opencv-python
```

OpenCV is optional — the app will run in single-photograph mode without it, but video walkthrough mode requires it.

---

## Running the app

```bash
streamlit run app.py
```

Enter your Anthropic API key in the sidebar when the app opens. The key is used only for the duration of the session and is not stored or logged by the application.

---

## How it works

1. The user selects an inspection category and uploads a photograph or video.
2. The image (or extracted video frame) is encoded to base64 and sent to Claude Opus 4 via the Anthropic Messages API along with a category-specific prompt.
3. The model returns a structured natural-language report identifying detected elements and assigning a compliance verdict to each.
4. The app parses the response, maps elements to image regions using a rule-based layout, and renders colour-coded annotation boxes over the original image.
5. The overall verdict and element-level breakdown are displayed alongside the full report text.

---

## Inspection logic

Prompts are designed around visually assessable criteria drawn from SS 638. The model is instructed to return verdicts as one of four values:

| Verdict | Meaning |
|---|---|
| PASS | No visible non-conformance detected |
| FLAG FOR REVIEW | Possible issue; requires engineer verification |
| FAIL | Clear non-conformance visible |
| CANNOT DETERMINE | Insufficient visibility to assess |

The model's confidence levels are indicative only and are not calibrated probability scores.

---

## Limitations and Scope

- The tool assesses visual evidence only; it cannot evaluate wiring continuity, insulation resistance, or any parameter not visible in the image.
- Annotation boxes are positioned using rule-based image region estimates, not object detection coordinates. Their position is approximate.
- Results must be verified by a qualified electrical engineer before any compliance decision is made.
- This prototype was evaluated against a controlled test set; performance on unseen site conditions may vary.
The following scopes define the boundaries within which this project is conducted: 
- Scope 1: The prototype will assess electrical installations only. Other MEP systems such as 
mechanical HVAC ductwork and plumbing pipework are excluded from this project. 
- Scope 2: Visual Inspection covers three defined categories 
1) Cable tray condition and fill 
2) Distribution panel condition 
3) Cable support, identification and loose cables identification 
               These categories were selected based on their suitability for visual assessment. This 
selection is covered in Section 4.1. 
- Scope 3: the input for the prototype is limited to site photos and recorded video. Real-time 
video and live camera feed analysis are outside of the scope of this project. 
- Scope 4: The AI assessment prototype is a preliminary screening tool only. It does not replace 
a formal electrical inspection and cannot substitute for the professional judgement of a 
licensed electrical engineer as required under SS 638 standard and EMA regulations 
- Scope 5: Testing of the prototype will use site photos and video collected from construction 
sites, subject to data access permissions.


---

## Project context

| Item | Detail |
|---|---|
| University | Coventry University |
| Programme | Electrical / Mechanical Engineering (Part-Time) |
| Cohort | 224 PT |
| Supervisor | Dr Jason Tan |
| Standard referenced | SS 638: Code of Practice for Electrical Installations |
| VLM used | Claude Opus 4 (`claude-opus-4-8`) via Anthropic API |

---

## Licence

This repository is released for academic demonstration purposes. It is not intended for use in live construction inspections or regulatory submissions.
