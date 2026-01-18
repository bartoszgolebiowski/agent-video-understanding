"""Video processing script using VideoLLMClient.

Processes videos one by one from the R2 bucket and analyzes them using the video LLM.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel, Field

from src.llm import VideoLLMClient


class VideoAnalysisResult(BaseModel):
    """Structured output for video analysis."""

    content: str = Field(
        description="Comprehensive analysis of the video content including financial data, sentiment, key insights, and actionable intelligence"
    )


# Configuration
VIDEOS_DIR = Path(__file__).parent / "videos"
NUM_FRAMES = 3


# Default prompt - update this as needed
DEFAULT_PROMPT = """
# MISSION
Perform a Frame-by-Frame Forensic Audit. Your goal is 100% data density. Analyze these images as if you are reconstructing a scene for a legal or scientific record.

# 1. PER-FRAME MICROSCOPIC AUDIT
For EACH frame provided, extract the following:
* **Subject Inventory:** List every distinct entity (person, animal, vehicle, tool).
* **Detailed Attributes:** For every entity, note color, texture, material, brand/logo, and current state (e.g., "Phone: black glass, cracked screen, 42% battery visible").
* **Text/OCR Dump:** Transcribe every character found. Include fine print, background signs, UI elements, and timestamps. **EXCEPTION: When stock charts, trading screens, or financial dashboards are detected, do NOT transcribe individual numbers, prices, or ticker values. Instead, describe the visual trends and patterns (e.g., "upward trend", "declining graph", "volatility spike").**
* **Environment & Lighting:** Specify the light source (direction/intensity), shadows, and background depth (e.g., "Blurred greenery in bokeh background").
* **Coordinates:** Note the relative position of key objects (Top-Left, Center, Bottom-Right).

# 2. CROSS-FRAME DELTA ANALYSIS (MOTION)
* **Displacement:** Exactly how far did objects move? (e.g., "Subject moved from Frame Left to Dead Center").
* **State Changes:** Note changes in light, focus, or object status (e.g., "The indicator light turned from Green to Red between Frame 2 and 3").
* **Hidden Data:** What was occluded (hidden) in Frame 1 that is now revealed in Frame 2?

# 3. SEMANTIC CONTEXT
* **The "Why":** Based strictly on the visual evidence, what is the specific objective of this sequence?
* **Technical Specs:** Estimate the focal length (Wide/Telephoto) and camera height (Eye-level, Low-angle, Overhead).

# OUTPUT STRUCTURE
1.  **Summary Table:** A high-level list of all identified objects and text.
2.  **Detailed Breakdown:** A frame-by-frame bulleted list of the audit points above.
3.  **Final Synthesis:** A one-paragraph conclusion on the "story" the frames tell.

# CONSTRAINT
Do not use "I see" or "The image shows." Start sentences directly with the data. If a detail is unclear, describe its "visual signature" (e.g., "unidentifiable silver rectangular object") rather than skipping it.
"""


def extract_frames_from_video(
    video_path: Path, num_frames: int = 10, save_frames: bool = True
) -> list[str]:
    """Extract evenly spaced frames from video and convert to base64 data URIs.

    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract
        save_frames: Whether to save frames as JPEG files for debugging

    Returns:
        List of base64-encoded data URIs (data:image/jpeg;base64,{base64})
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        raise ValueError(f"Could not read frames from {video_path}")

    # Create frames directory if saving is enabled
    frames_dir = None
    if save_frames:
        frames_dir = VIDEOS_DIR / "frames" / video_path.stem
        frames_dir.mkdir(parents=True, exist_ok=True)

    # Calculate frame indices to extract (evenly spaced)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames_base64 = []

    for frame_num, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)

        # Convert to JPEG bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=70)
        img_bytes = buffer.getvalue()

        # Save frame to disk if enabled
        if save_frames and frames_dir:
            frame_path = frames_dir / f"frame_{frame_num:03d}.jpg"
            pil_image.save(frame_path, format="JPEG", quality=70)

        # Encode to base64
        base64_str = base64.b64encode(img_bytes).decode("utf-8")

        # Create data URI
        data_uri = f"data:image/jpeg;base64,{base64_str}"
        frames_base64.append(data_uri)

    cap.release()

    return frames_base64


def get_video_files() -> list[str]:
    """Get list of video filenames from the videos directory."""
    if not VIDEOS_DIR.exists():
        return []
    return sorted([f.name for f in VIDEOS_DIR.glob("*.mp4")])


def process_video(
    client: VideoLLMClient,
    video_filename: str,
    prompt: Optional[str] = None,
) -> VideoAnalysisResult:
    """
    Process a single video by extracting frames and analyzing them.

    Args:
        client: VideoLLMClient instance
        video_filename: Name of the video file
        prompt: Custom prompt (uses DEFAULT_PROMPT if not provided)

    Returns:
        VideoAnalysisResult with the analysis
    """
    video_path = VIDEOS_DIR / video_filename
    analysis_prompt = prompt or DEFAULT_PROMPT

    print(f"\n{'='*80}")
    print(f"Processing: {video_filename}")
    print(f"Path: {video_path}")
    print(f"{'='*80}\n")

    # Extract frames from video
    print(f"Extracting {NUM_FRAMES} frames...")
    frame_data_uris = extract_frames_from_video(video_path, num_frames=NUM_FRAMES)
    print(f"Extracted {len(frame_data_uris)} frames")

    # Analyze frames using video LLM
    print("Sending to LLM for analysis...")
    result = client.invoke_with_media(
        text=analysis_prompt,
        image_blobs=frame_data_uris,
        output_model=VideoAnalysisResult,
    )

    return result


def save_result(
    video_filename: str,
    result: VideoAnalysisResult,
) -> None:
    """Save a single video analysis result to a txt file."""
    # Create analysis directory if it doesn't exist
    analysis_dir = VIDEOS_DIR / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    # Create output filename (replace .mp4 with .txt)
    output_filename = Path(video_filename).stem + ".txt"
    output_path = analysis_dir / output_filename

    # Save content to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.content)

    print(f"Saved to: {output_path}")


def main() -> None:
    """Main processing loop."""
    # Initialize the video client
    print("Loading environment variables...")
    load_dotenv()
    print("Initializing VideoLLMClient...")

    client = VideoLLMClient.from_env()

    # Get list of videos
    video_files = get_video_files()
    if not video_files:
        print("No video files found in the videos directory.")
        return

    print(f"Found {len(video_files)} videos to process.\n")

    # Process each video
    successful = 0
    failed = 0

    for i, video_file in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] ", end="")
        try:
            result = process_video(client, video_file)
            save_result(video_file, result)
            successful += 1
            print(f"✓ Success")
            print(f"Preview: {result.content[:200]}...")
        except Exception as e:
            failed += 1
            print(f"✗ Error: {e}")

    # Print summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total videos: {len(video_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
