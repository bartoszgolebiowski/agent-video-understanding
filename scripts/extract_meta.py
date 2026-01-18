from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import List, Optional, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from src.llm.client import LLMClient


class VideoTheme(str, Enum):
    """Themes for video content matching, used to suggest best matching text/scripts."""

    CRISIS = "crisis"
    SPENDING = "spending"
    WASTE = "waste"
    GROW = "grow"
    RECESSION = "recession"
    WEALTH = "wealth"
    INVESTMENT = "investment"
    ANALYSIS = "analysis"
    TRANSACTION = "transaction"
    COUNTING = "counting"
    SMOKING = "smoking"  # If relevant, e.g., for certain videos


class VideoAnalysisMeta(BaseModel):
    """
    Structured metadata extracted from raw per-video analysis notes.
    Use this model to index, store and match videos against scripts.
    """

    actions: List[str] = Field(
        default_factory=list,
        description="Detected actions / verbs present in the clip (e.g., ['counting','writing','smoking','examining']).",
    )
    currencies: List[str] = Field(
        default_factory=list,
        description="Detected currency types/denominations (e.g., ['USD','EUR','100','50','20']).",
    )

    spatial_tags: List[str] = Field(
        default_factory=list,
        description="Coarse spatial placement tags (e.g., ['center-left','background','right']).",
    )
    motion_summary: Optional[str] = Field(
        default="",
        description="Short human-readable summary of cross-frame motion/deltas.",
    )

    semantic_tags: List[str] = Field(
        default_factory=list,
        description="High-level semantic intent tags (e.g., ['transaction','analysis','wealth_display']).",
    )
    summary_text: str = Field(
        default="",
        description="Concatenated human-readable summary used for embedding and retrieval.",
    )

    themes: List[VideoTheme] = Field(
        default_factory=list,
        description="Detected themes relevant to the video content.",
    )


def extract_video_meta_from_file(
    file_path: Path,
    llm_client: LLMClient,
) -> Optional[VideoAnalysisMeta]:
    """
    Extracts structured metadata from a raw video analysis TXT file using LLM.
    """
    # Read the raw text
    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

    # Load and render the Jinja template
    prompt = f"""
        Analyze the following video analysis notes and extract structured metadata including actions, currencies, spatial tags, motion summary, semantic tags, and a summary text.
        The input is raw text analysis of a video clip.
        
        
        Raw text:
        {raw_text}
        """

    # Call LLM
    try:
        extracted_data = llm_client.invoke(
            prompt=prompt, output_model=VideoAnalysisMeta
        )
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None

    return extracted_data


def main():
    load_dotenv()
    llm_client = LLMClient.from_env()

    # Directory with analysis files
    analysis_dir = Path("videos/analysis")

    # Process all .txt files
    for txt_file in analysis_dir.glob("*.txt"):
        print(f"Processing {txt_file}...")
        meta = extract_video_meta_from_file(txt_file, llm_client)
        if meta:
            # Save to JSON in videos/analysis/json
            json_dir = Path("videos/analysis/json")
            json_dir.mkdir(parents=True, exist_ok=True)
            output_file = json_dir / txt_file.with_suffix(".json").name
            output_file.write_text(meta.model_dump_json(), encoding="utf-8")
            print(f"Saved structured data to {output_file}")
        else:
            print(f"Failed to process {txt_file}")


if __name__ == "__main__":
    main()
