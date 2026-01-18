from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.llm.client import LLMClient


class Match(BaseModel):
    """Individual video match entry."""

    filename: str = Field(..., description="Filename of the matched video.")
    score: int = Field(..., description="Relevance score from 1 to 100.")


class VideoMatch(BaseModel):
    """Result of matching videos to a transcript."""

    matches: List[Match] = Field(
        default_factory=list,
        description="List of matched videos with 'filename' and 'score' (1-100).",
    )


def load_video_metas(json_dir: Path) -> List[dict]:
    """Load all video metadata JSON files and add filename field."""
    metas = []
    for json_file in json_dir.glob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            data["filename"] = json_file.name  # Add filename field
            metas.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    return metas


def match_videos_for_transcript(
    transcript: str,
    video_metas: List[dict],
    llm_client: LLMClient,
) -> List[Match]:
    """Use LLM to find best matching videos for the transcript."""
    # Prepare the prompt
    videos_summary = "\n".join(
        [
            f"- {meta['filename']}: Summary: {meta.get('summary_text', '')} | Themes: {', '.join(meta.get('themes', []))} | Actions: {', '.join(meta.get('actions', []))} | Currencies: {', '.join(meta.get('currencies', []))} | Spatial Tags: {', '.join(meta.get('spatial_tags', []))} | Motion: {meta.get('motion_summary', '')} | Semantic Tags: {', '.join(meta.get('semantic_tags', []))}"
            for meta in video_metas
        ]
    )

    prompt = f"""
You are an expert at matching video clips to text scripts/transcripts.

Given the following transcript for a video script, select the best matching video clips from the list below. You can select multiple if they fit well, or none if none match.

Transcript:
{transcript}

Available videos:
{videos_summary}

Output a JSON object with a list of matches, each as a dict with 'filename' and 'score' (1-100, where 100 is perfect match).
Only include videos with score >= 50. Sort by score descending.
"""

    try:
        result = llm_client.invoke(prompt=prompt, output_model=VideoMatch)
        return sorted(result.matches, key=lambda x: x.score, reverse=True)
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return []


def main():
    if len(sys.argv) < 2:
        print("Usage: python match_videos.py 'your transcript here'")
        sys.exit(1)

    transcript = sys.argv[1]

    load_dotenv()
    llm_client = LLMClient.from_env()

    json_dir = Path("videos/analysis/json")
    video_metas = load_video_metas(json_dir)

    if not video_metas:
        print("No video metadata found.")
        return

    matched = match_videos_for_transcript(transcript, video_metas, llm_client)

    print("Matched videos (sorted by relevance):")
    for match in matched:
        print(f"- {match.filename}: Score {match.score}")


if __name__ == "__main__":
    main()
