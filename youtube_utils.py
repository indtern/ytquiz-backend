import os
import re
import requests
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    CouldNotRetrieveTranscript,
)

load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


def extract_playlist_id(playlist_url: str) -> Optional[str]:
    """
    Extract playlist ID from a YouTube playlist URL.
    Example: https://www.youtube.com/playlist?list=PL123...
    """
    match = re.search(r"[?&]list=([a-zA-Z0-9_-]+)", playlist_url)
    return match.group(1) if match else None


def get_playlist_video_ids(playlist_id: str, max_videos: int = 5) -> List[str]:
    """
    Use YouTube Data API to get up to max_videos video IDs from a playlist.
    """
    if not YOUTUBE_API_KEY:
        raise RuntimeError("YOUTUBE_API_KEY is not set in .env")

    url = "https://www.googleapis.com/youtube/v3/playlistItems"
    items_per_page = min(max_videos, 50)

    params = {
        "part": "contentDetails",
        "playlistId": playlist_id,
        "maxResults": items_per_page,
        "key": YOUTUBE_API_KEY,
    }

    video_ids: List[str] = []

    while True:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("items", []):
            vid = item["contentDetails"]["videoId"]
            video_ids.append(vid)
            if len(video_ids) >= max_videos:
                return video_ids

        next_page = data.get("nextPageToken")
        if not next_page:
            break
        params["pageToken"] = next_page

    return video_ids


def get_video_transcript(video_id: str, languages: List[str] = ["en"]) -> Optional[str]:
    """
    Fetch transcript text for a video using youtube-transcript-api.
    Returns None if no transcript is available.
    """
    try:
        segments = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        text = " ".join(seg["text"] for seg in segments if seg.get("text"))
        return text.strip() or None
    except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript):
        return None
    except Exception:
        # For MVP, ignore unexpected errors and treat as "no transcript"
        return None


def get_video_title_description(video_id: str) -> Optional[Tuple[str, str]]:
    """
    Fallback: use YouTube Data API to get video title and description
    when no transcript is available.
    """
    if not YOUTUBE_API_KEY:
        raise RuntimeError("YOUTUBE_API_KEY is not set in .env")

    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet",
        "id": video_id,
        "key": YOUTUBE_API_KEY,
    }

    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        if not items:
            return None

        snippet = items[0].get("snippet", {})
        title = snippet.get("title", "").strip()
        description = snippet.get("description", "").strip()

        # If both are empty, treat as no metadata
        if not title and not description:
            return None

        return title, description
    except Exception:
        # In case of API/network issues, just return None
        return None
def extract_video_id(url_or_id: str) -> Optional[str]:
    """
    Extract a YouTube video ID from:
    - https://www.youtube.com/watch?v=VIDEOID
    - https://youtu.be/VIDEOID
    - https://www.youtube.com/shorts/VIDEOID
    Or treat a plain 11-char string as a video ID.
    """
    s = url_or_id.strip()

    # If it's already a bare 11-char ID
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", s):
        return s

    # Look for v=VIDEOID in query string
    match = re.search(r"v=([a-zA-Z0-9_-]{11})", s)
    if match:
        return match.group(1)

    # youtu.be/VIDEOID
    match = re.search(r"youtu\.be/([a-zA-Z0-9_-]{11})", s)
    if match:
        return match.group(1)

    # /shorts/VIDEOID
    match = re.search(r"shorts/([a-zA-Z0-9_-]{11})", s)
    if match:
        return match.group(1)

    return None