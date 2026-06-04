import asyncio
import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound


def _extract_video_id(url: str) -> str:
    patterns = [
        r'(?:v=)([a-zA-Z0-9_-]{11})',
        r'(?:youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:shorts/)([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"유효한 YouTube URL이 아닙니다: {url}")


def _get_transcript_sync(video_id: str) -> str:
    ytt = YouTubeTranscriptApi()
    try:
        transcript = ytt.fetch(video_id, languages=['ko'])
        return ' '.join(snippet.text for snippet in transcript)
    except NoTranscriptFound:
        try:
            transcript_list = ytt.list(video_id)
            generated = transcript_list.find_generated_transcript(['ko'])
            fetched = generated.fetch()
            return ' '.join(snippet.text for snippet in fetched)
        except Exception:
            raise RuntimeError("이 영상에 한국어 자막이 없습니다.")
    except TranscriptsDisabled:
        raise RuntimeError("이 영상은 자막이 비활성화되어 있습니다.")


async def get_youtube_transcript(url: str) -> str:
    """YouTube URL에서 한국어 자막을 텍스트로 추출합니다."""
    video_id = _extract_video_id(url)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_transcript_sync, video_id)
