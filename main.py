from typing import List, Dict, Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from youtube_utils import (
    extract_playlist_id,
    extract_video_id,
    get_playlist_video_ids,
    get_video_transcript,
    get_video_title_description,
)
from quiz_generator import generate_questions_from_text

app = FastAPI(title="YTQuiz - YouTube Playlist & Video Quiz Generator API")

# CORS (open for frontend on Netlify)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict to your Netlify domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Pydantic models ----------

class GenerateQuizRequest(BaseModel):
    playlistUrl: str          # can be playlist URL or single video URL
    questionsPerVideo: int = 2
    maxVideos: int = 3
    difficulty: str | None = "mixed"  # "easy", "medium", "hard", "mixed"


class PublicQuestion(BaseModel):
    id: int
    question: str
    options: List[str]


class GenerateQuizResponse(BaseModel):
    quizId: str
    questions: List[PublicQuestion]


class AnswerItem(BaseModel):
    questionId: int
    selectedIndex: int


class SubmitQuizRequest(BaseModel):
    quizId: str
    answers: List[AnswerItem]


class QuestionResult(BaseModel):
    questionId: int
    correctIndex: int
    selectedIndex: int
    isCorrect: bool
    explanation: str | None = None


class SubmitQuizResponse(BaseModel):
    score: int
    total: int
    percentage: float
    results: List[QuestionResult]


# In-memory store: quizId -> list of full question dicts (with correct answers)
QUIZZES: Dict[str, List[Dict[str, Any]]] = {}


# ---------- Basic health check ----------

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def root():
    return {
        "message": "YTQuiz API is running.",
        "health": "/health",
        "generateQuiz": "/generate-quiz",
        "submitQuiz": "/submit-quiz",
    }


# ---------- API: generate quiz (playlist or single video) ----------

@app.post("/generate-quiz", response_model=GenerateQuizResponse)
def generate_quiz(payload: GenerateQuizRequest):
    """
    Generate quiz questions from a YouTube playlist OR a single video.

    Faster version:
    - Detect whether the URL is a playlist or a single video.
    - Build a list of video_ids accordingly.
    - Collect text (transcript OR title+description) from those videos.
    - Combine into one big text.
    - Call OpenAI ONCE to generate all questions.
    """

    # Clamp user inputs to avoid huge quiz requests
    questions_per_video = max(1, min(payload.questionsPerVideo, 3))  # 1–3
    max_videos = max(1, min(payload.maxVideos, 5))                   # 1–5

    raw_url = (payload.playlistUrl or "").strip()
    if not raw_url:
        raise HTTPException(status_code=400, detail="Please provide a YouTube playlist or video URL.")

    # 1) Decide: playlist OR single video
    playlist_id = extract_playlist_id(raw_url)
    video_ids: List[str] = []

    if playlist_id:
        # Treat as playlist
        video_ids = get_playlist_video_ids(playlist_id, max_videos=max_videos)
        if not video_ids:
            raise HTTPException(status_code=404, detail="No videos found in this playlist.")
    else:
        # Try as single video
        video_id = extract_video_id(raw_url)
        if not video_id:
            raise HTTPException(
                status_code=400,
                detail="Invalid YouTube playlist or video URL. Please check the link."
            )
        video_ids = [video_id]

    # 2) Collect text for each video (transcript OR title+description)
    video_texts: List[str] = []

    for vid in video_ids:
        transcript = get_video_transcript(vid)

        if transcript:
            text = transcript
        else:
            td = get_video_title_description(vid)
            if not td:
                # No transcript and no metadata -> skip this video
                continue

            title, description = td
            parts: List[str] = []
            if title:
                parts.append(f"Video title: {title}")
            if description:
                parts.append("Video description:")
                parts.append(description)
            text = "\n".join(parts).strip()

            if not text:
                continue

        video_texts.append(text)

    if not video_texts:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not gather any usable text from this URL. "
                "The video(s) may have very little transcript or description."
            ),
        )

    # 3) Combine all text into one big block for a single AI call
    combined_text = "\n\n".join(video_texts)

    # Total number of questions = questions_per_video * number of usable videos
    total_questions = questions_per_video * len(video_texts)
    # Global safety cap so it doesn't get huge
    total_questions = max(1, min(total_questions, 20))

    # 4) Call OpenAI ONCE to generate all questions
    questions_raw = generate_questions_from_text(
        combined_text,
        num_questions=total_questions,
        difficulty=payload.difficulty,
    )

    if not questions_raw:
        raise HTTPException(
            status_code=500,
            detail="The AI could not generate questions from this content. Try a different link.",
        )

    # 5) Prepare internal question data
    all_questions: List[Dict[str, Any]] = []
    for idx, q in enumerate(questions_raw):
        all_questions.append(
            {
                "id": idx,
                "question": q["question"],
                "options": q["options"],
                "correct_index": q["correct_index"],
                "explanation": q.get("explanation", ""),
            }
        )

    # 6) Store quiz in memory and return public version (no correct_index)
    quiz_id = str(uuid4())
    QUIZZES[quiz_id] = all_questions

    public_questions = [
        PublicQuestion(
            id=q["id"],
            question=q["question"],
            options=q["options"],
        )
        for q in all_questions
    ]

    return GenerateQuizResponse(quizId=quiz_id, questions=public_questions)


# ---------- API: submit quiz ----------

@app.post("/submit-quiz", response_model=SubmitQuizResponse)
def submit_quiz(payload: SubmitQuizRequest):
    quiz_questions = QUIZZES.get(payload.quizId)
    if quiz_questions is None:
        raise HTTPException(status_code=404, detail="Quiz not found")

    # Map questionId -> question
    question_map = {q["id"]: q for q in quiz_questions}

    correct_count = 0
    results: List[QuestionResult] = []

    for ans in payload.answers:
        q = question_map.get(ans.questionId)
        if not q:
            # Ignore invalid question IDs
            continue

        is_correct = ans.selectedIndex == q["correct_index"]
        if is_correct:
            correct_count += 1

        results.append(
            QuestionResult(
                questionId=q["id"],
                correctIndex=q["correct_index"],
                selectedIndex=ans.selectedIndex,
                isCorrect=is_correct,
                explanation=q.get("explanation", ""),
            )
        )

    total = len(question_map)
    percentage = (correct_count / total * 100) if total else 0.0

    return SubmitQuizResponse(
        score=correct_count,
        total=total,
        percentage=percentage,
        results=results,
    )