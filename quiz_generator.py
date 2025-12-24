import os
import json
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in .env")

client = OpenAI(api_key=OPENAI_API_KEY)


def generate_questions_from_text(
    text: str,
    num_questions: int = 5,
    difficulty: Optional[str] = "mixed",
) -> List[Dict[str, Any]]:
    """
    Generate multiple-choice questions from text using OpenAI (gpt-4o-mini).

    Strategy:
    - Oversample (ask for more questions than needed),
    - Enforce JSON output (response_format),
    - Filter invalid entries,
    - Then return up to num_questions valid questions.
    """

    # Truncate very long text to keep prompt manageable
    max_chars = 8000
    if len(text) > max_chars:
        text = text[:max_chars]

    # Normalize difficulty
    difficulty = (difficulty or "mixed").lower()
    if difficulty == "easy":
        diff_instruction = (
            "Focus mainly on fundamental concepts and clear understanding. "
            "Questions should be accessible to beginners but still meaningful."
        )
    elif difficulty == "medium":
        diff_instruction = (
            "Include a mix of understanding and application questions. "
            "Some questions can require connecting ideas or interpreting examples."
        )
    elif difficulty == "hard":
        diff_instruction = (
            "Focus on deeper conceptual understanding, application, and reasoning. "
            "Questions can involve multi-step thinking, explaining 'why' something is true, "
            "or evaluating different cases."
        )
    else:  # mixed
        diff_instruction = (
            "Include a natural mix of easy, medium, and hard questions. "
            "At least some questions should require application or reasoning, not just recall."
        )

    system_prompt = (
        "You are an expert teacher and exam setter. "
        "You design rigorous, fair, and conceptually deep multiple-choice questions "
        "for students, based ONLY on the provided study material."
    )

    # Target number of questions requested by caller
    target_questions = max(1, int(num_questions))

    # Oversample: ask the model for more questions than we need,
    # then we will select the first target_questions valid ones.
    overshoot = max(target_questions + 3, target_questions * 2)
    overshoot = min(overshoot, 30)  # absolute safety cap

    user_prompt = f"""
You are given study material (transcript or text) from one or more video lessons.

Your task:
- Write {overshoot} HIGH-QUALITY multiple-choice questions that help students
  truly understand and think about the concepts in the text.

VERY IMPORTANT CONSTRAINTS:

1) CONTENT-BASED, NOT META
   - Questions MUST be about the actual subject matter (concepts, definitions, reasoning,
     examples, procedures) found in the text.
   - DO NOT ask meta questions like:
     - "What is taught in this video?"
     - "What will you learn in this playlist?"
     - "Which topics are covered in this lesson?"
     - "Who is the instructor?" or anything about the structure of the video.
   - Avoid questions that are only about the fact that it is a video or a playlist.

2) CONCEPTUAL AND EDUCATIONAL
   - Prefer questions that test understanding and application, not just word-spotting.
   - Good questions might ask:
     - Why something is true.
     - What would happen if X changes.
     - How to apply a rule or definition to an example.
     - To compare or distinguish between two related ideas.
   - Recall questions (basic definitions) are allowed, but do NOT make them trivial or vague.

3) GROUNDED IN THE TEXT
   - Every question and correct answer MUST be answerable strictly from the given text.
   - Do NOT invent facts that are not implied or stated in the text.
   - Do NOT use outside knowledge beyond what a careful reader could infer.

4) OPTIONS QUALITY
   - Each question must have EXACTLY 4 options.
   - Only ONE option is correct.
   - Wrong options should be plausible but clearly incorrect if you understand the concept.
   - Avoid options like "All of the above" or "None of the above".

5) EXPLANATIONS
   - For each question, provide a short explanation that:
     - Justifies why the correct option is correct.
     - Optionally mentions why a common wrong option is wrong.

6) DIFFICULTY CONTROL
   - Difficulty setting: {difficulty}
   - {diff_instruction}

7) EXACT COUNT
   - You MUST return EXACTLY {overshoot} questions in the JSON.
   - If the study material seems short or repetitive, you must still create {overshoot}
     distinct, non-duplicate questions by focusing on nuances, comparisons,
     typical misconceptions, or implications of the content.

Study material (use ONLY this):

\"\"\"{text}\"\"\"

Return JSON ONLY in this exact structure (no backticks, no extra text):

{{
  "questions": [
    {{
      "question": "string",
      "options": ["option A", "option B", "option C", "option D"],
      "correct_index": 0,
      "explanation": "string"
    }}
  ]
}}
"""

    # Use gpt-4o-mini (cheap) but enforce JSON output
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content

    # Parse JSON
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []

    questions = data.get("questions", [])
    cleaned: List[Dict[str, Any]] = []

    for q in questions:
        if (
            isinstance(q.get("question"), str)
            and isinstance(q.get("options"), list)
            and len(q["options"]) == 4
            and isinstance(q.get("correct_index"), int)
            and 0 <= q["correct_index"] < 4
        ):
            cleaned.append(
                {
                    "question": q["question"],
                    "options": q["options"],
                    "correct_index": q["correct_index"],
                    "explanation": q.get("explanation", ""),
                }
            )

    # If we got at least target_questions valid ones, just return the first target_questions
    if len(cleaned) >= target_questions:
        return cleaned[:target_questions]

    # Otherwise return whatever we got (still could be fewer if the content is very poor)
    return cleaned
