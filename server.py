"""FastAPI backend — Your Learning Buddy: curriculum upload + AI question
generation with grade detection and per-subject difficulty (Gemini 2.5 Flash
via Emergent universal LLM key)."""

from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    FastAPI,
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

import asyncio
import json
import logging
import os
import re
import tempfile
import uuid

from emergentintegrations.llm.chat import (
    FileContentWithMimeType,
    LlmChat,
    UserMessage,
)


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")


MONGO_URL = os.environ["MONGO_URL"]
DB_NAME = os.environ.get("DB_NAME", "your_learning_buddy")
EMERGENT_LLM_KEY = os.environ.get("EMERGENT_LLM_KEY")

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

app = FastAPI(title="Your Learning Buddy API")
api_router = APIRouter(prefix="/api")


Difficulty = Literal["easy", "medium", "hard", "ap"]

INITIAL_BATCH = 30
GEN_BATCH_SIZE = 40
MAX_TARGET = 3000
MIN_TARGET = 10
MAX_FAILED_BATCHES = 4

SUPPORTED_MIME = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "txt": "text/plain",
}

SUBJECT_TAGS = ["hindi", "math", "time", "science", "geography", "english", "other"]

GRADE_AGE = {
    1: 6, 2: 7, 3: 8, 4: 9, 5: 10, 6: 11, 7: 12, 8: 13, 9: 14, 10: 15, 11: 16, 12: 17,
}


class CurriculumOut(BaseModel):
    id: str
    file_name: str
    topics: List[str]
    grade: int
    pool_size: int
    target_size: int
    status: str


class GeneratedQuestion(BaseModel):
    id: str
    subject: str
    prompt: str
    sub_prompt: Optional[str] = Field(default=None)
    options: List[str]
    answer_index: int


class QuestionPoolOut(BaseModel):
    questions: List[GeneratedQuestion]


class CurriculumStatusOut(BaseModel):
    id: str
    file_name: str
    topics: List[str]
    grade: int
    pool_size: int
    target_size: int
    status: str


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(.+?)```", text, flags=re.DOTALL)
    candidate = m.group(1).strip() if m else text.strip()
    try:
        return json.loads(candidate)
    except Exception:
        first = candidate.find("{")
        last = candidate.rfind("}")
        if first == -1 or last == -1 or last <= first:
            return None
        try:
            return json.loads(candidate[first : last + 1])
        except Exception:
            return None


async def _gemini_chat(session_id: str, system_message: str) -> LlmChat:
    if not EMERGENT_LLM_KEY:
        raise HTTPException(500, "EMERGENT_LLM_KEY not set on server")
    return LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=session_id,
        system_message=system_message,
    ).with_model("gemini", "gemini-2.5-flash")


@api_router.get("/")
async def root():
    return {"app": "Your Learning Buddy", "status": "ok"}


def _parse_diff_map(difficulties_json: Optional[str], single: Optional[str]) -> dict:
    base: dict = {s: "medium" for s in ("hindi", "math", "time", "science", "geography", "english")}
    if difficulties_json:
        try:
            obj = json.loads(difficulties_json)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in base and isinstance(v, str) and v in ("easy", "medium", "hard", "ap"):
                        base[k] = v
        except Exception:
            logging.warning("Bad difficulties JSON, falling back to single")
    if single and single in ("easy", "medium", "hard", "ap"):
        if not difficulties_json:
            for k in base:
                base[k] = single
    return base


@api_router.post("/curriculum/upload", response_model=CurriculumOut)
async def upload_curriculum(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    difficulty: Difficulty = Form("medium"),
    difficulties: Optional[str] = Form(None),
    pool_size: int = Form(1000),
):
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED_MIME:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(SUPPORTED_MIME.keys())}")
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    if len(raw) > 8 * 1024 * 1024:
        raise HTTPException(400, "File too large (>8MB)")
    target = max(MIN_TARGET, min(MAX_TARGET, int(pool_size)))
    diff_map = _parse_diff_map(difficulties, difficulty)

    suffix = "." + ext
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    curriculum_id = str(uuid.uuid4())

    try:
        meta = await _extract_curriculum_meta(tmp_path, ext, file.filename or "curriculum")
        topics: List[str] = meta["topics"]
        grade: int = meta["grade"]
        initial_count = min(INITIAL_BATCH, target)
        first_batch = await _generate_questions_batch(topics, diff_map, grade, initial_count, 0)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not first_batch:
        raise HTTPException(502, "AI returned no usable questions for initial batch")

    status = "ready" if len(first_batch) >= target else "generating"
    doc = {
        "id": curriculum_id,
        "file_name": file.filename or "curriculum",
        "topics": topics,
        "grade": grade,
        "difficulty_map": diff_map,
        "target_size": target,
        "status": status,
        "questions": [q.dict() for q in first_batch],
    }
    await db.curricula.insert_one(doc)

    if status == "generating":
        background_tasks.add_task(
            _fill_curriculum_in_background,
            curriculum_id=curriculum_id,
            topics=topics,
            diff_map=diff_map,
            grade=grade,
            target=target,
        )

    return CurriculumOut(
        id=curriculum_id, file_name=doc["file_name"], topics=topics, grade=grade,
        pool_size=len(first_batch), target_size=target, status=status,
    )


@api_router.get("/curriculum/{curriculum_id}/status", response_model=CurriculumStatusOut)
async def curriculum_status(curriculum_id: str):
    doc = await db.curricula.find_one({"id": curriculum_id}, {"_id": 0})
    if not doc:
        raise HTTPException(404, "Curriculum not found")
    return CurriculumStatusOut(
        id=doc["id"], file_name=doc.get("file_name", "curriculum"),
        topics=doc.get("topics", []), grade=int(doc.get("grade", 3)),
        pool_size=len(doc.get("questions", [])),
        target_size=int(doc.get("target_size", len(doc.get("questions", [])))),
        status=doc.get("status", "ready"),
    )


@api_router.get("/curriculum/{curriculum_id}/questions", response_model=QuestionPoolOut)
async def get_curriculum_questions(curriculum_id: str):
    doc = await db.curricula.find_one({"id": curriculum_id}, {"_id": 0})
    if not doc:
        raise HTTPException(404, "Curriculum not found")
    return QuestionPoolOut(questions=[GeneratedQuestion(**q) for q in doc.get("questions", [])])


@api_router.post("/curriculum/{curriculum_id}/regenerate", response_model=CurriculumStatusOut)
async def regenerate_curriculum_questions(
    curriculum_id: str,
    background_tasks: BackgroundTasks,
    difficulty: Difficulty = Form("medium"),
    difficulties: Optional[str] = Form(None),
    pool_size: int = Form(1000),
):
    doc = await db.curricula.find_one({"id": curriculum_id}, {"_id": 0})
    if not doc:
        raise HTTPException(404, "Curriculum not found")
    target = max(MIN_TARGET, min(MAX_TARGET, int(pool_size)))
    diff_map = _parse_diff_map(difficulties, difficulty)
    grade = int(doc.get("grade", 3))

    # Existing pool — we APPEND to this rather than replace, so each tap of
    # "Update & Regenerate" GROWS the variety instead of starting over.
    existing_questions: List[dict] = list(doc.get("questions") or [])
    existing_keys: set = {
        _q_dedup_key(q.get("prompt") or "", q.get("options") or [])
        for q in existing_questions
    }
    anti_examples = _build_anti_examples(existing_questions)

    first_count = min(INITIAL_BATCH, max(0, target - len(existing_questions)) or INITIAL_BATCH)
    first_batch = await _generate_questions_batch(
        doc["topics"], diff_map, grade, first_count, 0, anti_examples=anti_examples,
    )
    if not first_batch and not existing_questions:
        raise HTTPException(502, "AI returned no usable questions")

    # Dedupe newly returned batch against the existing pool.
    fresh = []
    for q in first_batch:
        k = _q_dedup_key(q.prompt, q.options)
        if k in existing_keys:
            continue
        existing_keys.add(k)
        fresh.append(q)

    merged = existing_questions + [q.dict() for q in fresh]
    # Cap at target so the pool doesn't grow unbounded across many taps.
    new_target = max(target, len(merged))
    new_target = min(new_target, MAX_TARGET)
    status = "ready" if len(merged) >= new_target else "generating"

    await db.curricula.update_one(
        {"id": curriculum_id},
        {"$set": {"difficulty_map": diff_map, "target_size": new_target,
                  "status": status, "questions": merged}},
    )
    if status == "generating":
        background_tasks.add_task(
            _fill_curriculum_in_background,
            curriculum_id=curriculum_id, topics=doc["topics"], diff_map=diff_map,
            grade=grade, target=new_target,
        )
    return CurriculumStatusOut(
        id=curriculum_id, file_name=doc.get("file_name", "curriculum"),
        topics=doc["topics"], grade=grade, pool_size=len(merged),
        target_size=new_target, status=status,
    )


def _build_anti_examples(existing_questions: List[dict], k: int = 30) -> List[str]:
    """Take a random sample of existing prompts and return them as short
    anti-example strings to feed to Gemini so it doesn't regenerate them."""
    if not existing_questions:
        return []
    import random
    sample = random.sample(existing_questions, min(k, len(existing_questions)))
    out: List[str] = []
    for q in sample:
        p = (q.get("prompt") or "").strip()
        if not p:
            continue
        out.append(p[:120])
    return out


@api_router.delete("/curriculum/{curriculum_id}")
async def delete_curriculum(curriculum_id: str):
    await db.curricula.delete_one({"id": curriculum_id})
    return {"ok": True}


@api_router.post("/curriculum/{curriculum_id}/deduplicate")
async def deduplicate_curriculum_pool(curriculum_id: str):
    """One-shot cleanup: runs the composite-key dedup across the existing pool
    and drops near-paraphrase duplicates that the OLD generator let through.
    Returns counts so the frontend can show 'cleaned X -> Y'."""
    doc = await db.curricula.find_one({"id": curriculum_id}, {"_id": 0})
    if not doc:
        raise HTTPException(404, "Curriculum not found")
    questions = list(doc.get("questions") or [])
    before = len(questions)
    seen_keys: set = set()
    cleaned: List[dict] = []
    for q in questions:
        k = _q_dedup_key(q.get("prompt") or "", q.get("options") or [])
        if not k or k in seen_keys:
            continue
        seen_keys.add(k)
        cleaned.append(q)
    after = len(cleaned)
    await db.curricula.update_one(
        {"id": curriculum_id},
        {"$set": {"questions": cleaned}},
    )
    return {"id": curriculum_id, "before": before, "after": after, "removed": before - after}


TOPIC_EXTRACTION_PROMPT = (
    "You are reading a school curriculum document. "
    "1) Detect the target GRADE LEVEL (integer 1..12). PYP X = grade X. If unclear default 3. "
    "2) Extract 6-14 study TOPICS (short noun phrases, max 6 words each). "
    "Return STRICTLY: {\"grade\": <int>, \"topics\": [\"topic 1\", ...]}. No prose."
)


async def _extract_curriculum_meta(file_path: str, ext: str, file_name: str) -> dict:
    chat = await _gemini_chat(
        session_id=f"meta-{uuid.uuid4()}",
        system_message="You extract grade level and study topics from textbook content.",
    )
    file_content = FileContentWithMimeType(file_path=file_path, mime_type=SUPPORTED_MIME[ext])
    msg = UserMessage(
        text=TOPIC_EXTRACTION_PROMPT + f"\n\nThe curriculum file is named: '{file_name}'.",
        file_contents=[file_content],
    )
    try:
        resp = await chat.send_message(msg)
    except Exception as e:
        logging.exception("Curriculum meta extraction failed")
        raise HTTPException(502, f"Could not read curriculum: {e}")
    parsed = _extract_json(resp) or {}
    topics_raw = parsed.get("topics")
    grade_raw = parsed.get("grade")
    if not isinstance(topics_raw, list) or not topics_raw:
        raise HTTPException(502, "AI could not extract topics from this file.")
    clean: List[str] = []
    seen = set()
    for t in topics_raw:
        if not isinstance(t, str):
            continue
        s = t.strip()
        if not s or s.lower() in seen:
            continue
        seen.add(s.lower())
        clean.append(s[:80])
        if len(clean) >= 14:
            break
    if not clean:
        raise HTTPException(502, "AI returned no usable topics")
    grade = 3
    try:
        if isinstance(grade_raw, (int, float)):
            grade = max(1, min(12, int(grade_raw)))
        elif isinstance(grade_raw, str):
            digits = re.search(r"\d+", grade_raw)
            if digits:
                grade = max(1, min(12, int(digits.group(0))))
    except Exception:
        grade = 3
    return {"grade": grade, "topics": clean}


QUESTION_GEN_SCHEMA = (
    "Each question MUST be a JSON object with these fields: "
    "id (string, unique), "
    "subject (MUST be one of EXACTLY: 'hindi','math','time','science','geography','english','other'), "
    "topic (specific topic, max 6 words), "
    "prompt (the question), "
    "sub_prompt (optional hint), "
    "options (array of EXACTLY 4 non-empty strings), "
    "answer_index (integer 0..3). "
    "Return: {\"questions\": [ ... ]}. No prose, no fences. "
    "Subject classification: 'math' for numerical/geometry/fractions/measurement/data. "
    "'time' only for clock/time-interval (subset of math). "
    "'hindi' for any Hindi (Devanagari) language item. "
    "'english' for English vocab/grammar/comprehension. "
    "'science' for living things/energy/changes/body/plants/animals/cycles. "
    "'geography' for maps/places/history/communities/culture/civics. "
    "Use 'other' only as last resort."
)


def _difficulty_brief(diff: str, grade: int = 3) -> str:
    age = GRADE_AGE.get(grade, 8)
    if diff == "easy":
        return (
            f"EASY level for a Grade {grade} student (~age {age}): direct recall and "
            f"single-step problems within Grade {grade}. Vocabulary entry-level."
        )
    if diff == "hard":
        next_g = min(12, grade + 1)
        return (
            f"HARD level — TOUGH at the top of Grade {grade}, stretching to Grade {next_g}. "
            f"Every question requires AT LEAST TWO steps of reasoning. Use larger numbers, "
            f"reverse problems, word problems with extra info, unit conversions, comparative "
            f"reasoning. Distractors must be common student mistakes. Reject any 1-step question."
        )
    if diff == "ap":
        ap_grade = min(12, grade + 2)
        return (
            f"AP/OLYMPIAD level — ELITE challenge at Grade {ap_grade} depth for a top "
            f"Grade {grade} student. Every question needs THREE or more steps OR a clever "
            f"insight (pattern, working backwards, casework). Use multi-step word problems, "
            f"fractions/decimals, area/perimeter, percent, simple algebra with placeholders. "
            f"Science = cause-and-effect chains. English = comprehension/inference. "
            f"All 4 options must be tempting distractors. NO 1-step recall. "
            f"If a smart Grade {grade} kid solves in <30s without writing, REPLACE IT."
        )
    return (
        f"MEDIUM level for Grade {grade}: single or simple two-step on core Grade {grade} content."
    )


SUBJECT_DISPLAY = {
    "hindi": "Hindi", "math": "Math", "time": "Time", "science": "Science",
    "geography": "Geography", "english": "English",
}


def _diff_map_brief(diff_map: dict, grade: int) -> str:
    lines = ["PER-SUBJECT DIFFICULTY (apply the matching level by question subject):"]
    for sub, level in diff_map.items():
        lines.append(f"- {SUBJECT_DISPLAY.get(sub, sub)} ({sub}): {_difficulty_brief(level, grade)}")
    return "\n".join(lines)


def _normalize_subject(raw: object) -> str:
    if not isinstance(raw, str):
        return "other"
    s = raw.strip().lower()
    if s in SUBJECT_TAGS:
        return s
    if any(k in s for k in ("hindi", "देव", "वर्ण")):
        return "hindi"
    if any(k in s for k in ("clock", "time")):
        return "time"
    if any(k in s for k in ("math", "number", "fraction", "geometry", "shape", "area", "perimeter", "money", "measure", "data", "graph", "addition", "subtraction", "multiplication", "division")):
        return "math"
    if any(k in s for k in ("science", "energy", "plant", "animal", "ecosystem", "body", "water", "weather", "physics", "chemical", "biology", "living")):
        return "science"
    if any(k in s for k in ("geography", "social", "map", "place", "country", "state", "city", "history", "community", "culture", "civics")):
        return "geography"
    if any(k in s for k in ("english", "grammar", "vocab", "reading", "writing", "spelling", "language")):
        return "english"
    return "other"


def _coerce_question(raw_q: dict, batch_idx: int, item_idx: int) -> Optional[GeneratedQuestion]:
    if not isinstance(raw_q, dict):
        return None
    opts = raw_q.get("options")
    ai = raw_q.get("answer_index")
    prompt_text = raw_q.get("prompt")
    if (not isinstance(opts, list) or len(opts) != 4
        or not all(isinstance(o, str) and o.strip() for o in opts)
        or not isinstance(ai, int) or ai < 0 or ai > 3
        or not isinstance(prompt_text, str) or not prompt_text.strip()):
        return None
    subject = _normalize_subject(raw_q.get("subject"))
    topic_text = raw_q.get("topic") or raw_q.get("sub_prompt")
    return GeneratedQuestion(
        id=str(raw_q.get("id") or f"ai-{batch_idx}-{item_idx}-{uuid.uuid4().hex[:6]}"),
        subject=subject, prompt=prompt_text.strip()[:240],
        sub_prompt=(str(topic_text).strip()[:120] if topic_text else None),
        options=[o.strip()[:60] for o in opts], answer_index=ai,
    )


async def _generate_questions_batch(
    topics: List[str], diff_map: dict, grade: int, batch_size: int, batch_idx: int,
    anti_examples: Optional[List[str]] = None,
) -> List[GeneratedQuestion]:
    batch_size = max(1, min(80, batch_size))
    chat = await _gemini_chat(
        session_id=f"qgen-{uuid.uuid4()}",
        system_message=(
            "You write engaging, age-appropriate multiple-choice questions for school children. "
            "Output strictly valid JSON. Do NOT repeat questions from earlier batches."
        ),
    )

    # Honour the user's preference: if every subject in the diff_map is set to
    # hard/ap, the prompt explicitly forbids easy/medium output. Many users
    # (this one in particular) "almost never use easy" — defend against Gemini
    # silently watering down the level.
    levels = set(diff_map.values()) if diff_map else set()
    only_advanced = levels.issubset({"hard", "ap"}) and len(levels) > 0
    advanced_clause = (
        "STRICT DIFFICULTY FLOOR: every question MUST be hard or AP-level. "
        "If you find yourself writing a 1-step recall question, DELETE IT and replace it with a "
        "multi-step reasoning, inference, word-problem, or pattern question. "
        if only_advanced else ""
    )

    anti_clause = ""
    if anti_examples:
        sample = "\n".join(f"- {p}" for p in anti_examples[:30])
        anti_clause = (
            "DO NOT regenerate any of these existing questions (or close paraphrases of them):\n"
            f"{sample}\n"
            "Produce questions on DIFFERENT angles, different numbers, different scenarios.\n"
        )

    prompt = (
        f"Generate exactly {batch_size} multiple-choice questions (batch #{batch_idx + 1}), "
        f"appropriate for Grade {grade}, spread across these {len(topics)} topics: {topics}. "
        f"{_diff_map_brief(diff_map, grade)}\n"
        f"{advanced_clause}"
        f"{anti_clause}"
        "VARIETY IS CRITICAL. Every single question MUST differ from every other question in this batch in BOTH the prompt wording AND the underlying concept being tested. "
        "Use a wide MIX of question STYLES: direct question, riddle, real-life word problem with Indian names (Aarav/Diya/Rishi/Meera/Aanya), scenario with a hint, fill-in-the-blank, 'which is true', 'which is FALSE', 'odd-one-out', 'next in pattern'. "
        "Across the batch, no two questions should test the same exact fact or use the same numbers. Rotate which topic comes next so the batch doesn't cluster on one topic. "
        "Subject diversity: include questions across math, time, hindi, english, science, geography. "
        "No 70%+ single subject. "
        "All distractors must be plausible common student mistakes. "
        "No duplicate prompts. No prompts that paraphrase each other. Keep prompts <35 words. Keep options <8 words. "
        "DIFFICULTY SELF-AUDIT: before finalising, ask 'Is this actually hard enough for its level? If a strong student solves in <30s without writing, REPLACE.' "
        "DUPLICATE SELF-AUDIT: before returning, scan your own list — if any two prompts test the same concept or share >50% of the same words, REPLACE the later one. "
        f"{QUESTION_GEN_SCHEMA}"
    )
    try:
        resp = await chat.send_message(UserMessage(text=prompt))
    except Exception:
        logging.exception("Question gen batch failed")
        return []
    parsed = _extract_json(resp) or {}
    raw = parsed.get("questions") if isinstance(parsed, dict) else None
    if not isinstance(raw, list):
        return []
    out: List[GeneratedQuestion] = []
    seen_keys: set = set()
    for i, q in enumerate(raw):
        coerced = _coerce_question(q, batch_idx, i)
        if not coerced:
            continue
        key = _q_dedup_key(coerced.prompt, coerced.options)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append(coerced)
    return out


def _q_dedup_key(prompt: str, options: List[str]) -> str:
    """Normalised key used to detect duplicate questions inside and across batches.
    Includes prompt + sorted options so two questions that just shuffle options
    are detected as duplicates, while procedurally-generated questions with the
    same template but different numerics still pass through."""
    p = " ".join((prompt or "").strip().lower().split())
    opts = "|".join(sorted([(o or "").strip().lower() for o in options]))
    return f"{p}||{opts}"


async def _fill_curriculum_in_background(curriculum_id: str, topics: List[str], diff_map: dict, grade: int, target: int) -> None:
    failed = 0
    batch_idx = 1
    try:
        while True:
            doc = await db.curricula.find_one({"id": curriculum_id}, {"questions": 1, "_id": 0})
            if not doc:
                return
            current_questions = doc.get("questions", []) or []
            current = len(current_questions)
            if current >= target:
                await db.curricula.update_one({"id": curriculum_id}, {"$set": {"status": "ready"}})
                return
            need = target - current
            this_batch = min(GEN_BATCH_SIZE, need)
            # Feed each successive batch a fresh sample of the existing pool as
            # "do-not-repeat" anti-examples — keeps the new batch varied.
            anti = _build_anti_examples(current_questions)
            new_qs = await _generate_questions_batch(
                topics, diff_map, grade, this_batch, batch_idx, anti_examples=anti,
            )
            batch_idx += 1
            if not new_qs:
                failed += 1
                if failed >= MAX_FAILED_BATCHES:
                    await db.curricula.update_one({"id": curriculum_id}, {"$set": {"status": "ready"}})
                    return
                await asyncio.sleep(2)
                continue
            # Dedupe by composite key (normalized prompt + sorted options)
            existing_keys = {
                _q_dedup_key(q.get("prompt") or "", q.get("options") or [])
                for q in doc.get("questions", [])
            }
            unique_new = []
            for q in new_qs:
                key = _q_dedup_key(q.prompt, q.options)
                if not key or key in existing_keys:
                    continue
                existing_keys.add(key)
                unique_new.append(q)
            if not unique_new:
                failed += 1
                if failed >= MAX_FAILED_BATCHES:
                    await db.curricula.update_one({"id": curriculum_id}, {"$set": {"status": "ready"}})
                    return
                await asyncio.sleep(2)
                continue
            failed = 0
            await db.curricula.update_one(
                {"id": curriculum_id},
                {"$push": {"questions": {"$each": [q.dict() for q in unique_new]}}},
            )
            logging.info("Curriculum %s: appended %d -> %d/%d", curriculum_id, len(unique_new), current + len(unique_new), target)
            await asyncio.sleep(0.5)
    except Exception:
        logging.exception("Background fill crashed for %s", curriculum_id)
        try:
            await db.curricula.update_one({"id": curriculum_id}, {"$set": {"status": "ready"}})
        except Exception:
            pass


app.include_router(api_router)
app.add_middleware(
    CORSMiddleware, allow_credentials=True, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"],
)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
