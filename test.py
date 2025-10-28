# main.py
import os
import json
import re
import unicodedata
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from sentence_transformers import SentenceTransformer, util

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ----------------------------
# Initialize FastAPI
# ----------------------------
app = FastAPI(title="JD Extractor with Gemini & Embeddings", version="1.3")

# ----------------------------
# Initialize Gemini
# ----------------------------
chat = ChatGoogleGenerativeAI(
    model=MODEL,
    api_key=API_KEY,
    temperature=0.2,
)

# ----------------------------
# Initialize Embedding model
# ----------------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

FIELD_TEMPLATES = {
    "job_title": ["job title, position, role name"],
    "location": ["city, country, office location"],
    "responsibilities": ["tasks, duties, responsibilities of the employee"],
    "required_skills": ["technical skills, programming languages, tools, software knowledge required"],
    "soft_skills": ["teamwork, collaboration, communication, leadership, problem solving"],
    "education": ["degree, graduation, qualification required"],
    "experience_required": ["years of experience, prior work experience"],
    "tools_and_technologies": ["software tools, frameworks, technologies used"],
}

template_embeddings = {k: embedding_model.encode(v, convert_to_tensor=True) for k, v in FIELD_TEMPLATES.items()}

# ----------------------------
# Utility Functions
# ----------------------------
def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)

    def replace_math_bold_char(match):
        char = match.group(0)
        code = ord(char)
        if 0x1D400 <= code <= 0x1D419:
            return chr(code - 0x1D400 + ord('A'))
        elif 0x1D41A <= code <= 0x1D433:
            return chr(code - 0x1D41A + ord('a'))
        elif 0x1D7CE <= code <= 0x1D7D7:
            return chr(code - 0x1D7CE + ord('0'))
        return char

    text = re.sub(r"[\U0001D400-\U0001D7FF]", replace_math_bold_char, text)
    text = re.sub(r"[–—−]", "-", text)
    text = text.replace("\u00A0", " ").replace("\u00AD", "")
    text = "".join(c for c in text if unicodedata.category(c)[0] != "C")
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ----------------------------
# Gemini-based parser
# ----------------------------
def extract_job_info_gemini(jd_text: str):
    jd_text = clean_text(jd_text)
    prompt = f"""
You are a job description parser. Extract the following fields from the JD text and return valid JSON only:

- job_title (string)
- department (string)
- employment_type (string)
- experience_required (string)
- required_skills (list of strings)
- responsibilities (list of strings)
- education (string)
- certifications (list of strings)
- tools_and_technologies (list of strings)
- soft_skills (list of strings)
- location (string)
- keywords (list of strings)

Instructions:
- Use double quotes for keys and string values.
- Represent lists as JSON arrays.
- Do NOT include explanations or extra text.
- If a field is not found, return an empty string or empty list.

Job Description:
\"\"\"{jd_text}\"\"\"
"""
    try:
        response = chat.invoke([HumanMessage(content=prompt)])
        text_output = response.content.strip()
        text_output = re.sub(r"^```(?:json)?\s*", "", text_output)
        text_output = re.sub(r"\s*```$", "", text_output)
        return json.loads(text_output)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Gemini output not valid JSON: {text_output}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini parsing failed: {e}")

# ----------------------------
# Embedding-based parser
# ----------------------------
def extract_job_info_embeddings(jd_text: str, threshold=0.6):
    jd_text = clean_text(jd_text)
    sentences = [s.strip() for s in jd_text.split("\n") if s.strip()]
    sentence_embeddings = embedding_model.encode(sentences, convert_to_tensor=True)
    jd_parsed = {k: [] for k in FIELD_TEMPLATES.keys()}

    for idx, sent_emb in enumerate(sentence_embeddings):
        best_field = None
        best_score = 0
        for field, temp_emb in template_embeddings.items():
            score = util.cos_sim(sent_emb, temp_emb)[0][0].item()
            if score > best_score:
                best_score = score
                best_field = field
        if best_score >= threshold:
            jd_parsed[best_field].append(sentences[idx])

    # Convert single-valued fields
    for f in ["job_title", "location", "education", "experience_required"]:
        jd_parsed[f] = jd_parsed[f][0] if jd_parsed[f] else ""

    return jd_parsed

# ----------------------------
# Comparison function
# ----------------------------
def compare_parsers(gemini_result, embedding_result):
    comparison = []
    fields = set(gemini_result.keys()).union(set(embedding_result.keys()))
    for f in fields:
        comparison.append({
            "field": f,
            "gemini": gemini_result.get(f, ""),
            "embedding": embedding_result.get(f, ""),
            "match": gemini_result.get(f, "") == embedding_result.get(f, "")
        })
    return comparison

# ----------------------------
# FastAPI Endpoints
# ----------------------------
@app.post("/extract_jd_text")
async def extract_from_text(
    jd_text: str = Form(...),
    method: str = Query("gemini", description="Parser method: 'gemini' or 'embedding'")
):
    """Extract JD data from text input using selected method and return comparison."""
    gemini_result = extract_job_info_gemini(jd_text)
    embedding_result = extract_job_info_embeddings(jd_text)

    if method.lower() == "embedding":
        primary_result = embedding_result
    else:
        primary_result = gemini_result

    comparison_table = compare_parsers(gemini_result, embedding_result)

    return {
        "status": "success",
        "method_used": method.lower(),
        "parsed_result": primary_result,
        "gemini_result": gemini_result,
        "embedding_result": embedding_result,
        "comparison": comparison_table
    }

@app.post("/extract_jd_file")
async def extract_from_file(
    file: UploadFile = File(...),
    method: str = Query("gemini", description="Parser method: 'gemini' or 'embedding'")
):
    """Extract JD data from uploaded text file using selected method and return comparison."""
    content_bytes = await file.read()
    encodings_to_try = ["utf-8", "utf-16", "utf-8-sig", "latin-1"]
    jd_text = None
    for enc in encodings_to_try:
        try:
            jd_text = content_bytes.decode(enc)
            break
        except Exception:
            continue
    if not jd_text:
        raise HTTPException(status_code=400, detail="Could not decode file with known encodings.")

    gemini_result = extract_job_info_gemini(jd_text)
    embedding_result = extract_job_info_embeddings(jd_text)

    if method.lower() == "embedding":
        primary_result = embedding_result
    else:
        primary_result = gemini_result

    comparison_table = compare_parsers(gemini_result, embedding_result)

    return {
        "status": "success",
        "filename": file.filename,
        "method_used": method.lower(),
        "parsed_result": primary_result,
        "gemini_result": gemini_result,
        "embedding_result": embedding_result,
        "comparison": comparison_table
    }
