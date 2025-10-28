import json
import os
import re
from fastapi import HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from utils.text_cleaner import clean_text
from config import API_KEY, MODEL

# Load prompts from JSON
PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompts.json")
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    PROMPTS = json.load(f)

# Initialize Gemini
chat = ChatGoogleGenerativeAI(
    model=MODEL,
    api_key=API_KEY,
    temperature=0.2,
)

def extract_job_info_gemini(jd_text: str, prompt_name: str = "jd_parser"):
    """Parse JD text using Gemini and return structured JSON."""
    jd_text = clean_text(jd_text)
    if prompt_name not in PROMPTS:
        raise ValueError(f"Prompt '{prompt_name}' not found in PROMPTS.")

    prompt_template = PROMPTS[prompt_name]
    prompt = prompt_template.format(jd_text=jd_text)

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
