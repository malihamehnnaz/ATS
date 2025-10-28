from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from parsers.gemini_parser import extract_job_info_gemini

app = FastAPI(title="JD Extractor with Gemini", version="1.0")

@app.post("/extract_jd_text")
async def extract_from_text(
    jd_text: str = Form(...),
    prompt_name: str = Form("jd_parser")
):
    """Extract JD data from text input using Gemini parser."""
    gemini_result = extract_job_info_gemini(jd_text, prompt_name=prompt_name)
    return {"status": "success", "parsed_result": gemini_result}

@app.post("/extract_jd_file")
async def extract_from_file(
    file: UploadFile = File(...),
    prompt_name: str = Form("jd_parser")
):
    """Extract JD data from uploaded text file using Gemini parser."""
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

    gemini_result = extract_job_info_gemini(jd_text, prompt_name=prompt_name)
    return {"status": "success", "filename": file.filename, "parsed_result": gemini_result}
