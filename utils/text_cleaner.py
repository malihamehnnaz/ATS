import unicodedata
import re

def clean_text(text: str) -> str:
    """Normalize and clean JD text."""
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
