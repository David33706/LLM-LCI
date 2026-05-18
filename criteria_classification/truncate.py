import re

def clean_and_truncate_pdf_text(text: str, tokenizer, max_tokens: int) -> str:
    """
    Cleans raw PDF text by preserving structural headers, removing noise,
    stripping the references section, and safely truncating to a token budget.
    """
    if text is None:
        return ""

    # Ensure downstream tokenizer calls always receive a string.
    text = str(text)
    if not text.strip():
        return ""

    # 1. Remove non-ASCII/special characters but preserve basic punctuation and newlines
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # 2. Strip References/Acknowledgments to save token space
    # Matches "\n 8. References \n", "\n References \n", "\n Acknowledgements \n", etc.
    tail_pattern = r'\n\s*(?:[0-9]{1,2}[\.\s]*)?(?:References|Bibliography|Acknowledg(?:e?)ments|Literature Cited|Declaration of Competing Interest|Data Availability)\s*\n.*$'
    text = re.sub(tail_pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    # 3. Normalize whitespace but keep structural paragraph breaks
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    # 4. Enforce Token Budget.
    # Gemma-4 may pass an AutoProcessor; use its text tokenizer when available.
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    tokens = text_tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(tokens) <= max_tokens:
        return text

    # Fast truncation: slice tokens and decode back to string
    return text_tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)