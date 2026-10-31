"""REQ 5 - Text prompt handling. The prompt is optional: user text > detected label > generic default."""
from .config import DEFAULT_PROMPT, STYLE_PRESETS, SUFFIX


def build_prompt(user_prompt, understanding, style="none", min_conf=0.25):
    """Returns (final_prompt, source_description)."""
    user_prompt = (user_prompt or "").strip()
    if user_prompt:
        base, source = user_prompt, "user"
    else:
        label, conf = (understanding.get("label_top3") or [("", 0.0)])[0]
        guess = understanding.get("classifier_guess")
        if guess and conf < min_conf and guess[1] >= 0.6:       # CNN only helps when CLIP is unsure
            label, conf = guess
        if label and conf >= min_conf:
            base, source = f"a {label}", f"auto ({label}, p={conf})"
        else:
            base, source = DEFAULT_PROMPT, "default"
    parts = [base, STYLE_PRESETS.get(style, "")]
    return ", ".join(p for p in parts if p) + SUFFIX, source
