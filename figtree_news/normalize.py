import re

_HONORIFICS = re.compile(
    r'^(mr|mrs|ms|dr|prof|sen|rep|gov|gen|col|lt|cpt|maj|capt|sgt|'
    r'ambassador|judge|attorney|sheriff|officer|detective'
    r')\.?\s*',
    re.I,
)


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = _HONORIFICS.sub('', text)
    text = re.sub(r"'s\b", '', text)
    text = re.sub(r"s'\b", 's', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
