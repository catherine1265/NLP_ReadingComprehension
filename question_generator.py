import re
import spacy
import sys
import subprocess

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spacy model...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")
NER_PRIORITY = {
    "PERSON": 1, "NORP": 2,     "ORG": 3,
    "GPE"   : 4, "LOC" : 5,     "FAC": 6,
    "DATE"  : 7, "TIME": 8, "CARDINAL": 9,
    "MONEY" : 10, "QUANTITY": 11
}

NER_TO_QWORD = {
    "PERSON"  : "Who",
    "NORP"    : "Who",
    "ORG"     : "What organization",
    "GPE"     : "Where",
    "LOC"     : "Where",
    "FAC"     : "Where",
    "DATE"    : "When",
    "TIME"    : "When",
    "CARDINAL": "How many",
    "MONEY"   : "How much",
    "QUANTITY": "How much",
}

def split_paragraphs(text):
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]
    return paragraphs

def extract_candidates(paragraph):
    doc  = nlp(paragraph)
    seen = set()
    candidates = []

    for ent in doc.ents:
        if ent.label_ not in NER_PRIORITY:
            continue
        if ent.text.lower() in seen:
            continue
        if ent.label_ in ["CARDINAL", "ORDINAL"] and len(ent.text.split()) == 1:
            continue
        seen.add(ent.text.lower())
        candidates.append({
            "answer"  : ent.text,
            "ner_type": ent.label_,
            "priority": NER_PRIORITY[ent.label_],
            "qword"   : NER_TO_QWORD.get(ent.label_, "What")
        })

    return sorted(candidates, key=lambda x: x["priority"])[:3]

def to_base_form(sentence):
    doc    = nlp(sentence)
    result = []
    for token in doc:
        if token.pos_ == "VERB" and token.tag_ == "VBD":
            result.append(token.lemma_)
        else:
            result.append(token.text)
    return " ".join(result)

def form_question(sentence, answer, qword):
    sentence = re.sub(r"^(One day|Once|Long ago|At that time|In those days)[,\s]+", "", sentence, flags=re.IGNORECASE).strip()
    sentence = to_base_form(sentence)
    blanked = sentence.replace(answer, "").strip()
    blanked = re.sub(r"\s+", " ", blanked)
    blanked = re.sub(r"^[,\.]\s*", "", blanked).strip()
    blanked = blanked.rstrip(".")

    if qword == "Who":
        return f"Who {blanked}?"
    elif qword == "Where":
        blanked = re.sub(r"\s+in\s*$|\s+at\s*$|\s+to\s*$", "", blanked).strip()
        return f"Where did {blanked}?"
    elif qword == "When":
        blanked = re.sub(r"\s+in\s*$|\s+at\s*$|\s+on\s*$", "", blanked).strip()
        return f"When did {blanked}?"
    elif qword == "What organization":
        return f"What organization {blanked}?"
    elif qword == "How many":
        return f"How many {blanked}?"
    elif qword == "How much":
        return f"How much {blanked}?"
    return f"What {blanked}?"

def generate_answer_key(passage):
    paragraphs   = split_paragraphs(passage)
    result       = []
    used_answers = set()

    for i, para in enumerate(paragraphs):
        candidates  = extract_candidates(para)
        used_qwords = set()
        doc         = nlp(para)
        sentences   = [s.text.strip() for s in doc.sents]

        for c in candidates:
            if c["qword"] in used_qwords:
                continue
            if c["answer"].lower() in used_answers:
                continue

            source = next(
                (s for s in sentences if c["answer"].lower() in s.lower()),
                sentences[0]
            )

            question = form_question(source, c["answer"], c["qword"])

            result.append({
                "paragraph_index": i + 1,
                "paragraph"      : para,
                "question"       : question,
                "answer"         : c["answer"],
                "ner_type"       : c["ner_type"]
            })
            used_qwords.add(c["qword"])
            used_answers.add(c["answer"].lower())

    return result
