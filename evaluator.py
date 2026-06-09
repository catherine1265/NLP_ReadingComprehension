import pickle
import warnings
import numpy as np
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from model_loader import ensure_models

ensure_models()

ner_model = pickle.load(open('model.pkl', 'rb'))

def evaluate_answer(user_answer: str, correct_answer: str) -> dict:
    if not user_answer or not user_answer.strip():
        return {
            'is_correct': False, 'score': 0.0, 'similarity': 0.0,
            'user_ner': 'EMPTY',
            'correct_ner': ner_model.predict([correct_answer])[0],
            'ner_match': False
        }

    user_ner    = ner_model.predict([user_answer])[0]
    correct_ner = ner_model.predict([correct_answer])[0]
    ner_match   = (user_ner == correct_ner)

    # Step 2: Hitung cosine similarity dengan TF-IDF
    tfidf   = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    vectors = tfidf.fit_transform([
        user_answer.lower().strip(),
        correct_answer.lower().strip()
    ])
    similarity = float(cosine_similarity(vectors[0], vectors[1])[0][0])

    # Step 3: Tentukan score & is_correct
    # NER match → minimal score 0.5 (tipe jawaban benar)
    score = similarity
    if ner_match:
        score = max(score, 0.5)
    score = round(score, 4)

    # Benar jika: NER sama ATAU similarity >= 0.5
    is_correct = ner_match or (similarity >= 0.5)

    return {
        'is_correct':  is_correct,
        'score':       score,
        'similarity':  round(similarity, 4),
        'user_ner':    user_ner,
        'correct_ner': correct_ner,
        'ner_match':   ner_match,
    }

def evaluate_quiz(questions: list, user_answers: list) -> dict:
    results = []
    for i, (q, user_ans) in enumerate(zip(questions, user_answers)):
        correct_ans = q.get('answer', '')
        eval_result = evaluate_answer(user_ans, correct_ans)
        results.append({
            'no':          i + 1,
            'question':    q.get('question', ''),
            'correct_ans': correct_ans,
            'user_ans':    user_ans,
            'ner_type':    q.get('ner_type', ''),
            **eval_result
        })

    correct_count = sum(1 for r in results if r['is_correct'])
    avg_score     = float(np.mean([r['score'] for r in results])) if results else 0.0

    return {
        'results':   results,
        'total':     len(results),
        'correct':   correct_count,
        'score_pct': round(correct_count / len(results) * 100, 1) if results else 0,
        'avg_score': round(avg_score, 4),
    }
