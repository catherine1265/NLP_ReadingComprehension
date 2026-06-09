import streamlit as st
from question_generator import generate_answer_key
from evaluator import evaluate_quiz

st.set_page_config(page_title="NLP Auto-Quiz Generator", page_icon="🤖")

st.markdown("""
    <style>
        .stApp { background-color: #0e1117; color: #fafafa; }
        .stTextArea textarea { background-color: #1c1f26; color: #fafafa; border: 1px solid #e63946; }
        .stTextInput input  { background-color: #1c1f26; color: #fafafa; border: 1px solid #e63946; }
        .stButton > button  { background-color: #1c1f26; color: #fafafa; border: 1px solid #fafafa; }
        .stButton > button:hover { background-color: #e63946; border-color: #e63946; }
        .stExpander { background-color: #1c1f26; border: 1px solid #2e3440; }
    </style>
""", unsafe_allow_html=True)

if "questions"    not in st.session_state: st.session_state.questions    = None
if "user_answers" not in st.session_state: st.session_state.user_answers = []
if "submitted"    not in st.session_state: st.session_state.submitted    = False

if st.session_state.questions is None:
    st.title("🤖 NLP Auto-Quiz Generator")
    st.write("Masukkan teks cerita/konteks di sini:")
    passage = st.text_area("", height=150)

    if st.button("Generate Quiz!"):
        if passage.strip():
            with st.spinner("Generating questions..."):
                st.session_state.questions    = generate_answer_key(passage)
                st.session_state.user_answers = [""] * len(st.session_state.questions)
                st.session_state.submitted    = False
            st.rerun()

elif not st.session_state.submitted:
    st.subheader("Jawab Pertanyaan Berikut:")

    user_answers = []
    for i, q in enumerate(st.session_state.questions):
        st.write(f"**Q{i+1}: {q['question']}**")
        ans = st.text_input("", key=f"ans_{i}")
        user_answers.append(ans)

    if st.button("Submit & Score"):
        st.session_state.user_answers = user_answers
        st.session_state.submitted    = True
        st.rerun()

else:
    result = evaluate_quiz(st.session_state.questions, st.session_state.user_answers)

    st.subheader("🏛 Hasil Evaluasi")
    st.write(f"**Score Akhir: {result['score_pct']}% ({result['correct']}/{result['total']} Benar)**")
    st.write("")

    for r in result['results']:
        status = "✅ Benar" if r['is_correct'] else "❌ Salah"
        with st.expander(f"Q{r['no']}: {r['question']} - {status}"):
            st.write(f"**Jawaban Kamu:** {r['user_ans']}")
            st.write(f"**Kunci Jawaban:** {r['correct_ans']}")
            st.write(f"**Similarity Score:** {r['similarity']}")

    if st.button("← Coba Lagi"):
        st.session_state.questions    = None
        st.session_state.user_answers = []
        st.session_state.submitted    = False
        st.rerun()
