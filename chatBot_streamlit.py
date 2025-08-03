import streamlit as st
from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt
import datetime

# --- Streamlit page configuration ---
st.set_page_config(
    page_title="Emotion-Aware Chatbot",
    layout="centered"
)

# --- Load Emotion Detection Model ---
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

emotion_pipeline = load_model()

# --- Detect Emotion ---
def detect_emotion(text):
    try:
        results = emotion_pipeline(text)
        emotions = {res['label']: res['score'] for res in results[0]}
        primary_emotion = max(emotions, key=emotions.get)
        return primary_emotion, emotions
    except Exception:
        return "neutral", {}



import matplotlib.pyplot as plt

def show_emotion_chart(emotions_dict):
    st.subheader("Emotion Scores (Pie Chart)")
    labels = list(emotions_dict.keys())
    scores = [score * 100 for score in emotions_dict.values()]

    fig, ax = plt.subplots()
    ax.pie(
        scores,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=plt.cm.Pastel1.colors
    )
    ax.axis('equal')  # Equal aspect ratio ensures pie is circular.
    st.pyplot(fig)


# --- Title and Description ---
st.title(" Emotion-Aware Chatbot")
st.write("I can understand your emotions and chat with you. Please type your message below.")





# --- Creator Info ---
st.markdown("""
### 👤 Creator Profile  
**Md. Junayed Bin Karim**  
🎓 Computer Science & Engineering (CSE), Daffodil International University  



""")

st.markdown("---")

# --- User Input ---
user_input = st.text_input("✉️ Enter your message:")

# --- Session State for History ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Sample Messages ---
with st.expander(" Try Sample Messages"):
    col1, col2 = st.columns(2)
    if col1.button("Try Joy"):
        user_input = "I'm feeling so excited today!"
    if col2.button("Try Sadness"):
        user_input = "I feel very lonely and depressed."

# --- Process Input and Respond ---
if st.button("Send"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a message!")
    else:
        emotion, emotion_scores = detect_emotion(user_input)
        confidence = emotion_scores.get(emotion, 0.0)
        confidence_percent = confidence * 100

        # Save to history
        st.session_state.history.append((user_input, emotion))

        # Predefined Responses
        responses = {
            "joy": "Glad to hear you're feeling happy! 😊",
            "anger": "I’m here to listen if something’s bothering you.",
            "sadness": "I'm sorry you're feeling sad. I'm here for you.",
            "fear": "It sounds like something worries you. Feel free to talk about it.",
            "surprise": "Wow! That sounds surprising!",
            "neutral": "Thank you for sharing that with me."
        }

        response = responses.get(emotion, responses["neutral"]) if confidence >= 0.2 else "I'm not sure how you're feeling, but I'm here to chat."

        # Display
        st.markdown(f"###  Detected Emotion: `{emotion.capitalize()}` ({confidence_percent:.2f}%)")
        st.markdown(f"###  Chatbot: {response}")
        show_emotion_chart(emotion_scores)

# --- Conversation History ---
with st.expander("🕓 Conversation History"):
    for i, (msg, emo) in enumerate(st.session_state.history, 1):
        st.markdown(f"**{i}.** _You:_ {msg}  →  _Detected:_ `{emo}`")

# --- Download Report ---
if st.button("📄 Download Emotion Report"):
    report = "\n".join(
        [f"{i+1}. Message: {msg} | Emotion: {emo}" for i, (msg, emo) in enumerate(st.session_state.history)]
    )
    st.download_button("Download as TXT", report, file_name=f"emotion_report_{datetime.date.today()}.txt")

