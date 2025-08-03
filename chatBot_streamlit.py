import streamlit as st
from transformers import pipeline
from PIL import Image

# --- Streamlit page configuration: অবশ্যই প্রথমেই থাকতে হবে ---
st.set_page_config(
    page_title="Emotion-Aware Chatbot",
    page_icon="🤖",
    layout="centered"
)

# --- Emotion detection model load with cache ---
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

emotion_pipeline = load_model()

# --- Function to detect emotion ---
def detect_emotion(text):
    try:
        results = emotion_pipeline(text)
        emotions = {res['label']: res['score'] for res in results[0]}
        primary_emotion = max(emotions, key=emotions.get)
        return primary_emotion, emotions[primary_emotion]
    except Exception:
        return "neutral", 0.0

# --- Title and Intro ---
st.title("🤖 Emotion-Aware Chatbot")
st.write("I can understand your emotions and chat with you. Please type your message below.")

# --- Project Description (shown in UI) ---
st.markdown("""
### 📄 Project Description

**Emotion-Aware Chatbot** is an intelligent conversational agent that can detect human emotions from text in real-time.  
It uses a pre-trained transformer model to classify emotions such as **joy, anger, sadness, fear, surprise**, and **neutral**.

Based on the detected emotion, it responds empathetically to enhance user interaction. This chatbot has applications in:
- 💬 Mental health support  
- 🛎️ Emotion-aware customer service  
- 🤖 Human-computer interaction systems  

**Key Features:**
- ✅ Real-time emotion detection
- ✅ Emotion-based chatbot responses
- ✅ Simple and interactive Streamlit UI
""")

# --- Display Creator Image ---
try:
    image = Image.open("junayed.jpeg")
    st.image(image, caption="Md. Junayed Bin Karim", width=150)
except FileNotFoundError:
    st.warning("Creator image not found. Please add 'junayed.jpeg' to the app directory.")

# --- Creator Info ---
st.markdown("""
### 👤 Creator Profile  
**Md. Junayed Bin Karim**  
🎓 Computer Science & Engineering (CSE), Daffodil International University  

🔗 Connect with me:  
- GitHub: [Junayed-Bin-Karim](https://github.com/Junayed-Bin-Karim)  
- LinkedIn: [Junayed Bin Karim](https://www.linkedin.com/in/junayed-bin-karim-47b755270/)  

💡 Passionate about AI, Machine Learning, and Software Development.  
🚀 Building innovative projects to solve real-world problems.
""")

st.markdown("---")

# --- User input box ---
user_input = st.text_input("✉️ Enter your message:")

# --- Button & response section ---
if st.button("Send"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a message!")
    else:
        emotion, confidence = detect_emotion(user_input)
        confidence_percent = confidence * 100

        # Predefined responses
        responses = {
            "joy": "Glad to hear you're feeling happy! 😊",
            "anger": "I’m here to listen if something’s bothering you.",
            "sadness": "I'm sorry you're feeling sad. I'm here for you.",
            "fear": "It sounds like something worries you. Feel free to talk about it.",
            "surprise": "Wow! That sounds surprising!",
            "neutral": "Thank you for sharing that with me."
        }

        if confidence < 0.2:
            response = "I'm not sure how you're feeling, but I'm here to chat."
        else:
            response = responses.get(emotion, responses["neutral"])

        st.markdown(f"### 🤔 Detected Emotion: `{emotion.capitalize()}` ({confidence_percent:.2f}%)")
        st.markdown(f"### 🤖 Chatbot: {response}")
