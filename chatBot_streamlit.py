import streamlit as st
from transformers import pipeline
from PIL import Image

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Emotion-Aware Chatbot", layout="centered")

# --- Load Emotion Detection Model ---
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

emotion_pipeline = load_model()

# --- Detect Emotion Function ---
def detect_emotion(text):
    try:
        results = emotion_pipeline(text)
        emotions = {res['label']: res['score'] for res in results[0]}
        primary_emotion = max(emotions, key=emotions.get)
        return primary_emotion, emotions[primary_emotion]
    except Exception:
        return "neutral", 0.0

# --- Title & Intro ---
st.title("🤖 Emotion-Aware Chatbot")
st.write("I can understand your emotions and chat with you. Please type your message below.")

# --- Project Description ---
st.markdown("""
### 📄 1. Project Name & Description
**Emotion-Aware Chatbot** is an AI-powered chatbot that can understand human emotions from text using Natural Language Processing (NLP).  
It can detect **joy, anger, sadness, fear, surprise**, and **neutral** emotions in real-time and reply accordingly to create an empathetic interaction.

---

### 🎯 2. Objective
- Enhance human-computer interaction
- Support mental well-being through emotion-aware responses
- Demonstrate how NLP and AI can be used in real-world applications

---

### 🛠️ 3. Technologies Used
- **Python 3**
- **Streamlit** – for building the interactive web interface
- **Transformers (HuggingFace)** – pre-trained emotion detection model
- **Pillow** – for image rendering

---

### ⚙️ 4. How It Works
1. User types a message.
2. The pre-trained transformer model analyzes the text.
3. The chatbot detects the most likely emotion with a confidence score.
4. It replies based on the detected emotion.

---

### 🚀 5. Future Improvements
- Add multilingual emotion detection (e.g., বাংলা)
- Store chat history and emotions over time
- Integrate with voice recognition
- Improve emotion response customization

---

### ⚠️ 6. Challenges Faced
- Emotion models mostly trained on English only
- Low accuracy for very short or ambiguous texts
- Limited interactivity compared to full chatbot frameworks

---

### 👨‍💻 7. My Role & Learning
- **Design:** I designed and implemented the full chatbot interface.
- **Development:** Integrated the transformer model into Streamlit.
- **Learning:** Learned how to use NLP pipelines, model inference, and build user-friendly Streamlit UIs.

---
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

Passionate about AI, Machine Learning, and Software Development.  
Building innovative projects to solve real-world problems.
""")

st.markdown("---")

# --- User Input ---
user_input = st.text_input("✉️ Enter your message:")

# --- Detect & Respond to Emotion ---
if st.button("Send"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a message!")
    else:
        emotion, confidence = detect_emotion(user_input)
        confidence_percent = confidence * 100

        # Emotion-based response
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
