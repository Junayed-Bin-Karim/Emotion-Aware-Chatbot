import streamlit as st
from transformers import pipeline
from PIL import Image

# --- Project Description ---
"""
Project: Emotion-Aware Chatbot

This chatbot detects and understands human emotions from user text input using
a transformer-based pre-trained model (DistilRoBERTa). It responds empathetically
according to the detected emotion, enhancing user experience in conversational AI.

Applications: Mental health support, customer service, emotion-sensitive chatbots.

Key Features:
- Real-time emotion classification
- Emotion-aware responses
- Simple and interactive UI built with Streamlit
"""

# --- Streamlit page configuration ---
st.set_page_config(page_title="Emotion-Aware Chatbot", page_icon="🤖", layout="centered")

# --- Load the emotion detection model with caching ---
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

# --- UI: Title and Instructions ---
st.title("🤖 Emotion-Aware Chatbot")
st.write("I can understand your emotions and chat with you. Please type your message below.")

# --- Display creator image ---
try:
    image = Image.open("junayed.jpeg")
    st.image(image, caption="Md. Junayed Bin Karim", width=150)
except FileNotFoundError:
    st.warning("Creator image not found.")

# --- Creator info ---
st.markdown(
    """
    ### 👤 About the Creator  
    **Name:** Md. Junayed Bin Karim  
    **University:** Daffodil International University  
    **Department:** Computer Science and Engineering (CSE)  
    **GitHub:** [github.com/Junayed-Bin-Karim](https://github.com/Junayed-Bin-Karim)  
    **LinkedIn:** [linkedin.com/in/junayed-bin-karim-47b755270](https://www.linkedin.com/in/junayed-bin-karim-47b755270/)  
    """
)

st.markdown("---")

# --- User input ---
user_input = st.text_input("✉️ Enter your message:")

# --- Button and response ---
if st.button("Send"):
    if user_input.strip() == "":
        st.warning("Please enter a message!")
    else:
        emotion, confidence = detect_emotion(user_input)
        confidence_percent = f"{confidence*100:.2f}%"

        if confidence < 0.2:
            response = "I'm not sure how you're feeling, but I'm here to chat."
        else:
            if emotion == "joy":
                response = "Glad to hear you're feeling happy! 😊"
            elif emotion == "anger":
                response = "I’m here to listen if something’s bothering you."
            elif emotion == "sadness":
                response = "I'm sorry you're feeling sad. I'm here for you."
            elif emotion == "fear":
                response = "It sounds like something worries you. Feel free to talk about it."
            elif emotion == "surprise":
                response = "Wow! That sounds surprising!"
            else:
                response = "Thank you for sharing that with me."

        st.markdown(f"**🤔 Detected Emotion:** `{emotion}` ({confidence_percent} confidence)")
        st.markdown(f"**🤖 Chatbot:** {response}")
