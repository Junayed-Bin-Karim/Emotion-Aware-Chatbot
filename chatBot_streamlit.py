import streamlit as st
from transformers import pipeline

# set_page_config() অবশ্যই প্রথমে আসবে
st.set_page_config(page_title="Emotion-Aware Chatbot", page_icon="🤖", layout="centered")

# মডেল লোড করার ফাংশন
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

emotion_pipeline = load_model()

def detect_emotion(text):
    try:
        results = emotion_pipeline(text)
        emotions = {res['label']: res['score'] for res in results[0]}
        primary_emotion = max(emotions, key=emotions.get)
        return primary_emotion, emotions[primary_emotion]
    except Exception:
        return "neutral", 0.0

# Title
st.title("Emotion-Aware Chatbot")
st.write("I can understand your emotions and chat with you. Please type your message below.")

from PIL import Image

# Display your photo
image = Image.open("junayed.jpeg")  # Make sure the image file is in the same directory
st.image(image, caption="Md. Junayed Bin Karim", width=150)

# Updated Creator Info
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

# ইউজারের ইনপুট নেওয়া
user_input = st.text_input("✉️ Enter your message:")

# বাটনে ক্লিক করলে রেসপন্স দেখানো
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
