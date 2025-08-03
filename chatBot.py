#installing the required library
#pip install transformers torch

from transformers import pipeline

# Load the emotion detection pipeline
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

def detect_emotion(text):
    """Detects the emotion in the text using the pipeline."""
    try:
        results = emotion_pipeline(text)
        # Get the highest scoring emotion
        emotions = {res['label']: res['score'] for res in results[0]}
        primary_emotion = max(emotions, key=emotions.get)
        return primary_emotion, emotions[primary_emotion]
    except Exception as e:
        print("Error detecting emotion:", e)
        return "neutral", 0.0

def chatbot():
    """Simple chatbot that detects emotions and provides text-only responses."""
    print("Hi! I'm here to chat with you. Type 'exit' to stop.")
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Check for exit condition
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye! Take care.")
            break

        # Detect emotion in user input
        emotion, confidence = detect_emotion(user_input)

        # If confidence is too low, respond neutrally
        if confidence < 0.2:
            response = "I'm not sure how you're feeling, but I'm here to chat."
        else:
            # Generate response based on detected emotion
            if emotion == "joy":
                response = "I'm glad to hear that! It sounds like you're feeling happy."
            elif emotion == "anger":
                response = "I'm here to listen. Feel free to share what's bothering you."
            elif emotion == "sadness":
                response = "I'm sorry you're feeling this way. I'm here for you if you need to talk."
            elif emotion == "fear":
                response = "It sounds like something is on your mind. Let me know if you'd like to talk about it."
            elif emotion == "surprise":
                response = "Wow! That sounds unexpected!"
            else:
                response = "Thank you for sharing that with me."

        # Print the chatbot's response and detected emotion
        print(f"Chatbot [{emotion} ({confidence*100:.2f}% confident)]: {response}")

# initializing the chatbot
chatbot()
