# Emotion-Aware Chatbot

An advanced conversational AI application that detects user emotions from text input in real-time and responds empathetically. Built using Streamlit and Hugging Face Transformers, this chatbot leverages state-of-the-art NLP techniques to enhance human-computer interactions.

---

## Overview

The Emotion-Aware Chatbot utilizes a pretrained transformer model (`j-hartmann/emotion-english-distilroberta-base`) to classify emotions such as **joy, anger, sadness, fear, surprise**, and **neutral** from user messages. It then generates contextually appropriate and empathetic responses, improving user experience in applications like mental health support, customer service, and emotional analytics.

---

## Features

- **Real-time emotion recognition:** Analyze input text instantly to determine emotional state.  
- **Dynamic empathetic responses:** Reply with context-aware messages tailored to detected emotions.  
- **Visual emotion confidence:** Display confidence scores using interactive bar charts powered by Matplotlib.  
- **Conversation history:** Keep track of user inputs and detected emotions for review.  
- **Downloadable reports:** Export conversation history with emotion labels as a TXT file.  
- **User-friendly UI:** Clean and intuitive interface built with Streamlit.

---

## Technology Stack

| Component            | Technology/Library                           |
|----------------------|----------------------------------------------|
| Frontend             | Streamlit                                   |
| NLP Model            | Hugging Face Transformers (`distilroberta`) |
| Programming Language | Python 3.12                                |
| Visualization        | Matplotlib                                  |
| Image Handling       | Pillow (PIL)                                |

---

## Getting Started

### Prerequisites

- Python 3.12 or later  
- pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Junayed-Bin-Karim/emotion-aware-chatbot.git
   cd emotion-aware-chatbot
2.(Optional) Create and activate a virtual environment:

python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
Usage
Enter your message in the input box.

Click Send to analyze the emotion.

View the detected emotion, chatbot response, and emotion confidence scores.

Use the Try Sample Messages expander to test with predefined inputs.

Download the conversation report using the Download Emotion Report button.

Screenshots
(Add relevant screenshots here to showcase UI and features)

Author
Md. Junayed Bin Karim
Computer Science & Engineering (CSE)
Daffodil International University

GitHub: Junayed-Bin-Karim

LinkedIn: Junayed Bin Karim

License
This project is licensed under the MIT License. See the LICENSE file for details.

Future Enhancements
Extend support for multilingual emotion detection.

Integrate speech-to-text and text-to-speech capabilities for voice interaction.

Collaborate with mental health experts to validate responses and improve utility.

Develop mobile-friendly interface and chatbot deployment options.

