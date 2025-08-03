# ChatBot
The program is a simple chatbot that detects emotions in user input using a pre-trained machine learning model. It generates responses based on the detected emotion, making the interaction feel more empathetic and engaging.

### Transformers library
This library is used to load pre-trained models for various natural language processing tasks, including emotion detection. The transformers library offers a convenient way to load such models using a "pipeline," which is an easy-to-use wrapper that applies the model to perform specific tasks, such as emotion detection, sentiment analysis, and text classification.

### pipeline function
Here, the pipeline function initializes an emotion detection model. The specific model used (j-hartmann/emotion-english-distilroberta-base) is pre-trained for emotion classification tasks. The return_all_scores=True parameter allows the pipeline to return scores for all possible emotions, not just the highest one.

### model information
This model is based on DistilRoBERTa, a smaller and faster version of the RoBERTa model (Robustly Optimized BERT Pretraining Approach). DistilRoBERTa is a transformer-based model that uses self-attention mechanisms to analyze relationships between words in a sentence, which allows it to capture complex language patterns and nuances that are associated with different emotions.

Output:
![output](https://github.com/user-attachments/assets/27d8d844-be84-4ee0-8ead-114aab3eceea)
