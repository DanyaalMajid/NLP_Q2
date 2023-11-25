import streamlit as st
from transformers import pipeline

# Load models

# Distilled Sentiment Classifier
# Link: https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student
distilled_sentiment_classifier = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
    return_all_scores=True
)

# Emotion Classifier
# Link: https://huggingface.co/SamLowe/roberta-base-go_emotions
emotion_text_classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")

# Named Entity Recognition
# Link: https://huggingface.co/mdarhri00/named-entity-recognition
named_entity_classifier = pipeline("token-classification", model="mdarhri00/named-entity-recognition")

# Toxicity Classifier
# Link: https://huggingface.co/s-nlp/roberta_toxicity_classifier
toxicity_classifier = pipeline("text-classification", model="s-nlp/roberta_toxicity_classifier")

# Streamlit app
def main():
    st.title("HuggingFace Model Demo App")

    # User input for text
    user_text = st.text_area("Enter some text:")

    if user_text:
        # Available Models
        # Sentiment Analysis
        sentiment_checkbox = st.checkbox("Sentiment Analysis")

        # Emotion Analysis
        emotion_checkbox = st.checkbox("Emotion Analysis")

        # Named Entity Recognition
        ner_checkbox = st.checkbox("Named Entity Recognition")

        # Toxicity Analysis
        toxicity_checkbox = st.checkbox("Toxicity Analysis")

        # Run custom and display outputs
        st.header("Function Outputs:")

        if sentiment_checkbox:
            st.subheader("Sentiment Analysis:")
            
            # Parse JSON data
            data = distilled_sentiment_classifier(user_text)
            
            # Extract and display label and score values
            for labels_and_scores in data:
                for entry in labels_and_scores:
                    label = entry["label"]
                    score = entry["score"]
                    st.write(f"Label: {label}, Score: {score}")


        if emotion_checkbox:
            st.subheader("Emotion Analysis:")
            
            # Parse JSON data
            data = emotion_text_classifier(user_text)
            
            # Extract and display label and score values
            for labels_and_scores in data:
                for entry in labels_and_scores:
                    label = entry["label"]
                    score = entry["score"]
                    st.write(f"Label: {label}, Score: {score}")

        if ner_checkbox:
            st.subheader("Named Entity Recognition:")
            
            # Parse JSON data
            data = named_entity_classifier(user_text)

            # Extract and display data
            for entry in data:
                entity_group = entry["entity_group"]
                score = entry["score"]
                word = entry["word"]
                start = entry["start"]
                end = entry["end"]
                st.write(f"Word: {word}, Entity Group: {entity_group}, Score: {score}, Start: {start}, End: {end}")

        if toxicity_checkbox:
            st.subheader("Toxicity Analysis:")
            
            # Parse JSON data
            data = toxicity_classifier(user_text)

            # Extract and display label and score values
            for labels_and_scores in data:
                for entry in labels_and_scores:
                    label = entry["label"]
                    score = entry["score"]
                    st.write(f"Label: {label}, Score: {score}")

if __name__ == "__main__":
    main()
