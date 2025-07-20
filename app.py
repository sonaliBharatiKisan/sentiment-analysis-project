
import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
import nltk

# Download NLTK data (only runs once)
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Load the saved model and vectorizer
@st.cache_resource
def load_model():
    with open('sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# Main app
def main():
    st.title("🎬 Movie Review Sentiment Analysis")
    st.write("Enter a movie review and I'll tell you if it's positive or negative!")
    
    # Load model
    model, vectorizer = load_model()
    
    # User input
    user_input = st.text_area("Enter your movie review here:", 
                              placeholder="e.g., This movie was absolutely amazing!")
    
    if st.button("Analyze Sentiment"):
        if user_input.strip():
            # Process the input
            cleaned = preprocess_text(user_input)
            text_vector = vectorizer.transform([cleaned])
            prediction = model.predict(text_vector)[0]
            probability = model.predict_proba(text_vector)[0]
            
            sentiment = "Positive" if prediction == 1 else "Negative"
            confidence = max(probability) * 100
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                if sentiment == "Positive":
                    st.success(f"😊 {sentiment}")
                else:
                    st.error(f"😞 {sentiment}")
            
            with col2:
                st.metric("Confidence", f"{confidence:.1f}%")
            
            # Show processed text
            with st.expander("View processed text"):
                st.write(f"**Original:** {user_input}")
                st.write(f"**Cleaned:** {cleaned}")
        else:
            st.warning("Please enter a review to analyze!")

if __name__ == "__main__":
    main()
