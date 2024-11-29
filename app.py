import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC

# Load the saved model, vectorizer, and tfidf transformer
lsvm = joblib.load('arabic_ner_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
tfidf = joblib.load('tfidf.pkl')

# Function for predictions on external input
def predict_external_input(input_text):
    tokens = input_text.split()
    predictions = []

    for token in tokens:
        test_vec = vectorizer.transform([token])
        test_tfvec = tfidf.transform(test_vec)
        pred_label = lsvm.predict(test_tfvec)[0]
        predictions.append((token, pred_label))
    
    result_df = pd.DataFrame(predictions, columns=['Token', 'Entity'])
    return result_df

# Streamlit UI
st.title("Arabic Named Entity Recognition (NER)")

st.markdown("""
This is a simple Streamlit app to predict named entities in Arabic text.
You can enter a sentence below and get the named entity predictions for each token.
""")

# Input box for custom text
input_text = st.text_area("Enter Arabic text for NER", "أحمد سافر إلى مصر لمشاهدة الأهرامات.")

if st.button('Predict'):
    # Predict and display results
    predicted_df = predict_external_input(input_text)
    st.write("Predictions:", predicted_df)
    # Save the predictions to a CSV file (optional)
    predicted_df.to_csv("predictions.csv", index=False)
    st.write("Predictions have been saved to predictions.csv.")
