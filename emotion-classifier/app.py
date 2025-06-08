import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Set page config
st.set_page_config(page_title="Emotion Detector", layout="centered")

# Load the model
pipe_lr = joblib.load(open("text_emotion.pkl", "rb"))

# Emojis for emotions
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨", "happy": "🤗", "joy": "😂",
    "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"
}

# Prediction functions
def predict_emotions(docx):
    return pipe_lr.predict([docx])[0]

def get_prediction_proba(docx):
    return pipe_lr.predict_proba([docx])

# Main App
def main():
    st.markdown("<h1 style='text-align: center; color: #4B0082;'>💬 Emotion Detector</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size:16px;'>Understand the emotions hidden in your text.</p>", unsafe_allow_html=True)
    st.markdown("---")

    with st.form(key='emotionForm'):
        raw_text = st.text_area("📝 Enter your message here", height=150, placeholder="e.g., I feel great today!")
        submit_text = st.form_submit_button(label='🔍 Analyze')

    if submit_text:
        if raw_text.strip() == "":
            st.warning("Please enter some text before submitting.")
            return

        # Predict
        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)
        confidence = np.max(probability)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 💡 Input Summary")
            st.info(raw_text)

            st.markdown("### 🔮 Predicted Emotion")
            st.success(f"{prediction.capitalize()} {emotions_emoji_dict.get(prediction, '')}")

            st.markdown(f"**Confidence Score:** `{confidence:.2f}`")

        with col2:
            st.markdown("### 📊 Emotion Probabilities")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["Emotion", "Probability"]

            chart = alt.Chart(proba_df_clean).mark_bar(
                cornerRadiusTopLeft=10,
                cornerRadiusTopRight=10
            ).encode(
                x=alt.X('Emotion', sort='-y'),
                y='Probability',
                color=alt.Color('Emotion', legend=None),
                tooltip=['Emotion', 'Probability']
            ).properties(
                width=350,
                height=300
            )

            st.altair_chart(chart, use_container_width=True)

    st.markdown("---")
    st.markdown("<small style='text-align: center; display: block;'>Built with ❤️ by Shiwangi using Streamlit</small>", unsafe_allow_html=True)

# Run the app
if __name__ == '__main__':
    main()

