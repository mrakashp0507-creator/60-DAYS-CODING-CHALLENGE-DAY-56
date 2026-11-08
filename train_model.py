
import os
import joblib
import streamlit as st


# ---------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------

st.set_page_config(
    page_title="Smart Spam Detector AI",
    page_icon="🛡️",
    layout="centered"
)


# ---------------------------------------
# 2. LOAD TRAINED MODEL
# ---------------------------------------

MODEL_FILE = "spam_model.joblib"


@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_FILE):
        return None

    return joblib.load(MODEL_FILE)


model = load_model()


# ---------------------------------------
# 3. APPLICATION HEADER
# ---------------------------------------

st.title("🛡️ Smart Spam Detector AI")

st.write(
    "An intelligent SMS classification system "
    "powered by Machine Learning."
)

st.divider()


# ---------------------------------------
# 4. CHECK MODEL
# ---------------------------------------

if model is None:

    st.error(
        "Trained model not found! "
        "Please run train_model.py first."
    )

    st.code("py train_model.py")

    st.stop()


# ---------------------------------------
# 5. MESSAGE INPUT
# ---------------------------------------

st.subheader("📩 Test Your Message")

message = st.text_area(
    "Enter your SMS message:",
    placeholder=(
        "Example: Congratulations! "
        "You have won a free prize..."
    ),
    height=150
)


# ---------------------------------------
# 6. PREDICTION FUNCTION
# ---------------------------------------

def predict_message(text):

    prediction = model.predict([text])[0]

    probabilities = model.predict_proba([text])[0]

    classes = list(model.classes_)

    spam_index = classes.index("spam")

    spam_score = probabilities[spam_index] * 100

    return prediction, spam_score


# ---------------------------------------
# 7. PREDICTION BUTTON
# ---------------------------------------

if st.button(
    "🔍 Analyze Message",
    type="primary",
    use_container_width=True
):

    if not message.strip():

        st.warning("Please enter a message first.")

    else:

        try:

            prediction, spam_score = predict_message(
                message.strip()
            )

            st.divider()

            st.subheader("Analysis Result")

            col1, col2 = st.columns(2)

            col1.metric(
                "Spam Score",
                f"{spam_score:.2f}%"
            )

            col2.metric(
                "Message Length",
                f"{len(message)} characters"
            )

            if prediction == "spam":

                st.error(
                    "🚨 SPAM MESSAGE DETECTED"
                )

                st.write(
                    "The model classifies this message "
                    "as potentially unwanted or suspicious."
                )

            else:

                st.success(
                    "✅ NOT SPAM"
                )

                st.write(
                    "The model classifies this message "
                    "as a legitimate message."
                )

            st.caption(
                "Spam score is the model's estimated spam "
                "probability, not a guarantee of safety."
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# ---------------------------------------
# 8. INFORMATION SECTION
# ---------------------------------------

st.divider()

st.subheader("⚙️ About This Project")

st.markdown(
    """
    - **Algorithm:** Multinomial Naive Bayes
    - **Feature Extraction:** TF-IDF
    - **Language:** Python
    - **Interface:** Streamlit
    - **Model:** Trained SMS Spam Classifier

    This application analyzes SMS text and predicts
    whether it belongs to the spam or legitimate category.
    """
)

st.caption(
    "Smart Spam Detector AI | Day 56 Mini Project"
)
