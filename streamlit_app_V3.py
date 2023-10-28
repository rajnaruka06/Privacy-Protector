import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import Trainer
from nltk import sent_tokenize
from transformers import AutoModelForSeq2SeqLM

new_model = AutoModelForSequenceClassification.from_pretrained(
    "./Pretrained_Classification_Model/"
)
new_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
trainer = Trainer(model=new_model, tokenizer=new_tokenizer)

# Load the saved model
summary_tokenizer = AutoTokenizer.from_pretrained("t5-small")
summary_model = AutoModelForSeq2SeqLM.from_pretrained("./Pretrained_Summary_Model/")

def get_prediction(text):
    encoding = new_tokenizer(text, return_tensors="pt")  ## Try removing return tensors
    encoding = {k: v.to(trainer.model.device) for k, v in encoding.items()}

    outputs = new_model(**encoding)

    logits = outputs.logits

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    probs = probs.detach().numpy()

    label = np.argmax(probs, axis=-1)
    return label, probs[label]


def highlight_problematic_sentence(paragraph):
    # Split the paragraph into sentences
    sentences = sent_tokenize(paragraph)

    # Encode each sentence and obtain the predicted label
    problematic_sentences = []
    for sentence in sentences:
        pred = get_prediction(sentence)
        print(pred)
        if pred[0] and pred[1] > 0.5:
            problematic_sentences.append(sentence)

    print(problematic_sentences)
    # Highlight the problematic sentence in the paragraph
    highlighted_paragraph = paragraph[:]
    for problematic_sentence in problematic_sentences:
        highlighted_paragraph = highlighted_paragraph.replace(
            problematic_sentence,
            f"<span style='background-color:#ff6666'> **{problematic_sentence}**</span>",
        )

    return highlighted_paragraph


def generate_summary(text):
    prefix = "summarize: "
    inputs = prefix + text
    input_ids = summary_tokenizer.encode(
        inputs, return_tensors="pt", max_length=1024, truncation=True
    )
    output = summary_model.generate(
        input_ids=input_ids, max_length=128, num_beams=4, early_stopping=True
    )
    summary = summary_tokenizer.decode(output[0], skip_special_tokens=True)
    return summary


def main():
    st.title("Terms and Policy Classifier")
    st.write(
        "Enter text and click the button to get the prediction and highlight problematic sentences."
    )

    policies = pd.read_excel("test_df.xlsx")["Policy"].to_numpy()
    # random_example = np.random.choice([policies[0], policies[2], policies[4]])
    random_example = policies[0]

    # Create a text area for user input
    user_input = st.text_area(
        label="Enter text here:", height=400, value=random_example
    )

    if st.button("Get Prediction!"):
        if user_input:
            label, probability = get_prediction(user_input)

            if label == 0:
                st.success(f"Prediction: Acceptable")
            else:
                st.error(f"Prediction: Not Acceptable (Confidence: {probability:.2f})")

            st.markdown(
                "<h2>Problemetic Parts in the Policy:</h2>", unsafe_allow_html=True
            )
            highlighted_paragraph = highlight_problematic_sentence(user_input)
            st.markdown(highlighted_paragraph, unsafe_allow_html=True)

            st.markdown("<h2>Summary of the Policy:</h2>", unsafe_allow_html=True)
            reason = generate_summary(user_input)
            st.markdown(reason, unsafe_allow_html=True)
        else:
            st.warning("Please enter some text for prediction.")


if __name__ == "__main__":
    main()
