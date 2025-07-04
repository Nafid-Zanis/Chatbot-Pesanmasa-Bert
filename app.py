# app.py
import streamlit as st
import json
import random
import pickle
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# === Load model & tokenizer ===
model = BertForSequenceClassification.from_pretrained("chatbot-pesanmasa-bert")
tokenizer = BertTokenizer.from_pretrained("chatbot-pesanmasa-bert")

# === Load label encoder ===
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# === Load intents dataset ===
with open("datasets.json", "r") as f:
    intents = json.load(f)

# === Fungsi prediksi intent ===
def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class_id = torch.argmax(outputs.logits).item()
    tag = label_encoder.inverse_transform([predicted_class_id])[0]
    return tag

# === Fungsi ambil respons berdasarkan tag ===
def get_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Maaf, saya belum bisa menjawab pertanyaan itu."

# === Styling chat ala WhatsApp ===
st.markdown("""
<style>
.chat-container {
    background-color: #f7f7f7;
    padding: 15px;
    border-radius: 15px;
    max-height: 400px;
    overflow-y: auto;
    display: flex;
    flex-direction: column-reverse;
}
.user-msg {
    align-self: flex-end;
    background-color: #dcf8c6;
    padding: 10px;
    margin: 5px;
    border-radius: 10px;
    max-width: 75%;
}
.bot-msg {
    align-self: flex-start;
    background-color: #ffffff;
    padding: 10px;
    margin: 5px;
    border-radius: 10px;
    max-width: 75%;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div style='text-align: center;'><img src='./pesanmasa.png' width='120'></div>", unsafe_allow_html=True)

st.title("Chatbot Seputar PESANMASA")

# === Simpan riwayat chat ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === Input form ===
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Tulis pesan Anda:")
    submitted = st.form_submit_button("Kirim")

    if submitted and user_input:
        tag = predict_intent(user_input)
        response = get_response(tag)

        # Simpan percakapan
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", response))

# === Tampilkan chat ===
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for sender, message in reversed(st.session_state.chat_history):
    css_class = "user-msg" if sender == "user" else "bot-msg"
    st.markdown(f'<div class="{css_class}">{message}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
