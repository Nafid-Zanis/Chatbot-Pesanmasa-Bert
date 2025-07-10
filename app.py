import streamlit as st
import json
import random
import pickle
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Pilih device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("Nafid-Zanis/chatbot-pesanmasa-bert")
model.to(device)  # <- ini WAJIB

tokenizer = BertTokenizer.from_pretrained("Nafid-Zanis/chatbot-pesanmasa-bert")

# === Load label encoder ===
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# === Load intents dataset ===
with open("datasets.json", "r") as f:
    intents = json.load(f)

def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # <- PENTING
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
    tag = label_encoder.inverse_transform([predicted_class_id])[0]
    return tag

# === Fungsi ambil respons berdasarkan tag ===
def get_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Maaf, saya belum bisa menjawab pertanyaan itu."

# === CSS Styling ===
st.markdown("""
    <style>
    body {
        font-family: 'Segoe UI', sans-serif;
    }
    .main-title {
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 0;
        color: #1a936f;
    }
    .chat-container {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 15px;
        max-height: 500px;
        overflow-y: auto;
        display: flex;
        flex-direction: column-reverse;
        border: 1px solid #ddd;
    }
    .user-msg {
        align-self: flex-end;
        background-color: #c8e6c9;
        color: #000;
        padding: 10px 15px;
        margin: 6px 0;
        border-radius: 15px 15px 0 15px;
        max-width: 75%;
    }
    .bot-msg {
        align-self: flex-start;
        background-color: #ffffff;
        color: #000;
        padding: 10px 15px;
        margin: 6px 0;
        border-radius: 15px 15px 15px 0;
        max-width: 75%;
        box-shadow: 0 0 5px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# === Header ===
col1, col2 = st.columns([1, 5])
with col1:
    st.image("pesanmasa.png", width=80)
with col2:
    st.markdown("<div class='main-title'>🤖 Chatbot PESANMASA</div>", unsafe_allow_html=True)
    st.caption("Sampaikan pertanyaan Anda, dan saya akan membantu sebisa saya!")

# === Simpan riwayat chat ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



# === Input form ===
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ketik pesan Anda di sini...")
    submitted = st.form_submit_button("📨 Kirim")

    if submitted and user_input:
        tag = predict_intent(user_input)
        response = get_response(tag)

        # Simpan percakapan
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", response))

# === Tampilkan chat ===
# === CSS Styling ===
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for sender, message in reversed(st.session_state.chat_history):
    css_class = "user-msg" if sender == "user" else "bot-msg"
    st.markdown(f'<div class="{css_class}">{message}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
