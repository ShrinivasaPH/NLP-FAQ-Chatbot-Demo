# pseudo_website_chatbot_v3.py
# Run with: streamlit run pseudo_website_chatbot_v3.py

import streamlit as st
from sentence_transformers import SentenceTransformer, util
import textwrap
from datetime import datetime
import time
# -------------------------------
# 1. Streamlit UI Setup
# -------------------------------
st.set_page_config(
    page_title="AI ChatBot Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# -------------------------------
# 2. FAQ Data
# -------------------------------
faq_data = [
    {"question": "Product info.", "answer": "Hello! I'm here to help you with information about our products and services. Feel free to ask about anything you'd like to know about our company."},
    {"question": "Hi. Hello.", "answer": "Hello! Welcome to our company. I'm here to assist you with any questions about our products, services, or business. How can I help you today?"},
    {"question": "Hi. How are you?", "answer": "I'm doing great, thanks for asking! I'm ready to help you with information about our products and services. What would you like to know?"},
    {"question": "How to order the product/item/items/goods?", "answer": "It's simple. Our various range of products are available on all major retail and online platforms like Amazon, Flipkart etc."},
    {"question": "Hi. How are you doing?", "answer": "I'm doing well and happy to assist you! Feel free to ask me anything about our company, products, or services."},
    {"question": "What are your business hours?", "answer": "Our office is open from 9:00 AM to 6:00 PM, Monday through Friday (Indian Standard Time). Our team is available during these hours for support."},
    {"question": "What are your office timings?", "answer": "We operate from 9:00 AM to 6:00 PM, Monday through Friday (Indian Standard Time). You can reach us for assistance during these hours."},
    {"question": "Where is your office located?", "answer": "Our office is located at 123 Main Street, Bangalore, Karnataka, India. This central location helps us serve our clients effectively."},
    {"question": "What is your office location?", "answer": "You can find us at 123 Main Street, Bangalore, Karnataka, India. Our office is well-equipped to support all your business needs."},
    {"question": "Do you offer international shipping?", "answer": "Yes, we ship to over 50 countries worldwide. Our international shipping network ensures reliable delivery to your location."},
    {"question": "How can I reset my password?", "answer": "To reset your password, go to the login page and click 'Forgot Password.' Follow the simple steps to securely reset your account access."},
    {"question": "What is your refund policy?", "answer": "We offer a full refund within 30 days of purchase. Our straightforward refund process ensures your satisfaction with every transaction."},
    {"question": "How do I contact customer service?", "answer": "You can reach our support team at support@example.com or call +91-12345-67890 during business hours. We're here to help with any questions."},
    {"question": "How do I contact?", "answer": "Contact us at support@example.com or call +91-12345-67890 during business hours for assistance with any inquiries."},
    {"question": "How to contact?", "answer": "Reach us at support@example.com or call +91-12345-67890 during business hours. We're ready to assist you."},
    {"question": "How to get in touch?", "answer": "You can email us at support@example.com or call +91-12345-67890 during business hours for prompt assistance."},
    {"question": "How to get in touch with customer care?", "answer": "Contact our customer care team at support@example.com or call +91-12345-67890 during business hours for support."},
    {"question": "Customer care number?", "answer": "Our customer care number is +91-12345-67890 (business hours only). You can also email us at support@example.com anytime."},
    {"question": "Can I change my order after placing it?", "answer": "Yes, you can modify your order within 24 hours of placement. Contact our customer service team to make any necessary changes."},
    {"question": "Do you provide gift wrapping?", "answer": "Yes, we offer complimentary gift wrapping services. Simply request gift wrapping when placing your order."},
    {"question": "What payment methods do you accept?", "answer": "We accept major credit cards, debit cards, UPI, and net banking. All transactions are processed securely."},
    {"question": "Do you have a loyalty program?", "answer": "Yes, join our rewards program to earn points on every purchase. Members enjoy special discounts and exclusive offers."},
    {"question": "How long does shipping take?", "answer": "Domestic orders take 3-5 business days, while international orders take 7-14 business days for delivery."},
    {"question": "Can I track my order?", "answer": "Yes, you'll receive a tracking link once your order ships. You can monitor your package's progress in real-time."},
    {"question": "Shipping tracking details?", "answer": "You'll receive tracking information via email once your order is dispatched. This allows you to monitor delivery progress."},
    {"question": "Do you offer bulk order discounts?", "answer": "Yes, we offer volume pricing for bulk orders. Contact sales@example.com for customized pricing on large quantities."},
    {"question": "Is my payment information secure?", "answer": "Yes, all transactions are encrypted and secure. We use industry-standard security protocols to protect your financial information."},
    {"question": "Do you offer installation/assembly/assemgling services?", "answer": "Yes, we provide installation and assembly services for select products. Please inquire about availability for your specific purchase."},
    {"question": "How do you handle returning damaged goods/products?", "answer": "Report damaged items through our website or mobile app. We'll quickly process a replacement to ensure you receive quality products."},
]

# -------------------------------
# 3. Load Sentence-Transformer Model
# -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_model()
faq_questions = [item['question'] for item in faq_data]
faq_embeddings = embed_model.encode(faq_questions, convert_to_tensor=True)

# -------------------------------
# 4. Retrieval function (FIXED)
# -------------------------------
def retrieve_answer(user_query):
    # Use only the current user query for embedding matching
    query_embedding = embed_model.encode(user_query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, faq_embeddings)
    best_idx = similarities.argmax()
    best_score = similarities[0][best_idx].item()

    if best_score < 0.5:
        top_idx = similarities[0].topk(1).indices[0].item()
        suggested_q = faq_questions[top_idx]
        return f"⚠️ Sorry, I don't have an exact answer. Did you mean: '{suggested_q}'?", [suggested_q]
    return faq_data[best_idx]['answer'], None

# -------------------------------
# 5. Main pseudo-company content
# -------------------------------
st.markdown("<h1>🤖 AI Chatbot Demo </h1>", unsafe_allow_html=True)
st.markdown("""
<p> 📃  Powered by the NLP model <strong>"all-MiniLM-L6-v2"</strong>, this app demonstrates chatbot-usecases of a fictional company.</p>
<p> 💻  Replace conventional FAQs with AI chatBOT.</p>
""", unsafe_allow_html=True)

# -------------------------------
# 6. Session State Initialization
# -------------------------------
for key in ["chat_history", "followup_questions", "user_input", "chat_open"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else None if key != "chat_open" else False

# -------------------------------
# 7. Chatbot Functions
# -------------------------------
def display_message(speaker, message):
    timestamp = datetime.now().strftime("%H:%M")
    wrapped_text = "<br>".join(textwrap.wrap(message, width=60))
    color = "#DCF8C6" if speaker == "You" else "#ECECEC"
    align = "right" if speaker == "You" else "left"
    st.markdown(
        f"<div style='text-align:{align};margin:5px 0;'>"
        f"<span style='background-color:{color};padding:10px 15px;border-radius:20px;display:inline-block;max-width:80%'>"
        f"{wrapped_text}<br><span style='font-size:10px;color:gray'>{timestamp}</span></span></div>",
        unsafe_allow_html=True
    )

def handle_input():
    user_input = st.session_state.user_input.strip()
    if not user_input:
        return

    # ---------------- Follow-up question handling ----------------
    if st.session_state.followup_questions:
        follow_q = st.session_state.followup_questions[0]
        if user_input.lower() in ["yes", "y"]:
            # User confirmed suggestion
            for faq in faq_data:
                if faq["question"] == follow_q:
                    st.session_state.chat_history.append(("You", "Yes"))
                    st.session_state.chat_history.append(("Bot", faq["answer"]))
                    st.session_state.followup_questions = None
                    st.session_state.user_input = ""
                    return
        elif user_input.lower() in ["no", "n"]:
            st.session_state.chat_history.append(("You", "No"))
            st.session_state.chat_history.append(("Bot", "Okay, please rephrase your question."))
            st.session_state.followup_questions = None
            st.session_state.user_input = ""
            return
        else:
            # If user types anything else, treat as a new question
            pass

    # ---------------- Normal retrieval (FIXED) ----------------
    st.session_state.chat_history.append(("You", user_input))
    # Remove prev_question parameter - use only current query
    answer, followups = retrieve_answer(user_input)
    st.session_state.followup_questions = followups
    st.session_state.chat_history.append(("Bot", answer))
    st.session_state.user_input = ""

# -------------------------------
# 8. Chatbot UI (floating at bottom-right)
# -------------------------------
st.markdown("""
<style>
.chat-container {
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 350px;
    max-height: 500px;
    background-color: white;
    border-radius: 10px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    z-index: 9999;
}
.chat-header {
    background-color: #34B7F1;
    color: white;
    padding: 10px;
    font-weight: bold;
    cursor: pointer;
    border-radius:10px 10px 0 0;
}
.chat-body {
    padding: 10px;
    overflow-y: auto;
    max-height: 400px;
}
.chat-input {
    width: 100%;
    padding: 8px;
    box-sizing: border-box;
    border-top: 1px solid #ddd;
}
.quick-reply {
    display:inline-block;
    background-color:#34B7F1;
    color:white;
    padding:5px 10px;
    border-radius:15px;
    margin:2px;
    cursor:pointer;
}
</style>
""", unsafe_allow_html=True)

chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown('<div class="chat-header">💬 Chat here</div>', unsafe_allow_html=True)
    chat_body = st.empty()
    st.text_input("", key="user_input", placeholder="Type your question here...", on_change=handle_input)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# 9. Display chat messages
# -------------------------------
with chat_body.container():
    for speaker, message in st.session_state.chat_history:
        display_message(speaker, message)
    

# -------------------------------
# 10. Footer Disclaimer
# -------------------------------
st.markdown("""
<div style="background-color:#fffae6;padding:10px 20px;border-left:6px solid #ffc107;margin-top:30px;">
<b>⚠️ Disclaimer:</b> This is a <i>fake/sample website</i> created purely to demonstrate my chatbot skills. 
It is <b>not affiliated with any real company</b>.
</div>
""", unsafe_allow_html=True)