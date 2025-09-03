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
    {"question": "Product info.", "answer": "Welcome! I'm delighted to assist you with detailed information about our comprehensive product portfolio and enterprise solutions. Whether you're interested in our flagship products, customized services, or innovative solutions, I'm here to provide you with expert guidance tailored to your specific requirements."},
    {"question": "Hi. Hello.", "answer": "Hello and welcome! It's wonderful to connect with you today. I'm your dedicated virtual assistant, ready to provide you with exceptional support and comprehensive information about our organization, products, and services. How may I elevate your experience with us?"},
    {"question": "Hi. How are you?", "answer": "Thank you for your thoughtful inquiry! I'm operating at full capacity and thoroughly enjoying the opportunity to assist valued clients like yourself. I'm here to provide you with expert knowledge about our products, services, and company initiatives. What specific information can I help you discover today?"},
    {"question": "Hi. How are you doing?", "answer": "I appreciate you asking! I'm performing excellently and am genuinely excited to support your needs today. My expertise spans our complete range of products, services, and organizational capabilities. Please feel free to explore any aspect of our business that interests you."},
    {"question": "What are your business hours?", "answer": "Our corporate headquarters operates during standard business hours: 9:00 AM to 6:00 PM, Monday through Friday (Indian Standard Time). Our dedicated team is fully available during these hours to provide you with comprehensive support and address any inquiries you may have."},
    {"question": "What are your office timings?", "answer": "Our professional team is available to serve you from 9:00 AM to 6:00 PM, Monday through Friday (Indian Standard Time). During these hours, you'll have access to our full range of customer support services and expert consultation."},
    {"question": "Where is your office located?", "answer": "Our state-of-the-art corporate headquarters is strategically positioned at 123 Main Street, Bangalore, Karnataka, India. This prime location enables us to serve our clients effectively while maintaining our commitment to excellence in all business operations."},
    {"question": "What is your office location?", "answer": "You'll find our modern corporate facility at 123 Main Street, Bangalore, Karnataka, India. Our headquarters features cutting-edge infrastructure designed to support our commitment to delivering exceptional products and services to our valued clients."},
    {"question": "Do you offer international shipping?", "answer": "Absolutely! We're proud to offer comprehensive global shipping services, reaching clients across more than 50 countries worldwide. Our international logistics network ensures reliable, secure, and timely delivery of your orders, regardless of your location."},
    {"question": "How can I reset my password?", "answer": "Resetting your password is simple and secure. Please navigate to our login portal and select 'Forgot Password.' Our advanced security system will guide you through a streamlined authentication process, ensuring your account remains protected while providing convenient access."},
    {"question": "What is your refund policy?", "answer": "We stand behind our products with a comprehensive, customer-friendly refund policy. You can request a full refund within 30 days of purchase, reflecting our commitment to your complete satisfaction. Our streamlined process ensures quick resolution of any concerns."},
    {"question": "How do I contact customer service?", "answer": "Our customer support team is readily available to assist you. You can reach our specialists via email at support@example.com or connect directly by calling +91-12345-67890 during business hours. We're committed to providing prompt, professional assistance."},
    {"question": "How do I contact?", "answer": "Connecting with our expert support team is effortless. Email us at support@example.com for detailed assistance, or call +91-12345-67890 to speak directly with our knowledgeable representatives during business hours."},
    {"question": "How to contact?", "answer": "We've made it easy to reach our dedicated support professionals. Simply email support@example.com or call +91-12345-67890 during our business hours for immediate, personalized assistance with any questions or needs you may have."},
    {"question": "How to get in touch?", "answer": "Getting in touch with our exceptional customer care team is straightforward. Contact us at support@example.com for comprehensive email support, or dial +91-12345-67890 to speak with our experienced representatives who are ready to assist you."},
    {"question": "How to get in touch with customer care?", "answer": "Our customer care excellence team is just a click or call away! Reach us at support@example.com for detailed email support, or call +91-12345-67890 during business hours to connect with our professional customer care specialists."},
    {"question": "Customer care number?", "answer": "Our dedicated customer care hotline is +91-12345-67890, staffed by experienced professionals during business hours. For comprehensive email support, please reach out to support@example.com. We're here to ensure your complete satisfaction."},
    {"question": "Can I change my order after placing it?", "answer": "Certainly! We understand that requirements can evolve. You can modify your order within 24 hours of placement. Simply contact our responsive customer service team, and we'll work diligently to accommodate your changes and ensure your complete satisfaction."},
    {"question": "Do you provide gift wrapping?", "answer": "Yes, we're delighted to offer elegant, complimentary gift wrapping services! Our premium packaging adds a sophisticated touch to your purchases, making them perfect for special occasions. Simply request this service when placing your order."},
    {"question": "What payment methods do you accept?", "answer": "We offer a comprehensive suite of secure payment options for your convenience, including all major credit cards, debit cards, UPI transactions, and net banking. Our advanced payment infrastructure ensures safe, encrypted transactions every time."},
    {"question": "Do you have a loyalty program?", "answer": "Absolutely! We're excited to invite you to join our exclusive rewards program, designed to recognize and celebrate our valued customers. Earn valuable points with every purchase and unlock special benefits, discounts, and premium experiences."},
    {"question": "How long does shipping take?", "answer": "We've optimized our logistics for your convenience. Domestic orders are delivered within 3-5 business days, while international shipments typically arrive within 7-14 business days. Our efficient fulfillment network ensures your orders reach you promptly and in perfect condition."},
    {"question": "Can I track my order?", "answer": "Absolutely! Transparency is important to us. Once your order ships, you'll receive a detailed tracking link providing real-time updates on your package's journey. Monitor your shipment's progress from our fulfillment center right to your doorstep."},
    {"question": "Shipping tracking details?", "answer": "Order tracking is seamless with our advanced system. You'll receive comprehensive tracking information via email once your order is dispatched, allowing you to monitor delivery progress in real-time and plan accordingly for receipt."},
    {"question": "Do you offer bulk order discounts?", "answer": "Yes, we're pleased to offer attractive volume pricing for bulk orders! Our sales specialists at sales@example.com will work with you to create customized pricing solutions that deliver exceptional value for your large-quantity requirements."},
    {"question": "Is my payment information secure?", "answer": "Security is our highest priority. We employ enterprise-grade encryption and advanced security protocols to protect all your financial information. Our payment systems meet the strictest industry standards, ensuring your transactions are completely secure and confidential."},
    {"question": "Do you offer installation/assembly/assemgling services?", "answer": "Yes, we provide professional installation and assembly services for eligible products. Our certified technicians ensure proper setup and optimal performance. Please inquire about availability for your specific purchase – we're committed to your complete satisfaction."},
    {"question": "How do you handle returning damaged goods/products?", "answer": "We take product quality seriously and make returns effortless. Simply report any damage through our user-friendly online portal or mobile application. Our quality assurance team will expedite a replacement, ensuring you receive perfect products quickly and hassle-free."},
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
    # Quick reply buttons for follow-up question
    if st.session_state.followup_questions:
        follow_q = st.session_state.followup_questions[0]
        cols = st.columns(2)
        if cols[0].button("Yes"):
            st.session_state.user_input = "Yes"
            handle_input()
        if cols[1].button("No"):
            st.session_state.user_input = "No"
            handle_input()

# -------------------------------
# 10. Footer Disclaimer
# -------------------------------
st.markdown("""
<div style="background-color:#fffae6;padding:10px 20px;border-left:6px solid #ffc107;margin-top:30px;">
<b>⚠️ Disclaimer:</b> This is a <i>fake/sample website</i> created purely to demonstrate my chatbot skills. 
It is <b>not affiliated with any real company</b>.
</div>
""", unsafe_allow_html=True)