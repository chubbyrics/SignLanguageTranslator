import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image

# Page config
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="👋",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .info-box {
        background: #e8f4fd;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🤝 Sign Language Recognition</div>', unsafe_allow_html=True)

# Deployment info
st.markdown("""
<div class="info-box">
🚀 <strong>Basic Deployment Version</strong><br>
This is a simplified version for deployment. Camera features will be added in future updates.
</div>
""", unsafe_allow_html=True)

# Define actions
actions = ['AreYou', 'Okay', 'IamFine', 'How', 'feeling']

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = True

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.success("🟢 Basic Mode Active")
    st.write("Demo sign detection available")
    
    st.subheader("Supported Signs")
    for action in actions:
        st.write(f"• {action}")
    
    # Demo controls
    st.subheader("Demo Controls")
    if st.button("🎭 Generate Demo Detection"):
        demo_action = np.random.choice(actions)
        demo_confidence = np.random.uniform(0.5, 0.9)
        
        st.session_state.predictions.append({
            'action': demo_action,
            'confidence': demo_confidence,
            'time': time.time(),
            'real_model': False
        })
        st.success(f"Demo: {demo_action} detected!")
    
    if st.button("🔄 Clear Data"):
        st.session_state.predictions = []
        st.success("Data cleared!")

# Main content
st.header("📱 How to Use")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Current Features")
    st.write("✅ Basic deployment")
    st.write("✅ Demo sign detection")
    st.write("✅ Mobile-friendly design")
    st.write("✅ Ready for camera integration")

with col2:
    st.subheader("Next Steps")
    st.write("📷 Add camera functionality")
    st.write("🤖 Integrate ML model")
    st.write("🎯 Real-time detection")
    st.write("🚀 Full deployment")

# Upload image for demo
st.header("🖼️ Image Upload Demo")
uploaded_file = st.file_uploader("Upload an image with hand signs", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Simulate processing
    with st.spinner("Analyzing image..."):
        time.sleep(2)
        demo_action = np.random.choice(actions)
        demo_confidence = np.random.uniform(0.6, 0.95)
        
        st.success(f"Detection: **{demo_action}** (Confidence: {demo_confidence:.1%})")
        
        st.session_state.predictions.append({
            'action': demo_action,
            'confidence': demo_confidence,
            'time': time.time(),
            'real_model': False
        })

# Recent detections
st.header("📊 Recent Detections")
if st.session_state.predictions:
    for i, pred in enumerate(reversed(st.session_state.predictions[-5:])):
        time_str = time.strftime('%H:%M:%S', time.localtime(pred['time']))
        confidence_color = "🟢" if pred['confidence'] > 0.8 else "🟡" if pred['confidence'] > 0.6 else "🔴"
        st.write(f"**{pred['action']}** {confidence_color} {pred['confidence']:.1%} - {time_str}")
else:
    st.info("No detections yet. Use the demo controls or upload an image.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Sign Language Recognition App</strong></p>
    <p><small>Deployment Version • Basic Features • Streamlit</small></p>
</div>
""", unsafe_allow_html=True)