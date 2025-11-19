import streamlit as st
import cv2
import numpy as np
import os
import time
import av

# Try imports with fallbacks
try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    st.error("streamlit-webrtc not available")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    st.warning("MediaPipe not available - using basic camera only")

# Page config
st.set_page_config(
    page_title="Sign Language Recognition - LIVE",
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
    .demo-mode {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🤝 Sign Language Recognition</div>', unsafe_allow_html=True)

# Demo mode info
st.markdown("""
<div class="demo-mode">
🚀 <strong>Deployment Mode Active</strong><br>
Basic camera functionality available. Enable processing to test sign detection.
</div>
""", unsafe_allow_html=True)

# Initialize MediaPipe if available
if MEDIAPIPE_AVAILABLE:
    @st.cache_resource
    def load_mediapipe():
        try:
            mp_holistic = mp.solutions.holistic
            mp_drawing = mp.solutions.drawing_utils
            return mp_holistic, mp_drawing
        except Exception as e:
            return None, None

    mp_holistic, mp_drawing = load_mediapipe()
else:
    mp_holistic, mp_drawing = None, None

# Define actions and parameters
actions = np.array(['AreYou', 'Okay', 'IamFine', 'How', 'feeling'])
sequence_length = 30

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'action_thresholds' not in st.session_state:
    st.session_state.action_thresholds = {
        'How': 0.5, 'Okay': 0.5, 'AreYou': 0.6, 'feeling': 0.6, 'IamFine': 0.6
    }

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.info("🟢 Basic Mode Active")
    st.write("Camera features available")
    
    st.session_state.is_processing = st.toggle(
        "Enable Camera", 
        value=False,
        help="Start/Stop camera feed"
    )
    
    st.subheader("Supported Signs")
    for action in actions:
        st.write(f"• {action}")
    
    if st.button("🔄 Clear Data"):
        st.session_state.predictions = []
        st.session_state.frame_count = 0
        st.success("Data cleared!")

# Video Processor Class
class LiveVideoProcessor(VideoTransformerBase):
    def __init__(self, actions=None, sequence_length=30):
        self.holistic = None
        if MEDIAPIPE_AVAILABLE and mp_holistic:
            try:
                self.holistic = mp_holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except Exception:
                pass
        
        self.actions = actions or []
        self.sequence_length = sequence_length
        self.sequence = []
        self.last_time = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (640, 480))
        
        # Basic MediaPipe processing if available
        if self.holistic:
            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.holistic.process(img_rgb)
                
                # Draw basic landmarks if available
                if mp_drawing and results:
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    if results.left_hand_landmarks:
                        mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    if results.right_hand_landmarks:
                        mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            except Exception:
                pass
        
        # Update frame count
        current_time = time.time()
        if (current_time - self.last_time) >= 0.2:  # 5 FPS
            if len(self.sequence) < self.sequence_length:
                self.sequence.append(1)
            self.last_time = current_time
        
        st.session_state.frame_count = len(self.sequence)
        
        # Demo detection (random for demonstration)
        if len(self.sequence) >= 10 and np.random.random() < 0.05:  # 5% chance
            demo_action = np.random.choice(self.actions)
            demo_confidence = np.random.uniform(0.3, 0.8)
            
            st.session_state.predictions.append({
                'action': demo_action,
                'confidence': demo_confidence,
                'time': time.time(),
                'real_model': False
            })
            
            if len(st.session_state.predictions) > 10:
                st.session_state.predictions = st.session_state.predictions[-10:]
            
            color = (0, 255, 0) if demo_confidence > 0.6 else (0, 165, 255)
            cv2.putText(img, f"Demo: {demo_action}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display info
        cv2.putText(img, f"Frames: {len(self.sequence)}/{self.sequence_length}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img, "Demo Mode - Basic Camera", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Main App
def main():
    st.header("🎥 Live Sign Translation")
    
    if not st.session_state.is_processing:
        st.info("""
        👆 **Enable 'Camera' in the sidebar to start**
        
        **Current Status:** Basic Deployment Mode
        - Camera functionality available
        - Demo sign detection active
        - MediaPipe landmarks if available
        
        **Next:** Once deployed, we can add the ML model back.
        """)
        return

    if not WEBRTC_AVAILABLE:
        st.error("Camera features not available in this deployment.")
        return

    # Camera stream
    def processor_factory():
        return LiveVideoProcessor(
            actions=actions,
            sequence_length=sequence_length
        )

    # WebRTC streamer
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            webrtc_ctx = webrtc_streamer(
                key="sign-language-demo",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=processor_factory,
                media_stream_constraints={
                    "video": {"width": 640, "height": 480},
                    "audio": False
                },
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
                async_processing=True,
            )

    # Status display
    if st.session_state.predictions:
        latest = st.session_state.predictions[-1]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Demo Detection", latest['action'])
        with col2:
            st.metric("Confidence", f"{latest['confidence']:.1%}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><small>Deployment Version • Basic Camera Mode • Streamlit</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()