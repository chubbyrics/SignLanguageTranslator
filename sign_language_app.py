import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from PIL import Image
import os
import time
import queue
import threading

# NEW imports for WebRTC
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av

# Handle TensorFlow import with fallback
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        tf = tflite
        TENSORFLOW_AVAILABLE = True
        st.success("Using tflite-runtime")
    except ImportError:
        TENSORFLOW_AVAILABLE = False
        st.warning("TensorFlow not available - using demo mode")

# Page config
st.set_page_config(
    page_title="Sign Language Recognition - LIVE",
    page_icon="👋",
    layout="centered"  # Changed from "wide" to "centered" for better mobile view
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .status-box {
        background: #f8f9fa;
        padding: 12px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 8px 0;
    }
    .collecting {
        background: #d4edda;
        border-left: 4px solid #28a745;
    }
    .live-indicator {
        background: #dc3545;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        animation: pulse 1.5s infinite;
    }
    
    /* Make video container smaller */
    .stVideo {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Adjust WebRTC streamer size */
    div[data-testid="stWebRTCStreamer"] {
        max-width: 640px !important;
        margin: 0 auto;
    }
    
    /* Sidebar adjustments */
    .css-1d391kg {
        padding: 1rem;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        div[data-testid="stWebRTCStreamer"] {
            max-width: 100% !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🤝 Real-Time Sign Language Translation</div>', unsafe_allow_html=True)

# Initialize MediaPipe
@st.cache_resource
def load_mediapipe():
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    return mp_holistic, mp_drawing

# Feature extraction functions
def extract_keypoints_optimized(results):
    """Optimized keypoint extraction"""
    essential_pose_indices = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    
    if results.pose_landmarks:
        pose = np.array([[results.pose_landmarks.landmark[i].x,
                         results.pose_landmarks.landmark[i].y,
                         results.pose_landmarks.landmark[i].z,
                         results.pose_landmarks.landmark[i].visibility] 
                        for i in essential_pose_indices]).flatten()
    else:
        pose = np.zeros(len(essential_pose_indices) * 4)
    
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, lh, rh])

# Load MediaPipe modules
mp_holistic, mp_drawing = load_mediapipe()

def draw_styled_landmarks(image, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=1, circle_radius=1))
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=1, circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=1, circle_radius=1))
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=1))
    return image

# Model loading function
@st.cache_resource
def load_tflite_model():
    """Load TFLite model"""
    possible_paths = [
        'Improved_SignLanguage_Model.tflite',
        'model.tflite',
        './models/Improved_SignLanguage_Model.tflite',
        'sign_language_model.tflite'
    ]
    
    for model_path in possible_paths:
        if os.path.exists(model_path):
            try:
                interpreter = tf.lite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                return interpreter, input_details, output_details
            except Exception as e:
                continue
    
    # No model found - use demo mode
    return None, None, None

# Load resources
interpreter, input_details, output_details = load_tflite_model()

# Define actions and parameters
actions = np.array(['AreYou', 'Okay', 'IamFine', 'How', 'feeling'])
sequence_length = 30

# Initialize session state - FIXED VERSION
if 'sequences' not in st.session_state:
    st.session_state.sequences = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'last_prediction_time' not in st.session_state:
    st.session_state.last_prediction_time = 0
if 'use_optimized_features' not in st.session_state:
    st.session_state.use_optimized_features = True
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'last_frame' not in st.session_state:
    st.session_state.last_frame = None

# FIX: Initialize action_thresholds with default values
if 'action_thresholds' not in st.session_state:
    st.session_state.action_thresholds = {
        'How': 0.5,
        'Okay': 0.5,
        'AreYou': 0.6,
        'feeling': 0.6,
        'IamFine': 0.6
    }

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("Model Info")
    if interpreter is not None:
        st.success("✅ Model Loaded")
        st.write(f"Actions: {len(actions)}")
    else:
        st.warning("⚠️ Demonstration Mode")
        st.write(f"Actions: {len(actions)}")
    
    st.subheader("Processing Control")
    st.session_state.is_processing = st.toggle(
        "Enable Live Processing", 
        value=False,
        help="Start/Stop real-time sign language processing"
    )
    
    st.subheader("Recognition Settings")
    st.session_state.use_optimized_features = st.checkbox(
        "Use Optimized Features", 
        value=True,
        help="Use 174 optimized features instead of full 1662"
    )
    
    # EASIER DETECTION: Lower thresholds for "How" and "Okay"
    st.subheader("Action Confidence Thresholds")
    
    # LOWER thresholds for "How" and "Okay" to make them easier to detect
    how_threshold = st.slider(
        "How - Confidence Threshold", 
        0.1, 1.0, 0.5, 0.05,
        help="LOWER threshold for 'How' sign to make it easier to detect"
    )
    
    okay_threshold = st.slider(
        "Okay - Confidence Threshold", 
        0.1, 1.0, 0.5, 0.05,
        help="LOWER threshold for 'Okay' sign to make it easier to detect"
    )
    
    # Slightly higher thresholds for already easy-to-detect actions
    areyou_threshold = st.slider(
        "AreYou - Confidence Threshold", 
        0.1, 1.0, 0.6, 0.05,
        help="Confidence threshold for 'AreYou' sign"
    )
    
    feeling_threshold = st.slider(
        "Feeling - Confidence Threshold", 
        0.1, 1.0, 0.6, 0.05,
        help="Confidence threshold for 'Feeling' sign"
    )
    
    # Default threshold for IamFine
    iamfine_threshold = st.slider(
        "IamFine - Confidence Threshold", 
        0.1, 1.0, 0.6, 0.05,
        help="Confidence threshold for 'IamFine' sign"
    )
    
    # Store thresholds in session state - UPDATED to use existing session state
    st.session_state.action_thresholds = {
        'How': how_threshold,
        'Okay': okay_threshold,
        'AreYou': areyou_threshold,
        'feeling': feeling_threshold,
        'IamFine': iamfine_threshold
    }
    
    # Frame collection rate
    collection_rate = st.slider(
        "Frame Collection Rate", 
        1, 10, 5,
        help="Frames collected per second (higher = more detection attempts)"
    )
    
    # Detection sensitivity
    detection_sensitivity = st.slider(
        "Detection Sensitivity", 
        1, 5, 3,
        help="How sensitive the detection is (higher = more detections)"
    )
    
    st.subheader("📋 Supported Signs")
    for action in actions:
        st.write(f"• {action}")
    
    # Tips for better detection
    st.subheader("💡 Detection Tips")
    st.info("""
    For better detection:
    - Ensure good lighting
    - Keep hands visible
    - Hold signs for 2-3 seconds
    - Use clear movements
    """)
    
    # Clear data button
    if st.button("🔄 Clear All Data"):
        st.session_state.sequences = []
        st.session_state.predictions = []
        st.session_state.frame_count = 0
        st.session_state.last_frame = None
        st.success("All data cleared!")

# --- WebRTC Video Processor class ---
class LiveVideoProcessor(VideoTransformerBase):
    def __init__(self, interpreter=None, input_details=None, output_details=None,
                 actions=None, sequence_length=30, collection_rate=5, 
                 action_thresholds=None, use_optimized_features=True,
                 detection_sensitivity=3):
        # Create a new Holistic instance per processor
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.interpreter = interpreter
        self.input_details = input_details
        self.output_details = output_details
        self.actions = actions
        self.sequence_length = sequence_length
        self.collection_rate = collection_rate
        self.action_thresholds = action_thresholds or {}
        self.use_optimized_features = use_optimized_features
        self.detection_sensitivity = detection_sensitivity

        self.sequence = []
        self.last_time = 0.0
        self.consecutive_detections = {action: 0 for action in actions}
        self.last_prediction = None

    def extract_keypoints_wrapper(self, results):
        return extract_keypoints_optimized(results)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Resize image to smaller size for better performance
        img = cv2.resize(img, (640, 480))
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(img_rgb)
        img = draw_styled_landmarks(img, results)
        keypoints = self.extract_keypoints_wrapper(results)

        current_time = time.time()
        if (current_time - self.last_time) >= (1.0 / max(1, self.collection_rate)):
            self.sequence.append(keypoints)
            if len(self.sequence) > self.sequence_length:
                self.sequence = self.sequence[-self.sequence_length:]
            self.last_time = current_time

        try:
            st.session_state.frame_count = len(self.sequence)
        except Exception:
            pass

        if self.interpreter is not None and len(self.sequence) >= self.sequence_length:
            seq = np.array([self.sequence[-self.sequence_length:]], dtype=np.float32)
            try:
                self.interpreter.set_tensor(self.input_details[0]['index'], seq)
                self.interpreter.invoke()
                prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            except Exception:
                prediction = np.random.random(len(self.actions))
                prediction = prediction / prediction.sum()
        else:
            prediction = np.random.random(len(self.actions)) if self.interpreter is None else None

        if prediction is not None:
            predicted_class = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            predicted_action = self.actions[predicted_class]
            
            # Get action-specific threshold
            threshold = self.action_thresholds.get(predicted_action, 0.5)
            
            # EASIER DETECTION: Apply sensitivity multiplier
            adjusted_threshold = threshold * (1 - (self.detection_sensitivity - 3) * 0.1)
            adjusted_threshold = max(0.1, min(0.9, adjusted_threshold))
            
            # Track consecutive detections for stability
            if predicted_action == self.last_prediction:
                self.consecutive_detections[predicted_action] += 1
            else:
                self.consecutive_detections[predicted_action] = 1
                self.last_prediction = predicted_action
            
            # EASIER DETECTION: Lower the required consecutive detections for "How" and "Okay"
            required_consecutive = 2 if predicted_action in ['How', 'Okay'] else 3
            
            # Only accept prediction if it meets threshold AND has some consistency
            if (confidence >= adjusted_threshold and 
                self.consecutive_detections[predicted_action] >= required_consecutive):
                
                try:
                    st.session_state.predictions.append({
                        'action': predicted_action,
                        'confidence': confidence,
                        'time': time.time(),
                        'real_model': (self.interpreter is not None),
                        'threshold_met': True
                    })
                    if len(st.session_state.predictions) > 50:
                        st.session_state.predictions = st.session_state.predictions[-50:]
                except Exception:
                    pass
            
            # Display prediction with adjusted threshold info
            if confidence >= adjusted_threshold:
                if self.consecutive_detections[predicted_action] >= required_consecutive:
                    color = (0, 255, 0)  # Green - high confidence and consistent
                else:
                    color = (0, 165, 255)  # Orange - meets threshold but not consistent yet
            else:
                color = (0, 0, 255)  # Red - below threshold
                
            pred_text = f"{predicted_action} ({confidence:.1%})"
            cv2.putText(img, f"Detection: {pred_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Display adjusted threshold and consistency info
            consistency = self.consecutive_detections[predicted_action]
            cv2.putText(img, f"Threshold: {adjusted_threshold:.1%}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, f"Consistency: {consistency}/{required_consecutive}", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        buffer_count = len(self.sequence)
        cv2.putText(img, f"Frames: {buffer_count}/{self.sequence_length}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
        # Display detection tips for "How" and "Okay"
        if buffer_count >= sequence_length:
            cv2.putText(img, "Tip: Hold signs for 2-3 seconds", (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---- Main UI: Video Section ----
st.header("🎥 Live Sign Translation")

if st.session_state.is_processing:
    use_opt = st.session_state.get("use_optimized_features", True)
    action_thresholds = st.session_state.get("action_thresholds", {})
    rate = collection_rate
    sensitivity = detection_sensitivity

    def processor_factory():
        return LiveVideoProcessor(
            interpreter=interpreter,
            input_details=input_details,
            output_details=output_details,
            actions=actions,
            sequence_length=sequence_length,
            collection_rate=rate,
            action_thresholds=action_thresholds,
            use_optimized_features=use_opt,
            detection_sensitivity=sensitivity
        )

    # Container with fixed width for the camera preview
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            webrtc_ctx = webrtc_streamer(
                key="live-sign-translation",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=processor_factory,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 640},  # Smaller resolution
                        "height": {"ideal": 480},
                        "frameRate": {"ideal": 15}  # Lower frame rate for performance
                    }, 
                    "audio": False
                },
                async_processing=True,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                }
            )

    # Status information below the video
    current_frames = st.session_state.frame_count
    
    # Compact status display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Frames Collected", f"{current_frames}/{sequence_length}")
    
    with col2:
        if st.session_state.predictions:
            latest_pred = st.session_state.predictions[-1]
            status_color = "🟢" if latest_pred['threshold_met'] else "🟡"
            st.metric("Latest Detection", f"{status_color} {latest_pred['action']}")
        else:
            st.metric("Status", "🔴 Waiting")
    
    with col3:
        if st.session_state.predictions:
            latest_pred = st.session_state.predictions[-1]
            threshold = st.session_state.action_thresholds.get(latest_pred['action'], 0.5)
            st.metric("Confidence", f"{latest_pred['confidence']:.1%}")

    # Compact status message
    st.info(f"""
    **Detection Active** | Rate: {collection_rate} FPS | Sensitivity: {sensitivity}/5
    """)
else:
    st.info("📹 Enable 'Live Processing' in the sidebar and allow camera access to start.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🚀 <strong>Sign Language Translation App</strong></p>
    <p><small>Optimized for deployment • Mobile-friendly • Real-time processing</small></p>
</div>
""", unsafe_allow_html=True)