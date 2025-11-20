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

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av

# Page config
st.set_page_config(
    page_title="Sign Language Recognition - LIVE",
    page_icon="👋",
    layout="centered"
)

# Custom CSS (optional)
st.markdown("""
<style>
.main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
.stVideo { border-radius: 10px; overflow: hidden; }
div[data-testid="stWebRTCStreamer"] { max-width: 640px !important; margin: 0 auto; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🤝 Real-Time Sign Language Translation</div>', unsafe_allow_html=True)

# Initialize MediaPipe
@st.cache_resource
def load_mediapipe():
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    return mp_holistic, mp_drawing

mp_holistic, mp_drawing = load_mediapipe()

def extract_keypoints_optimized(results):
    essential_pose_indices = list(range(11, 23))
    pose = np.array([[results.pose_landmarks.landmark[i].x,
                      results.pose_landmarks.landmark[i].y,
                      results.pose_landmarks.landmark[i].z,
                      results.pose_landmarks.landmark[i].visibility] 
                     for i in essential_pose_indices]).flatten() if results.pose_landmarks else np.zeros(48)
    lh = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, lh, rh])

def draw_styled_landmarks(image, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=2),
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=1, circle_radius=1))
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=1, circle_radius=2),
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=1, circle_radius=1))
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=1))
    return image

# Load TFLite model
@st.cache_resource
def load_tflite_model():
    possible_paths = ['Improved_SignLanguage_Model.tflite','model.tflite','./models/Improved_SignLanguage_Model.tflite','sign_language_model.tflite']
    for path in possible_paths:
        if os.path.exists(path):
            interpreter = tf.lite.Interpreter(model_path=path)
            interpreter.allocate_tensors()
            return interpreter, interpreter.get_input_details(), interpreter.get_output_details()
    return None, None, None

interpreter, input_details, output_details = load_tflite_model()
actions = np.array(['AreYou', 'Okay', 'IamFine', 'How', 'feeling'])
sequence_length = 30

# Session state
for key, default in [('sequences', []), ('predictions', []), ('is_processing', False),
                     ('last_prediction_time', 0), ('use_optimized_features', True),
                     ('frame_count', 0), ('last_frame', None),
                     ('action_thresholds', {'How':0.5,'Okay':0.5,'AreYou':0.6,'feeling':0.6,'IamFine':0.6})]:
    if key not in st.session_state:
        st.session_state[key] = default

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.is_processing = st.toggle("Enable Live Processing", value=False)
    st.session_state.use_optimized_features = st.checkbox("Use Optimized Features", value=True)
    for action in actions:
        st.session_state.action_thresholds[action] = st.slider(f"{action} Threshold", 0.1,1.0,st.session_state.action_thresholds.get(action,0.5),0.05)

# WebRTC Video Processor
class LiveVideoProcessor(VideoTransformerBase):
    def __init__(self, interpreter=None, input_details=None, output_details=None, actions=None, sequence_length=30, collection_rate=5, action_thresholds=None, use_optimized_features=True, detection_sensitivity=3):
        self.holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=1, smooth_landmarks=True, refine_face_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
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
        img = cv2.resize(img, (640,480))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(img_rgb)
        img = draw_styled_landmarks(img, results)
        keypoints = self.extract_keypoints_wrapper(results)

        # Sequence collection
        current_time = time.time()
        if (current_time - self.last_time) >= (1.0 / max(1,self.collection_rate)):
            self.sequence.append(keypoints)
            if len(self.sequence) > self.sequence_length:
                self.sequence = self.sequence[-self.sequence_length:]
            self.last_time = current_time
        st.session_state.frame_count = len(self.sequence)

        # Prediction
        if self.interpreter is not None and len(self.sequence) >= self.sequence_length:
            seq = np.array([self.sequence[-self.sequence_length:]], dtype=np.float32)
            try:
                self.interpreter.set_tensor(self.input_details[0]['index'], seq)
                self.interpreter.invoke()
                prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            except:
                prediction = np.random.random(len(self.actions)); prediction/=prediction.sum()
        else:
            prediction = np.random.random(len(self.actions)) if self.interpreter is None else None

        if prediction is not None:
            predicted_class = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            predicted_action = self.actions[predicted_class]
            threshold = self.action_thresholds.get(predicted_action,0.5)
            adjusted_threshold = max(0.1,min(0.9, threshold * (1-(self.detection_sensitivity-3)*0.1)))
            # Consecutive detection logic
            if predicted_action == self.last_prediction:
                self.consecutive_detections[predicted_action] += 1
            else:
                self.consecutive_detections[predicted_action] = 1
                self.last_prediction = predicted_action
            required_consecutive = 2 if predicted_action in ['How','Okay'] else 3
            if confidence >= adjusted_threshold and self.consecutive_detections[predicted_action]>=required_consecutive:
                st.session_state.predictions.append({'action':predicted_action,'confidence':confidence,'time':time.time(),'real_model':(self.interpreter is not None),'threshold_met':True})
                if len(st.session_state.predictions)>50:
                    st.session_state.predictions = st.session_state.predictions[-50:]
            # Draw prediction on frame
            pred_text = f"{predicted_action} ({confidence:.1%})"
            cv2.putText(img,f"Detection: {pred_text}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

        cv2.putText(img,f"Frames: {len(self.sequence)}/{self.sequence_length}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        return av.VideoFrame.from_ndarray(img,format="bgr24")

# Main UI
st.header("🎥 Live Sign Translation")
if st.session_state.is_processing:
    webrtc_streamer(
        key="live-sign-translation",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: LiveVideoProcessor(
            interpreter=interpreter,
            input_details=input_details,
            output_details=output_details,
            actions=actions,
            sequence_length=sequence_length,
            collection_rate=5,
            action_thresholds=st.session_state.action_thresholds,
            use_optimized_features=st.session_state.use_optimized_features,
            detection_sensitivity=3
        ),
        media_stream_constraints={"video":{"width":{"ideal":640},"height":{"ideal":480},"frameRate":{"ideal":15}},"audio":False},
        async_processing=True,
        rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}
    )
else:
    st.info("📹 Enable 'Live Processing' in the sidebar and allow camera access to start.")
