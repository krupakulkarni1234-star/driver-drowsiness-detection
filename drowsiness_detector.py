"""
Core Drowsiness Detection Module
Implements Eye Aspect Ratio (EAR) computation using Google MediaPipe Face Landmarker
with an automated fallback to local OpenCV Haar Cascades.
"""

import os
import math
import cv2
import numpy as np

def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two 2D points (x, y)."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def calculate_ear(eye_points):
    """
    Compute Eye Aspect Ratio (EAR) based on 6 landmark coordinates:
    p1: outer corner, p4: inner corner
    p2, p3: upper eyelid points
    p5, p6: lower eyelid points
    
    EAR formula: (||p2 - p6|| + ||p3 - p5||) / (2.0 * ||p1 - p4||)
    """
    if len(eye_points) < 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = eye_points[:6]

    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)

    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


class DrowsinessDetector:
    def __init__(self, ear_threshold=0.25, consecutive_frames_threshold=24):
        """
        :param ear_threshold: Eye aspect ratio below which eyes are considered closed.
        :param consecutive_frames_threshold: Consecutive frames eyes must remain closed to trigger alert.
        """
        self.ear_threshold = ear_threshold
        self.consecutive_frames_threshold = consecutive_frames_threshold

        self.closed_frame_counter = 0
        self.total_blinks = 0
        self.is_drowsy = False
        self._was_closed = False

        # Eye landmark indices from 468/478 face mesh
        # Left Eye: [outer, top1, top2, inner, bottom1, bottom2]
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        # Right Eye: [inner, top1, top2, outer, bottom1, bottom2]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

        # Directory containing models
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.base_dir, "models")

        # Initialize MediaPipe Tasks Face Landmarker
        self.detector = None
        self.use_mediapipe = False
        self._init_mediapipe()

        # Initialize Fallback Haar Cascades (using local XML files)
        self.face_cascade = None
        self.eye_cascade = None
        self._init_haar_cascades()

    def _init_mediapipe(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            model_path = os.path.join(self.models_dir, "face_landmarker.task")
            if os.path.exists(model_path):
                base_options = mp_python.BaseOptions(model_asset_path=model_path)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.detector = vision.FaceLandmarker.create_from_options(options)
                self.use_mediapipe = True
                print("[INFO] MediaPipe Face Landmarker loaded successfully.")
            else:
                self.use_mediapipe = False
        except Exception as e:
            print(f"[WARN] MediaPipe Tasks initialization note: {e}")
            self.use_mediapipe = False

    def _init_haar_cascades(self):
        face_xml = os.path.join(self.models_dir, "haarcascade_frontalface_default.xml")
        eye_xml = os.path.join(self.models_dir, "haarcascade_eye.xml")

        if os.path.exists(face_xml) and os.path.exists(eye_xml):
            self.face_cascade = cv2.CascadeClassifier(face_xml)
            self.eye_cascade = cv2.CascadeClassifier(eye_xml)
            if self.face_cascade.empty() or self.eye_cascade.empty():
                self.face_cascade = None
                self.eye_cascade = None
        else:
            self.face_cascade = None
            self.eye_cascade = None

    def process_frame(self, frame):
        """
        Processes a single BGR video frame.
        Returns:
            annotated_frame (frame with visual overlays)
            status_dict (contains ear, is_drowsy, blink_count, counter)
        """
        h, w = frame.shape[:2]
        left_ear = 0.0
        right_ear = 0.0
        avg_ear = 0.0
        face_detected = False

        if self.use_mediapipe and self.detector:
            try:
                import mediapipe as mp
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = self.detector.detect(mp_image)

                if detection_result.face_landmarks and len(detection_result.face_landmarks) > 0:
                    face_detected = True
                    landmarks = detection_result.face_landmarks[0]

                    # Extract coordinates for left eye
                    left_eye_pts = []
                    for idx in self.LEFT_EYE_INDICES:
                        lm = landmarks[idx]
                        left_eye_pts.append((int(lm.x * w), int(lm.y * h)))

                    # Extract coordinates for right eye
                    right_eye_pts = []
                    for idx in self.RIGHT_EYE_INDICES:
                        lm = landmarks[idx]
                        right_eye_pts.append((int(lm.x * w), int(lm.y * h)))

                    left_ear = calculate_ear(left_eye_pts)
                    right_ear = calculate_ear(right_eye_pts)
                    avg_ear = (left_ear + right_ear) / 2.0

                    # Draw eye contours and landmarks
                    cv2.polylines(frame, [np.array(left_eye_pts, dtype=np.int32)], True, (0, 255, 128), 1)
                    cv2.polylines(frame, [np.array(right_eye_pts, dtype=np.int32)], True, (0, 255, 128), 1)
                    for pt in left_eye_pts + right_eye_pts:
                        cv2.circle(frame, pt, 2, (0, 255, 255), -1)
            except Exception as e:
                pass

        # Fallback to Haar Cascades if MediaPipe produced no detection or is unavailable
        if not face_detected and self.face_cascade is not None and not self.face_cascade.empty():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))
            if len(faces) > 0:
                face_detected = True
                (fx, fy, fw, fh) = faces[0]
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 200, 0), 2)
                roi_gray = gray[fy:fy + int(fh * 0.6), fx:fx + fw]
                eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.15, minNeighbors=4, minSize=(20, 20))
                
                # Estimate openness based on visible eyes
                if len(eyes) >= 2:
                    avg_ear = 0.30
                elif len(eyes) == 1:
                    avg_ear = 0.22
                else:
                    avg_ear = 0.15

        # Drowsiness and Blink State Logic
        if face_detected:
            if avg_ear < self.ear_threshold:
                self.closed_frame_counter += 1
                self._was_closed = True

                if self.closed_frame_counter >= self.consecutive_frames_threshold:
                    self.is_drowsy = True
            else:
                if self._was_closed and self.closed_frame_counter < self.consecutive_frames_threshold:
                    self.total_blinks += 1
                self.closed_frame_counter = 0
                self.is_drowsy = False
                self._was_closed = False
        else:
            # No face detected
            self.closed_frame_counter = 0
            self.is_drowsy = False

        status = {
            "face_detected": face_detected,
            "avg_ear": avg_ear,
            "left_ear": left_ear,
            "right_ear": right_ear,
            "closed_frames": self.closed_frame_counter,
            "is_drowsy": self.is_drowsy,
            "blink_count": self.total_blinks,
            "threshold": self.ear_threshold
        }

        return frame, status
