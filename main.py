"""
Main Execution Script for Driver Drowsiness Detection System
Author: Krupa Kulkarni
Tech Stack: Python, OpenCV, MediaPipe / Eye Aspect Ratio (EAR) logic
"""

import cv2
import time
import argparse
from drowsiness_detector import DrowsinessDetector
from alarm import AlarmManager

def draw_hud(frame, status, fps):
    """Draws a modern Heads-Up Display (HUD) overlay on the video frame."""
    h, w = frame.shape[:2]

    # Top status bar background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Title & FPS
    cv2.putText(frame, "DRIVER DROWSINESS MONITORING SYSTEM", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 140, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    # Telemetry metrics
    ear = status["avg_ear"]
    blinks = status["blink_count"]
    thresh = status["threshold"]
    closed_frames = status["closed_frames"]

    metric_text = f"EAR: {ear:.3f} (Thresh: {thresh:.2f}) | Blinks: {blinks} | Frames Closed: {closed_frames}"
    cv2.putText(frame, metric_text, (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # EAR Visual Bar
    bar_x, bar_y, bar_w, bar_h = 20, 80, 280, 14
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
    fill_w = int(min(max(ear / 0.45, 0.0), 1.0) * bar_w)
    bar_color = (0, 230, 0) if ear >= thresh else (0, 69, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)

    # Threshold indicator line on bar
    thresh_x = int(bar_x + (thresh / 0.45) * bar_w)
    cv2.line(frame, (thresh_x, bar_y - 2), (thresh_x, bar_y + bar_h + 2), (0, 255, 255), 2)

    # Main Status Alert Banner
    if status["is_drowsy"]:
        # Flash Red Alert Banner across the center/bottom
        alert_y1, alert_y2 = h - 90, h - 20
        cv2.rectangle(frame, (20, alert_y1), (w - 20, alert_y2), (0, 0, 230), -1)
        cv2.putText(frame, "!!! DROWSINESS DETECTED - WAKE UP !!!", (w // 2 - 270, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 3, cv2.LINE_AA)
    elif not status["face_detected"]:
        cv2.putText(frame, "NO FACE DETECTED - POSITION CAMERA", (20, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "STATUS: DRIVER ALERT & ACTIVE", (20, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    # Controls hint
    cv2.putText(frame, "Press 'Q' to Exit", (w - 180, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

def main():
    parser = argparse.ArgumentParser(description="Driver Drowsiness Detection System")
    parser.add_argument("--source", default=0, help="Webcam index (0) or path to video file")
    parser.add_argument("--threshold", type=float, default=0.25, help="Eye Aspect Ratio (EAR) threshold (default: 0.25)")
    parser.add_argument("--consecutive-frames", type=int, default=24, help="Frames below threshold to trigger alert (default: 24)")
    args = parser.parse_args()

    # Convert numeric camera source
    source = int(args.source) if str(args.source).isdigit() else args.source

    print("=" * 65)
    print("  DRIVER DROWSINESS DETECTION SYSTEM")
    print("  Author: Krupa Kulkarni")
    print(f"  EAR Threshold: {args.threshold} | Frame Threshold: {args.consecutive_frames}")
    print("=" * 65)
    print(f"Opening video source: {source}...")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source {source}.")
        print("Tip: Make sure your webcam is plugged in and not used by another app.")
        return

    detector = DrowsinessDetector(
        ear_threshold=args.threshold,
        consecutive_frames_threshold=args.consecutive_frames
    )
    alarm = AlarmManager()

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video feed or camera disconnected.")
                break

            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30.0
            prev_time = curr_time

            # Mirror frame horizontally for intuitive webcam view
            frame = cv2.flip(frame, 1)

            # Process frame through detection engine
            annotated_frame, status = detector.process_frame(frame)

            # Manage audio alarm
            if status["is_drowsy"]:
                alarm.start_alarm()
            else:
                alarm.stop_alarm()

            # Render Heads-Up Display
            draw_hud(annotated_frame, status, fps)

            # Display feed
            cv2.imshow("Driver Drowsiness Detection - Krupa Kulkarni", annotated_frame)

            # Key controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("[INFO] Exiting program...")
                break

    finally:
        alarm.stop_alarm()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Resources released successfully.")

if __name__ == "__main__":
    main()
