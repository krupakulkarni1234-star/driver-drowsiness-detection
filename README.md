# Real-Time Driver Drowsiness Detection System

> **Resume Project:** Real-Time Driver Drowsiness Detection system using Eye Aspect Ratio (EAR) based logic for detecting fatigue.  
> **Developer:** Krupa Kulkarni  
> **Tech Stack:** Python, OpenCV, MediaPipe / Dlib, Computer Vision, Facial Landmark Tracking

---

## 📌 Project Overview
Driver fatigue and micro-sleeps are among the leading causes of automotive accidents globally. This system is an automated, real-time computer vision application that continuously monitors a driver's facial expressions and eye closure patterns through a webcam. 

By calculating the **Eye Aspect Ratio (EAR)** across successive video frames, the system differentiates between natural involuntary eye blinks and dangerous drowsiness episodes. If the driver's eyes remain closed longer than a configured threshold (e.g. > 1 second), an immediate non-blocking audio alarm triggers and visual alert warnings flash to wake the driver.

---

## 🧠 Mathematical Foundation: Eye Aspect Ratio (EAR)

The system identifies 6 key landmark coordinates for each eye (introduced by Soukupová and Čech, 2016):
- $p_1, p_4$: Horizontal eye corner landmarks (outer and inner canthus).
- $p_2, p_3$: Upper eyelid landmarks.
- $p_5, p_6$: Lower eyelid landmarks.

The **Eye Aspect Ratio (EAR)** is computed as:

$$\text{EAR} = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \times ||p_1 - p_4||}$$

### How It Works:
1. **Open Eyes:** The vertical distances $(||p_2 - p_6|| \text{ and } ||p_3 - p_5||)$ are large relative to the horizontal distance $(||p_1 - p_4||)$. The typical open-eye EAR value is **0.28 – 0.35**.
2. **Closed Eyes (Blinking / Sleep):** The vertical distances drop close to zero, causing the EAR value to sharply fall below **0.20 – 0.25**.
3. **Thresholding Logic:**
   - **Normal Blink:** Eyes close and reopen within 2 to 6 frames (~100–200 ms).
   - **Drowsiness Episode:** Eyes remain below the threshold for $\ge 24$ consecutive frames (~1.0–1.2 seconds at ~25 FPS). When reached, the system triggers the **Audio Alarm & Red Alert Banner**.

---

## 🏗️ System Architecture

```
+-----------------------------------------------------------+
|                      Camera Video Feed                    |
|             (Live Webcam 30 FPS / Video Stream)           |
+-----------------------------------------------------------+
                             |
                             v
+-----------------------------------------------------------+
|               Frame Preprocessing & Alignment             |
|       - Frame resize & Horizontal flip (mirror mode)      |
|       - BGR to RGB color space conversion                 |
+-----------------------------------------------------------+
                             |
                             v
+-----------------------------------------------------------+
|             Facial Landmark Detection Engine              |
|   - MediaPipe FaceMesh (468 3D landmarks)                 |
|   - Left Eye indices:  [33, 160, 158, 133, 153, 144]      |
|   - Right Eye indices: [362, 385, 387, 263, 373, 380]     |
|   - Fallback: OpenCV Haar Cascade face & eye detector     |
+-----------------------------------------------------------+
                             |
                             v
+-----------------------------------------------------------+
|                   EAR Calculation Unit                    |
|          Compute Euclidean distance vectors               |
|          Calculate Left EAR, Right EAR, and Average       |
+-----------------------------------------------------------+
                             |
                             v
+-----------------------------------------------------------+
|              State Machine & Temporal Analysis            |
|   - Is Average EAR < 0.25 ?                               |
|       YES -> closed_frame_counter += 1                    |
|       NO  -> closed_frame_counter = 0, update blink_count |
|   - Is closed_frame_counter >= 24 frames (~1.0s)?         |
+-----------------------------------------------------------+
               /                               \
             YES                                NO
             /                                   \
+-------------------------+             +-------------------------+
|    Drowsiness State     |             |      Normal State       |
| - Trigger Audio Alarm   |             | - Alarm stopped         |
| - Red Flashing HUD      |             | - Display Green Status  |
+-------------------------+             +-------------------------+
```

---

## 📁 Project Structure

```
driver_drowsiness_detection/
├── main.py                  # Main execution entrypoint with HUD & camera loop
├── drowsiness_detector.py    # Core EAR calculation and landmark tracking logic
├── alarm.py                 # Multi-threaded, non-blocking audio alarm manager
├── requirements.txt         # Package dependencies
├── README.md                # Detailed project documentation
└── INTERVIEW_PREP.md        # Technical interview questions, answers & talking points
```

---

## 🚀 Quickstart & Installation

### Step 1: Open Terminal in Project Directory
```powershell
cd "c:\Users\krupa\OneDrive\Documents\New folder (4)\projects\driver_drowsiness_detection"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application
```bash
python main.py
```

### Optional Command-Line Arguments:
- Run with a custom EAR threshold:
  ```bash
  python main.py --threshold 0.23
  ```
- Adjust consecutive frame duration:
  ```bash
  python main.py --consecutive-frames 30
  ```
- Run on a recorded video file instead of live webcam:
  ```bash
  python main.py --source "path/to/driver_video.mp4"
  ```

---

## ⌨️ Controls
- **`Q` or `ESC`**: Safely stop and exit the program.

---

## 🌟 Key Features
- **Real-Time Performance:** Runs at 25–35+ FPS on standard CPU hardware without requiring dedicated GPU acceleration.
- **Dynamic Heads-Up Display (HUD):** Live EAR status progress bar with threshold marker, real-time FPS counter, blink accumulator, and full-screen alert indicators.
- **Non-Blocking Multithreaded Alarm:** Audio beeps run asynchronously on an isolated thread, preventing UI lag or frame dropping during alarm bursts.
- **Dual Engine Architecture:** Uses high-precision MediaPipe Face Mesh landmarks with automated fallback to OpenCV Haar Cascades.
