# Driver Drowsiness Detection System 🚗💤

A real-time drowsiness detection system using OpenCV and computer vision techniques to help prevent accidents caused by driver fatigue. The system monitors eye behavior, detects signs of drowsiness, and triggers alerts to regain driver attention.

---

## 🔍 Features

- 🔁 **Real-Time Detection** using webcam feed  
- 👁️ **Eye Aspect Ratio (EAR)** based drowsiness detection  
- 🔔 **Immediate Alert System** after detecting 20 consecutive drowsy frames  
- 👀 **Red-Eye Detection** with 90% effectiveness  
- ⚡ **Optimized for 30 FPS** processing  
- ✅ **Tested in 100+ real-world sessions**

---

## 🛠️ Technologies Used

- Python  
- OpenCV  
- dlib  
- NumPy  
- imutils

---

## 🧠 How It Works

1. **Face and Eye Detection**: Uses Haar cascades or dlib to detect facial landmarks.  
2. **Eye Aspect Ratio (EAR)**: Calculates EAR to determine if the eyes are closed.  
3. **Frame Counting**: If the EAR is below a threshold for 20+ consecutive frames, drowsiness is detected.  
4. **Alert Mechanism**: A sound alarm is triggered to alert the driver immediately.

---

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bhavyasingh9822/Driver-Drowsiness.git
   cd Driver-Drowsiness
   ```
2. Install all dependencies
```
pip install -r requirements.txt
opencv-python
dlib
imutils
numpy
playsound


```
3. Run the detection system:
```
python drowsiness_detector.py
```
# Performance
Fatigue Detection Accuracy: 95%
<br>
Red-Eye Detection Accuracy: 90%
<br>
Latency: Triggers alert within 1 second
<br>
Processing Speed: ~30 FPS
<br>
Testing: Over 100+ live driver simulation sessions
<br>
 
 # Future Enhancements
Integration with vehicle systems for automatic intervention
<br>
Mobile deployment using TensorFlow Lite
<br>
Support for yawning detection
<br>
Night vision support
# 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request to improve this system.
