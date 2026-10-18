# ⚡ HandCommand

**HandCommand** is an advanced computer vision application designed to control your computer, media, and applications in real-time using natural hand gestures, secured by strict biometric facial authentication.

---

## ✨ Key Features

* **Biometric Security:** Real-time facial recognition authentication at startup using `face_recognition` to ensure authorized access.
* **Gesture Control:** Powered by **OpenCV** and **MediaPipe**, tracking hand landmarks to trigger system actions smoothly.
* **Non-Blocking HUD:** A sleek, context-aware floating HUD overlay built with **Tkinter** that displays active modes and contextual help menus.
* **Portable Executable:** Compiled into a standalone `.exe` using **PyInstaller**, allowing seamless system-wide launching (e.g., via `Win + R`).
* **Modular Architecture:** Clean separation of concerns featuring dedicated handlers for modes, detection, security, and OS automation actions.

---

## 🛠️ Tech Stack

* **Language:** Python 3.12+
* **Computer Vision & AI:** OpenCV, MediaPipe, `face_recognition`
* **GUI / Overlay:** Tkinter (Custom modern dark styling)
* **Automation:** `pyautogui` / system automation bindings
* **Packaging:** PyInstaller
  
