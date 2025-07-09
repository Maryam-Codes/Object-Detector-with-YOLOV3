
 🧠🎯 Object Detection with Voice Feedback

A real-time object detection system that identifies objects using **YOLOv3-tiny** and provides **voice feedback** using text-to-speech (TTS). Built with **OpenCV** and **Python**, this project demonstrates the integration of computer vision and audio-based accessibility.



 🚀 Features

* 🔍 Real-time object detection using **YOLOv3-tiny**
* 🎤 Voice alerts using **pyttsx3** for detected objects
* 🎥 Live webcam feed processing
* ⚙️ Optimized for performance and accuracy
* 🧠 Great learning project for AI beginners and accessibility enthusiasts


 🛠 Tech Stack

* Python
* OpenCV
* YOLOv3-tiny
* pyttsx3 (Text-to-Speech)
* NumPy



 📦 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/object-detection-voice-feedback.git
   cd object-detection-voice-feedback
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv3-tiny weights & config**

   * Download the following files:

     * `yolov3-tiny.cfg` from [official repo](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg)
     * `yolov3-tiny.weights` from [here](https://pjreddie.com/media/files/yolov3-tiny.weights)
     * `coco.names` for class labels
   * Place them in the project directory.

 ▶️ Usage

```bash
python detect_and_speak.py
```

* Make sure your webcam is connected.
* Detected objects will be labeled on-screen and spoken aloud via system speakers.


📁 Directory Structure

```
object-detection-voice-feedback/
│
├── detect_and_speak.py
├── yolov3-tiny.cfg
├── yolov3-tiny.weights
├── coco.names
├── requirements.txt
└── README.md
```



🧠 How It Works

* **YOLOv3-tiny** processes each frame from your webcam to detect objects.
* **OpenCV** handles image capture and annotation.
* **pyttsx3** reads the name of each detected object aloud, making the application accessible and interactive.



🎓 Learning Goals

* Hands-on practice with **deep learning for object detection**
* Integration of **TTS for accessibility**
* Real-time performance optimization
* Working with **pre-trained models** and **OpenCV pipelines**



✨ Future Improvements

* Add support for multiple languages (voice output)
* Record detection history
* Improve object filtering to avoid repetitive speech



📄 License

This project is open-source and free to use under the [MIT License](LICENSE).

