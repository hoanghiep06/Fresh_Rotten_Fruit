🍎 Fresh_Rotten_Fruit Detection

A real-time fruit freshness detection application powered by EfficientNet-B0 and built with PyQt5. This intelligent app analyzes fruits through webcam feed or uploaded images to determine both the type of fruit and its freshness level with high accuracy.


✨ Features

Real-time Detection: Live webcam feed with instant fruit analysis

Image Upload: Upload and analyze fruit images from your device

Dual Classification: Identifies both fruit type and freshness level

Confidence Scoring: Shows prediction confidence for reliable results

Smart Display Logic: Results persist for 0.5s after fruit is removed

User-friendly Interface: Clean PyQt5 GUI with visual feedback

🧠 Model Architecture
This application uses EfficientNet-B0 as the backbone architecture:

Base Model: EfficientNet-B0 (pre-trained on ImageNet)
Input Size: 128×128×3 (optimized for real-time performance)
Multi-task Learning: Dual-head architecture for simultaneous fruit type and freshness classification
Preprocessing: EfficientNet-specific preprocessing pipeline
Performance: Optimized balance between accuracy and inference speed

Model Details

Fruit Classification: Identifies various fruit types (apple, banana, orange, etc.)
Freshness Detection: Classifies freshness levels (fresh, rotten)
Confidence Threshold: 60% minimum confidence for reliable predictions

🎯 How It Works

Camera Mode: Point your webcam at a fruit within the green rectangle
Image Mode: Upload a fruit image for instant analysis
EfficientNet Processing: The model analyzes the 128×128 crop using EfficientNet-B0
Dual Classification: Simultaneous fruit type and freshness level prediction
Results: See fruit type, freshness level, and confidence scores

🚀 Quick Start
Prerequisites

Python 3.8 - 3.11
Webcam (for real-time detection)

Installation

Clone the repository
bashgit clone [my_project](https://github.com/hoanghiep06/Fresh_Rotten_Fruit.git)
cd Fresh_Rotten_Fruit

Install dependencies
bashpip install -r requirements.txt

Prepare model files (place in the same directory as app_pyqt5.py)

fruit_freshness_model.h5 - Your trained TensorFlow model
id2fruit_label.json - Fruit type labels mapping
freshness_id2label.json - Freshness level labels mapping


Run the application
bashpython app_pyqt5.py


📁 Project Structure
Fresh_Rotten_Fruit/
├── app_pyqt5.py              # Main application file
├── requirements.txt          # Python dependencies
├── fruit_freshness_model.h5  # EfficientNet-B0 based model (not included)
├── id2fruit_label.json       # Fruit type labels mapping
├── freshness_id2label.json   # Freshness level labels mapping
├── README.md                 # This file
├── Training.ipynb            # Fine-tuning model
└── convert_img_to_numpy.py   # Convert image that optimise time to read image whenever using for training   
       
🔧 Configuration
Model Requirements
The EfficientNet-B0 based model should:

Accept input shape: (batch_size, 128, 128, 3)
Use EfficientNet preprocessing: tf.keras.applications.efficientnet.preprocess_input
Return two outputs: fruit classification and freshness classification
Be saved in TensorFlow/Keras .h5 format
Trained with transfer learning from ImageNet pretrained EfficientNet-B0

Label Files Format
id2fruit_label.json:
json{
    "apple": 0,
    "banana": 1, 
    "bittergourd": 2, 
    "capsicum": 3, 
    "orange": 4, 
    "tomato": 5
}

freshness_id2label.json:
json{
    "fresh": 0,
    "stale": 1
}

Adjustable Parameters
In the code, you can modify:

MIN_CONFIDENCE = 0.6 - Minimum confidence threshold
CROP_SIZE = (300, 200) - Detection rectangle size
Display timers and colors

🎮 Usage
Camera Mode

Click "Bật Camera" to start webcam
Place fruit in the green rectangle area
Wait for detection results with confidence scores
Click "Tắt Camera" to stop

Image Mode

Click "Tải ảnh lên" to upload an image
Select a fruit image from your device
View instant analysis results
Upload another image or switch to camera mode

🛠️ Technical Details
Architecture

GUI Framework: PyQt5 for cross-platform interface
Computer Vision: OpenCV for image processing and webcam handling
Deep Learning: EfficientNet-B0 via TensorFlow/Keras for fruit classification
Threading: Separate thread for ML inference to prevent UI blocking
Preprocessing: EfficientNet-specific image preprocessing pipeline

Performance Features

Threaded Inference: ML predictions run in background thread
Frame Skipping: Processes every 10th frame to maintain smooth video
Confidence Filtering: Only shows high-confidence predictions
Memory Management: Queue-based frame processing with size limits

🔍 Troubleshooting
Common Issues
Camera not detected:

Ensure your webcam is connected and not used by other apps
Try different camera indices if you have multiple cameras

Model loading fails:

Check that all model files are in the correct directory
Verify model file integrity and format

Low detection accuracy:

Ensure good lighting conditions
Place fruit clearly within the detection rectangle
Check if your model was trained on similar data

Performance issues:

Consider using tensorflow-cpu instead of full TensorFlow
Reduce frame processing frequency by changing the modulo value


🙏 Acknowledgments

TensorFlow team for the machine learning framework
OpenCV community for computer vision tools
PyQt5 developers for the GUI framework

⭐ If you found this project helpful, please give it a star!
