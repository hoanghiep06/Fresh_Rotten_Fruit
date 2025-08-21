# 🍎 Fresh_Rotten_Fruit Detection

A real-time fruit freshness detection application powered by **EfficientNet-B0** and built with **PyQt5**. This intelligent app analyzes fruits through webcam feed or uploaded images to determine both the type of fruit and its freshness level with high accuracy.

![Fresh_Rotten_Fruit App](screenshot.png)

## ✨ Features

- 🎥 **Real-time Detection**: Live webcam feed with instant fruit analysis
- 📁 **Image Upload**: Upload and analyze fruit images from your device  
- 🔍 **Dual Classification**: Identifies both fruit type and freshness level
- 📊 **Confidence Scoring**: Shows prediction confidence for reliable results
- ⏱️ **Smart Display Logic**: Results persist for 0.5s after fruit is removed
- 🎨 **User-friendly Interface**: Clean PyQt5 GUI with visual feedback

## 🧠 Model Architecture

This application uses **EfficientNet-B0** as the backbone architecture:

- **Base Model**: EfficientNet-B0 (pre-trained on ImageNet)
- **Input Size**: 224×224×3 (optimized for real-time performance)
- **Multi-task Learning**: Dual-head architecture for simultaneous fruit type and freshness classification
- **Preprocessing**: EfficientNet-specific preprocessing pipeline
- **Performance**: Optimized balance between accuracy and inference speed

### Model Details

- **Fruit Classification**: Identifies 6 fruit types (apple, banana, bittergourd, capsicum, orange, tomato)
- **Freshness Detection**: Classifies freshness levels (fresh, stale)
- **Confidence Threshold**: 60% minimum confidence for reliable predictions

## 🎯 How It Works

1. **Camera Mode**: Point your webcam at a fruit within the green rectangle
2. **Image Mode**: Upload a fruit image for instant analysis
3. **EfficientNet Processing**: The model analyzes the 128×128 crop using EfficientNet-B0
4. **Dual Classification**: Simultaneous fruit type and freshness level prediction
5. **Results**: See fruit type, freshness level, and confidence scores

## 🚀 Quick Start

### Prerequisites

- Python 3.8 - 3.11
- Webcam (for real-time detection)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/hoanghiep06/Fresh_Rotten_Fruit.git
   cd Fresh_Rotten_Fruit
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare model files** (place in the same directory as app_pyqt5.py)
   - `fruit_freshness_model.h5` - Your trained TensorFlow model
   - `id2fruit_label.json` - Fruit type labels mapping
   - `freshness_id2label.json` - Freshness level labels mapping

4. **Run the application**
   ```bash
   python app_pyqt5.py
   ```

## 📁 Project Structure

```
Fresh_Rotten_Fruit/
├── app_pyqt5.py              # Main application file
├── requirements.txt          # Python dependencies
├── fruit_freshness_model.h5  # EfficientNet-B0 based model (not included)
├── id2fruit_label.json       # Fruit type labels mapping
├── freshness_id2label.json   # Freshness level labels mapping
├── README.md                 # This file
├── Training.ipynb            # Fine-tuning model
└── convert_img_to_numpy.py   # Convert image that optimise time to read image whenever using for training
```

## 🔧 Configuration

### Model Requirements

The EfficientNet-B0 based model should:
- Accept input shape: `(batch_size, 224, 224, 3)`
- Use EfficientNet preprocessing: `tf.keras.applications.efficientnet.preprocess_input`
- Return two outputs: fruit classification and freshness classification
- Be saved in TensorFlow/Keras `.h5` format
- Trained with transfer learning from ImageNet pretrained EfficientNet-B0

### Label Files Format

**id2fruit_label.json**:
```json
{
    "apple": 0,
    "banana": 1, 
    "bittergourd": 2, 
    "capsicum": 3, 
    "orange": 4, 
    "tomato": 5
}
```

**freshness_id2label.json**:
```json
{
    "fresh": 0,
    "stale": 1
}
```

### Adjustable Parameters

In the code, you can modify:
- `MIN_CONFIDENCE = 0.6` - Minimum confidence threshold
- `CROP_SIZE = (300, 200)` - Detection rectangle size  
- Display timers and colors

## 🎮 Usage

### Camera Mode
1. Click **"Bật Camera"** to start webcam
2. Place fruit in the green rectangle area
3. Wait for detection results with confidence scores
4. Click **"Tắt Camera"** to stop

### Image Mode  
1. Click **"Tải ảnh lên"** to upload an image
2. Select a fruit image from your device
3. View instant analysis results
4. Upload another image or switch to camera mode

## 🛠️ Technical Details

### Architecture
- **GUI Framework**: PyQt5 for cross-platform interface
- **Computer Vision**: OpenCV for image processing and webcam handling
- **Deep Learning**: EfficientNet-B0 via TensorFlow/Keras for fruit classification
- **Threading**: Separate thread for ML inference to prevent UI blocking
- **Preprocessing**: EfficientNet-specific image preprocessing pipeline

### Performance Features
- **EfficientNet Optimization**: Leverages EfficientNet-B0's efficiency for real-time inference
- **Threaded Inference**: ML predictions run in background thread
- **Frame Skipping**: Processes every 10th frame to maintain 30 FPS video
- **Confidence Filtering**: Only displays predictions above 60% confidence
- **Memory Management**: Queue-based frame processing with size limits
- **Input Optimization**: 224×224 input size for fast processing

## 🔍 Troubleshooting

### Common Issues

**Camera not detected**:
- Ensure your webcam is connected and not used by other apps
- Try different camera indices if you have multiple cameras

**Model loading fails**:
- Check that all model files are in the correct directory
- Verify model file integrity and format

**Low detection accuracy**:
- Ensure good lighting conditions
- Place fruit clearly within the detection rectangle  
- Check if your EfficientNet model was trained on similar fruit images
- Verify the preprocessing pipeline matches training preprocessing

**Performance issues**:
- Consider using `tensorflow-cpu` instead of full TensorFlow
- Reduce frame processing frequency by changing the modulo value



## 🙏 Acknowledgments

- **EfficientNet**: Google Research for the efficient and accurate architecture
- **TensorFlow/Keras**: For the deep learning framework and EfficientNet implementation
- **OpenCV**: Computer vision community for image processing tools
- **PyQt5**: Qt team for the cross-platform GUI framework


---

⭐ **If you found this project helpful, please give it a star!**
