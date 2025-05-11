# BlurAI - AI-Powered Image Editor

BlurAI is an intelligent image editor that uses AI to detect and edit objects in images. Built with Python and Streamlit, it provides an intuitive interface for image manipulation with AI-powered object detection.

## Features

- Drag and drop image upload
- AI-powered object detection
- Object-specific editing tools:
  - Blur
  - Square
  - Delete
  - Crop
  - Face blur (for people)
  - Number plate blur (for vehicles)
- Global editing tools:
  - Blur all faces
  - Blur all number plates
- High-resolution download options
- Real-time preview of edits

## Setup

1. Clone the repository:
```bash
git clone https://github.com/snntx/Blur-AI.git
cd Blur-AI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Technologies Used

- Python
- Streamlit
- YOLOv8 (Object Detection)
- OpenCV
- Pillow

## License

MIT License 