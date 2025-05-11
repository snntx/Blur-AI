# BlurAI - AI-Powered Image Editor

BlurAI is an intelligent image editor that uses AI to detect objects in images and provides selective editing capabilities. It allows users to apply various effects to specific objects or the entire image.

## Features

- Drag and drop image upload
- AI-powered object detection
- Selective object editing
- Global image editing tools
- High-resolution download options
- Real-time preview of edits

## Tech Stack

- Backend: Python (FastAPI)
- Frontend: React with TypeScript
- Object Detection: YOLOv8
- Image Processing: OpenCV, Pillow

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/snntx/Blur-AI.git
cd Blur-AI
```

2. Set up the backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up the frontend:
```bash
cd frontend
npm install
```

4. Run the development servers:

Backend:
```bash
cd backend
uvicorn main:app --reload
```

Frontend:
```bash
cd frontend
npm run dev
```

## License

MIT License 