# 🏏 AI Cricket Batting Analyzer

Advanced AI-powered cricket batting technique analysis using computer vision and deep learning.

## Features

- **Real-time Pose Analysis**: MediaPipe-based 33-point body landmark detection
- **Performance Scoring**: Comprehensive 0-100 scoring system with detailed grading
- **Technical Issue Detection**: Identifies stance, alignment, balance, and form issues
- **Video Annotations**: Annotated output videos with real-time analysis overlays
- **Professional Recommendations**: Actionable coaching tips for technique improvement

## Tech Stack

- **Backend**: FastAPI, Python 3.12
- **Computer Vision**: MediaPipe, OpenCV, YOLOv8
- **Deep Learning**: Ultralytics YOLO, TensorFlow Lite
- **Frontend**: HTML5, CSS3, Vanilla JavaScript

## Installation

### Prerequisites

- Python 3.12+
- pip
- Git

### Setup

1. **Clone the repository**
git clone https://github.com/akhilgarg29/ai-cricket-batting-analyzer.git
cd cricket-batting-analyzer


2. **Create virtual environment**
python -m venv cricket-env


3. **Activate virtual environment**
- Windows: `cricket-env\Scripts\activate`
- Linux/Mac: `source cricket-env/bin/activate`

4. **Install dependencies**
pip install -r requirements.txt


5. **Download YOLO model** (if not included)
The YOLOv8 nano model will auto-download on first run
Or manually download from: https://github.com/ultralytics/assets/releases


## Usage

### Run the application

python app.py


Or using uvicorn directly:
uvicorn app:app --host 0.0.0.0 --port 8000 --reload


### Access the application

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Mobile Access (Same Network)

1. Find your computer's IP address: `ipconfig` (Windows) or `ifconfig` (Linux/Mac)
2. Access from phone: `http://YOUR_IP:8000`

## Project Structure

```cricket-batting-analyzer/
├── backend/
│ ├── pose_analysis/
│ │ ├── init.py
│ │ ├── cricket_analyzer.py
│ │ ├── pose_estimator.py
│ │ └── pose_utils.py
│ └── video_processing/
│ ├── init.py
│ └── smart_processor.py
├── frontend/
│ └── index.html
├── uploads/ # Temporary upload storage
├── processed/ # Processed video output
├── app.py # Main FastAPI application
├── requirements.txt
├── .gitignore
└── README.md```


## API Endpoints

- `POST /analyze` - Upload and analyze cricket batting video
- `GET /processed/{filename}` - Retrieve processed video
- `GET /health` - Health check endpoint
- `GET /stats` - Processing statistics

## Configuration

Edit `config.py` (if created) to adjust:
- Frame skip rate (default: 2)
- Video codec settings
- Analysis thresholds
- Server host/port

## Performance Optimization

- **Frame Skip**: Processes every 2nd frame by default for faster analysis
- **Codec**: Uses H.264 (avc1) for browser-compatible videos
- **GPU Support**: Automatically uses GPU if available for YOLO detection

## Troubleshooting

### Video not playing in browser
- Ensure H.264 codec is available
- Install FFmpeg for better codec support

### Low performance scores
- Adjust thresholds in `cricket_analyzer.py`
- Check camera angle (side view recommended)

### YOLO warnings
- Non-critical warnings about PyTorch 2.6+ security
- Analysis works without YOLO (pose detection only)

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

MIT License - feel free to use for personal or commercial projects

## Author

Akhil - [Your GitHub Profile]

## Acknowledgments

- MediaPipe for pose estimation
- Ultralytics for YOLO
- FastAPI framework
- Cricket coaching community for technique insights
