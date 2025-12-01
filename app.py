from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import sys
from pathlib import Path
import traceback

# Add backend to sys.path
sys.path.append(str(Path(__file__).parent))

from backend.video_processing.smart_processor import SmartVideoProcessor

app = FastAPI(
    title="🏏 AI Cricket Batting Analyzer",
    description="Advanced AI-powered cricket batting technique analysis",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directories
app.mount("/static", StaticFiles(directory="frontend"), name="static")
app.mount("/processed", StaticFiles(directory="processed"), name="processed")

# Create directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("processed", exist_ok=True)

# Initialize processor with optimized settings
processor = SmartVideoProcessor(frame_skip=2)  # Process every 2nd frame

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main HTML page"""
    try:
        with open("frontend/index.html", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Frontend not found")

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """
    Analyze cricket batting video with comprehensive AI analysis
    
    Returns detailed analysis including:
    - Performance score (0-100)
    - Specific technical issues
    - Coaching recommendations
    - Annotated video
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload a video file (MP4, MOV, AVI)"
        )
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    input_path = f"uploads/{file_id}_{file.filename}"
    output_path = f"processed/{file_id}_analyzed.mp4"
    
    try:
        # Save uploaded file
        print(f"📥 Receiving file: {file.filename}")
        with open(input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        file_size_mb = len(content) / (1024 * 1024)
        print(f"📊 File size: {file_size_mb:.2f} MB")
        
        # Process video with comprehensive analysis
        print("🚀 Starting video analysis...")
        results = processor.process_video(input_path, output_path)
        
        # Add video URL to results
        results["output_video"] = f"/processed/{file_id}_analyzed.mp4"
        results["original_filename"] = file.filename
        results["file_size_mb"] = round(file_size_mb, 2)
        
        print(f"✅ Analysis complete! Score: {results.get('performance_score', 'N/A')}/100")
        
        return JSONResponse(content=results)
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"❌ Error during processing:\n{error_trace}")
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Processing failed: {str(e)}",
                "detail": "Check server logs for more information"
            }
        )
    
    finally:
        # Cleanup uploaded file
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
                print(f"🧹 Cleaned up upload: {input_path}")
            except Exception as e:
                print(f"⚠️ Could not remove upload file: {e}")

@app.get("/processed/{filename}")
async def get_processed_video(filename: str):
    """Serve processed video files"""
    file_path = f"processed/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Processed video not found")
    return FileResponse(
        file_path,
        media_type="video/mp4",
        headers={"Content-Disposition": f"inline; filename={filename}"}
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AI Cricket Batting Analyzer",
        "version": "2.0.0",
        "components": {
            "pose_estimator": "✅ Active",
            "cricket_analyzer": "✅ Active",
            "yolo_detector": "✅ Active" if processor.yolo_model else "⚠️ Disabled"
        }
    }

@app.get("/stats")
async def get_stats():
    """Get processing statistics"""
    processed_videos = len([f for f in os.listdir("processed") if f.endswith(".mp4")])
    return {
        "total_videos_processed": processed_videos,
        "storage_used_mb": sum(
            os.path.getsize(os.path.join("processed", f)) 
            for f in os.listdir("processed")
        ) / (1024 * 1024)
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🏏 AI Cricket Batting Analyzer - Starting Server")
    print("="*60)
    print("📡 Server will be available at: http://localhost:8000")
    print("📖 API docs available at: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

