from fastapi import FastAPI
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import uuid
from services import MLInferenceService

app = FastAPI()

# cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/analyze-video")
async def analyze_video(video: UploadFile = File(...)):
    # Create a temporary file to store the uploaded video
    temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp4")
    
    try:
        # Save uploaded file to temp location
        with open(temp_file, "wb") as buffer:
            buffer.write(await video.read())
        
        # Process the video and get results
        ml_service = MLInferenceService(temp_file)
        results = ml_service.predict(temp_file)

        return results.model_dump()
        
    finally:
        # Clean up the temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)