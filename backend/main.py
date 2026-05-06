import os
import asyncio
import json
from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from services.llm import get_ai_response, generate_study_guide
from services.tts import generate_audio
from services.resume_processor import extract_text_from_pdf # Ensure this file exists
from deepgram import DeepgramClient

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage
sessions = {}
# Global dictionary to store parsed resume text keyed by client_id
resume_vault = {} 

@app.post("/upload-resume/{client_id}")
async def upload_resume(client_id: str, file: UploadFile = File(...)):
    """Endpoint to receive, parse, and store resume text for RAG."""
    file_bytes = await file.read()
    
    # Extract text using our service
    text = extract_text_from_pdf(file_bytes)
    
    if not text:
        return {"status": "error", "message": "Could not read PDF content."}
    
    # Store text for this specific user session
    resume_vault[client_id] = text
    print(f"Resume stored for {client_id}. Length: {len(text)} characters.")
    
    return {"status": "success", "message": "Resume context uploaded."}

# NEW: The Report Generation Endpoint
@app.get("/api/report/{client_id}")
async def get_report(client_id: str):
    """Generates a study guide based on the user's interview history."""
    if client_id not in sessions:
        return {"guide": []}
    
    print(f"Generating report for {client_id}...")
    report_data = await generate_study_guide(sessions[client_id])
    return report_data

@app.websocket("/ws/interview/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    sessions[client_id] = [] # Initialize/Reset history
    
    deepgram = DeepgramClient()

    try:
        while True:
            # 1. Receive Audio Blob from Frontend
            data = await websocket.receive_bytes()
            
            if len(data) < 1000:
                continue
            
            try:
                # 2. Transcribe using Deepgram
                response = deepgram.listen.v1.media.transcribe_file(
                    request=data,
                    model="nova-2",
                    smart_format=True
                )
                transcript = response.results.channels[0].alternatives[0].transcript
            except Exception:
                continue 

            if not transcript or len(transcript) < 2:
                continue
            
            print(f"Candidate ({client_id}): {transcript}")
            
            # 3. Retrieve RAG context (if it exists)
            user_resume_context = resume_vault.get(client_id, "")
            
            # 4. Get LLM Response with Resume context
            ai_text, updated_history = await get_ai_response(
                transcript, 
                sessions[client_id], 
                user_resume_context
            )
            sessions[client_id] = updated_history
            
            print(f"AI: {ai_text}")

            # 5. Generate Audio (TTS)
            audio_path = await generate_audio(ai_text)
            
            # 6. Send Metadata and Audio back
            with open(audio_path, "rb") as audio_file:
                audio_data = audio_file.read()
                await websocket.send_text(json.dumps({"text": ai_text}))
                await websocket.send_bytes(audio_data)
            
            os.remove(audio_path) # Cleanup temporary audio file

    except Exception as e:
        print(f"Connection closed for {client_id}: {e}")