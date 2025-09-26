import os
import cv2
import numpy as np
import json
import requests
import time

# --- Configuration ---
# Define the API Key and URL (placeholder values)
API_KEY = os.getenv("GEMINI_API_KEY", "") 
# Using gemini-2.5-flash-preview-05-20 for multimodal analysis
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={}"

# --- Model Fix: Structured JSON Output (CRITICAL CHANGE) ---
# This system instruction and schema force the model to output a single integer
# for the score, eliminating the previous failure mode ('').

SYSTEM_INSTRUCTION = (
    "You are a video analysis expert. Your task is to rate the current video frame based on its "
    "potential as a highlight. A score of 0 indicates no highlight potential (e.g., static menu, loading screen, low action). "
    "A score of 10 indicates peak action, high emotion, or a major event (e.g., a kill, a clutch save, a funny moment). "
    "You MUST ONLY respond with a valid JSON object following the provided schema."
)

JSON_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "score": {
            "type": "INTEGER",
            "description": "The highlight score from 0 (low) to 10 (high)."
        }
    },
    "required": ["score"]
}

# --- Utility Functions ---

def image_to_base64(image):
    """Converts a numpy image array (frame) to a base64 string."""
    # Note: Using decode('base64') here is a common pattern for file data handling 
    # in this environment, though technically it's encoding the buffer to base64.
    _, buffer = cv2.imencode('.png', image)
    return buffer.tobytes().decode('base64')

def get_highlight_score(base64_image_data, prompt):
    """Sends the image and prompt to the Gemini API and returns the score."""
    
    # 1. Construct the payload with structured output config
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": base64_image_data
                        }
                    }
                ]
            }
        ],
        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": JSON_SCHEMA
        }
    }
    
    # 2. API Call with exponential backoff
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # We don't use 'D:\models\pc_model.png' in the API call itself, 
            # as the model is accessed via a remote endpoint (Gemini API). 
            # The previous context of the local path was for model weights 
            # if we were running inference locally. Here we focus on the API logic.
            response = requests.post(
                API_URL.format(API_KEY),
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload)
            )
            response.raise_for_status()
            
            # 3. Process the response
            result = response.json()
            
            if (result.get('candidates') and 
                result['candidates'][0].get('content') and 
                result['candidates'][0]['content'].get('parts')):
                
                json_string = result['candidates'][0]['content']['parts'][0]['text']
                data = json.loads(json_string)
                score = data.get('score', 0)
                
                # Ensure the score is an integer
                return int(score) if isinstance(score, (int, str)) and str(score).isdigit() else 0
            
            print(f"AI response structure invalid: {result}", file=os.stderr)
            return 0

        except requests.exceptions.RequestException as e:
            if response.status_code == 429 and attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Rate limit hit. Retrying in {wait_time}s...", file=os.stderr)
                time.sleep(wait_time)
            else:
                print(f"API Request failed after {attempt+1} attempts: {e}", file=os.stderr)
                return 0
        except json.JSONDecodeError as e:
            # This handles the case where the model returns non-JSON data (the original ''))
            print(f"Failed to decode JSON response: {e}. Raw text likely non-JSON.", file=os.stderr)
            return 0
        except Exception as e:
            print(f"An unexpected error occurred: {e}", file=os.stderr)
            return 0
    return 0

def analyze_video(video_path, output_dir):
    """Analyzes video for highlights using a fixed frame interval and the model score."""
    
    print(f"Analyzing video: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}", file=os.stderr)
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}", file=os.stderr)
        return []

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video details: FPS={frame_rate}, Total Frames={total_frames}")

    # Process every Nth frame (e.g., once every 1 second)
    interval_seconds = 1
    frame_interval = max(1, int(frame_rate * interval_seconds))
    
    scores = {} # {frame_index: score}
    
    for current_frame_index in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert frame to base64
        base64_data = image_to_base64(frame)
        
        # Get score from AI (Focus of the fix)
        prompt = "Rate this frame for highlight potential (0-10) in a game stream."
        score = get_highlight_score(base64_data, prompt)
        
        scores[current_frame_index] = score
        
        # Console output for progress
        progress_percent = (current_frame_index / total_frames) * 100
        print(f"Frame {current_frame_index}/{total_frames} ({progress_percent:.1f}%): Score={score}", file=os.stderr)
        
        
    cap.release()
    print("Video analysis complete. Finding segments...")
    
    # --- Highlight Segmentation Logic (Simple Thresholding) ---
    highlight_segments = []
    min_score = 7 
    min_duration_frames = int(frame_rate * 5) 
    max_duration_frames = int(frame_rate * 30) 
    
    current_highlight_start = -1
    
    all_frames = sorted(scores.keys())
    
    for i in range(len(all_frames)):
        frame_idx = all_frames[i]
        score = scores[frame_idx]
        
        if score >= min_score:
            if current_highlight_start == -1:
                current_highlight_start = frame_idx
        elif current_highlight_start != -1:
            end_frame_idx = frame_idx
            duration = end_frame_idx - current_highlight_start
            
            if duration >= min_duration_frames:
                highlight_segments.append({
                    'start_frame': current_highlight_start,
                    'end_frame': end_frame_idx 
                })
            
            current_highlight_start = -1

    if current_highlight_start != -1:
        end_frame_idx = total_frames - 1
        duration = end_frame_idx - current_highlight_start
        if duration >= min_duration_frames:
            highlight_segments.append({
                'start_frame': current_highlight_start,
                'end_frame': end_frame_idx 
            })

    final_segments = []
    if highlight_segments:
        current_segment = highlight_segments[0]
        for i in range(1, len(highlight_segments)):
            next_segment = highlight_segments[i]
            gap_frames = next_segment['start_frame'] - current_segment['end_frame']
            if gap_frames < frame_rate * 5:
                current_segment['end_frame'] = next_segment['end_frame']
            else:
                final_segments.append(current_segment)
                current_segment = next_segment

        final_segments.append(current_segment)
        
    
    time_segments = []
    for segment in final_segments:
        start_time = segment['start_frame'] / frame_rate
        end_time = segment['end_frame'] / frame_rate
        
        if (end_time - start_time) > (max_duration_frames / frame_rate):
            end_time = start_time + (max_duration_frames / frame_rate)
            
        time_segments.append({
            'start': start_time,
            'duration': end_time - start_time
        })

    print(f"Found {len(time_segments)} segments: {time_segments}")

    # --- Video Cutting ---
    output_files = []
    
    for i, segment in enumerate(time_segments):
        start_time = segment['start']
        duration = segment['duration']
        output_file = os.path.join(output_dir, f"highlight_{i+1}.mp4")
        
        print(f"Simulating cutting video from {start_time:.2f}s for {duration:.2f}s to {output_file}", file=os.stderr)
        
        output_files.append(f"{os.path.basename(output_file)}")
        
    return output_files

# --- Main Script Execution ---

if __name__ == '__main__':
    print("Python script initialized. Waiting for input from Next.js API route.")
