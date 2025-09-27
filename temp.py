import os
import cv2
import json
import time
import sys
import torch
import numpy as np
import subprocess
import shutil
from transformers import LlavaForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from PIL import Image

# --- Configuration ---
# The model files are expected to be on your SSD (D: drive).
model_path = r"D:\hugging_face_ai_model" # Using raw string with backslashes
SYSTEM_PROMPT = (
    "You are an expert stream analyst. Rate the current video frame based on its **Highlight Potential (1-10)**. "
    "Highlight Potential is defined by **AUDITORY AND EMOTIONAL INTENSITY**, inferred from visual cues. "
    "You MUST prioritize high scores for visual indicators of loud events. "
    "**VISUAL PROXIES FOR AUDIO:** Look for open mouths, visible shock/fear, major on-screen explosions/events, and rapid screen shaking, as these strongly imply screaming or loud game noise. "
    "You MUST use the full range of scores (1 to 10). "
    "**CRITICAL RULE FOR LOW SCORES (1-2)**: You MUST score 1 or 2 if the streamer is NOT showing a strong emotional change (e.g., neutral/idle face), or if the screen content is static, shows a menu, a scorecard, or simple navigation/walking for over 5 seconds. Complex *static* overlays (like VTuber backgrounds) must be scored 1 or 2. "
    "10 = Extreme Intensity (Screaming, clear shock/fear expression, massive in-game explosion/success). "
    "7-9 = High Intensity (Intense focus, rapid action, visible startle, big smile/laugh). "
    "3-6 = Medium Intensity (Mild conversation, minor movement, slightly engaged expression). "
    "1-2 = Low Intensity (Static scene, static scorecard/menu, neutral avatar, idle chat). "
    "RESPOND ONLY with a single JSON object containing the numeric score, like this: "
    '{"score": 1}' 
)

# --- Segmentation Constants ---
HIGH_SCORE_THRESHOLD = 7.0 
MERGE_GAP_SCORE_THRESHOLD = 6.0 # Avg score needed to justify merging two nearby segments
MAX_GAP_TO_MERGE_S = 10.0 # Don't try to merge segments if they are separated by more than 10s
MIN_SEGMENT_DURATION_S = 5.0 # Minimum raw duration (before buffers) for a high-score segment
CONTEXT_PRE_ROLL_SECONDS = 20  # Lead-in for context
CONTEXT_POST_ROLL_SECONDS = 10 # Cool-down for reaction
MINIMUM_FINAL_DURATION_S = 30.0 # Enforce a minimum clip length after buffers

# --- Resource Capping for 12GB VRAM Systems ---
GPU_DEVICE = "cuda" # Simplified device map to force GPU load (cuda:0)

# --- Utility Functions ---

def cleanup_directory(output_dir):
    """Removes the temporary debug_frames subdirectory."""
    debug_dir = os.path.join(output_dir, "debug_frames")
    if os.path.exists(debug_dir):
        try:
            shutil.rmtree(debug_dir)
            print(f"Cleanup: Removed temporary debug directory: {debug_dir}", file=sys.stderr)
        except Exception as e:
            print(f"Cleanup Error: Could not remove {debug_dir}: {e}", file=sys.stderr)

def load_ai_model(path: str):
    """Loads the Llava model, tokenizer, and processor from the specified local path."""
    print(f"Initializing AI model from local path: {path}")

    # Use AutoTokenizer and AutoProcessor for compatibility
    tokenizer = AutoTokenizer.from_pretrained(path)
    processor = AutoProcessor.from_pretrained(path) 

    # Use BitsAndBytesConfig for 4-bit quantization and GPU optimization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    try:
        print("DEBUG: Attempting to call LlavaForConditionalGeneration.from_pretrained...", file=sys.stderr, flush=True)
        
        # Force load to GPU using device_map
        model = LlavaForConditionalGeneration.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map=GPU_DEVICE, # Force load to GPU
            quantization_config=quantization_config
        )
        print("SUCCESS: Model, Tokenizer, and Processor loaded.", file=sys.stderr, flush=True)
        return model, tokenizer, processor
    except Exception as e:
        print(f"FATAL ERROR during model initialization: {e}", file=sys.stderr, flush=True)
        return None, None, None

def run_inference(model, tokenizer, processor, image, prompt, is_scoring=True):
    """
    Handles the core inference logic for both scoring and description.
    Returns the integer score (1-10) or the descriptive string.
    """
    if model is None or tokenizer is None or processor is None:
        return 0 if is_scoring else "Model not initialized."

    # Ensure the image is in the correct format (PIL Image)
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
    else:
        pil_image = image

    try:
        if pil_image is None:
             return 0 if is_scoring else "Image is None."

        # Construct the LLaVA prompt format
        if is_scoring:
            llava_prompt = f"USER: <image>\n{SYSTEM_PROMPT}\n{prompt}\nASSISTANT:"
        else:
            llava_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

        # Use the Processor for Unified Input Preparation
        inputs = processor(text=llava_prompt, images=pil_image, return_tensors='pt')
        
        if 'input_ids' not in inputs:
            print("FATAL INPUT ERROR: 'input_ids' key is missing.", file=sys.stderr)
            return 0 if is_scoring else "Tokenizer failed."
        
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=256 if not is_scoring else 50, # Longer output for description
                pad_token_id=tokenizer.pad_token_id
            )

        if output_ids is None or output_ids.dim() == 0 or output_ids.size(0) == 0:
            return 0 if is_scoring else "Model generation failed."
        
        input_len = inputs['input_ids'].size(1)
        raw_response = tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True).strip()

        if not is_scoring:
            # Return raw description text
            return raw_response

        # --- Structured Output Parsing for Scoring ---
        json_start = raw_response.find('{')
        json_end = raw_response.rfind('}') + 1

        if json_start != -1 and json_end != 0:
            json_str = raw_response[json_start:json_end]
            try:
                data = json.loads(json_str)
                score = int(data.get("score", 0))
                return max(1, min(10, score)) # Clamp score between 1 and 10
            except json.JSONDecodeError:
                print(f"AI returned invalid JSON: {json_str}", file=sys.stderr)
                return 0
        else:
            # If JSON parsing fails, try to aggressively extract a score if the model just output a number
            try:
                score = int(raw_response.strip())
                return max(1, min(10, score))
            except ValueError:
                print(f"AI returned non-JSON/non-numeric response: {raw_response[:50]}...", file=sys.stderr)
                return 0

    except Exception as e:
        print(f"Error during AI inference: {e}", file=sys.stderr)
        return 0 if is_scoring else f"Inference Error: {e}"

def get_highlight_score(model, tokenizer, processor, image, prompt):
    """Alias for scoring inference."""
    return run_inference(model, tokenizer, processor, image, prompt, is_scoring=True)

def get_visual_description(model, tokenizer, processor, image):
    """Function to perform the visual comprehension test."""
    prompt = "Describe the video frame in detail. Focus on the streamer avatar, the game interface, and any visible action or emotional state."
    return run_inference(model, tokenizer, processor, image, prompt, is_scoring=False)

def merge_segments_intelligently(highlight_segments, scores, frame_rate):
    """
    Merges closely spaced segments only if the sampled frames between them 
    maintain an average high score, preventing long, dull bridges.
    """
    if not highlight_segments:
        return []

    final_segments = []
    current_segment = highlight_segments[0]
    
    # Extract only the 5-second sampling points we actually scored
    scored_frames = sorted(scores.keys())

    for i in range(1, len(highlight_segments)):
        next_segment = highlight_segments[i]
        
        # Calculate gap metrics
        gap_frames = next_segment['start_frame'] - current_segment['end_frame']
        gap_time_s = gap_frames / frame_rate

        # 1. If the gap is already too large, finalize current segment and move on.
        if gap_time_s > MAX_GAP_TO_MERGE_S:
            final_segments.append(current_segment)
            current_segment = next_segment
            continue

        # 2. Check the scores in the bridging frames
        # Find the sampled frames that fall between the segments (must be greater than current end and less than next start)
        gap_samples = [
            frame_idx for frame_idx in scored_frames 
            if frame_idx >= current_segment['end_frame'] and frame_idx <= next_segment['start_frame']
        ]
        
        is_gap_high_intensity = False
        
        if not gap_samples:
             # If the gap is too small to contain a full 5-second sample point, 
             # and the gap is small (e.g., less than 5 seconds), we assume continuous action and merge.
             if gap_time_s < 5.0:
                 is_gap_high_intensity = True
        else:
            gap_scores = [scores[frame_idx] for frame_idx in gap_samples]
            avg_gap_score = sum(gap_scores) / len(gap_scores)
            
            is_gap_high_intensity = avg_gap_score >= MERGE_GAP_SCORE_THRESHOLD
            print(f"DEBUG: Segment gap from {current_segment['end_frame']} to {next_segment['start_frame']} (Duration {gap_time_s:.2f}s). Avg Gap Score: {avg_gap_score:.2f} (Merge: {is_gap_high_intensity})", file=sys.stderr)

        
        if is_gap_high_intensity:
            # Merge: Extend the current segment's end time to the next segment's end time
            current_segment['end_frame'] = next_segment['end_frame']
        else:
            # Don't merge: Finalize current segment and start tracking the next one
            final_segments.append(current_segment)
            current_segment = next_segment

    # Append the last segment being tracked
    final_segments.append(current_segment)
    return final_segments

def analyze_video(video_path, output_dir):
    """Analyzes video for highlights using the Llava AI model."""

    # --- Model Loading (Attempt once at the start) ---
    model, tokenizer, processor = load_ai_model(model_path)
    if model is None:
        print("FATAL: AI Model failed to load. Cannot proceed with analysis.", file=sys.stderr)
        return []

    print(f"Analyzing video: {video_path} using local Llava model.")
    print("INFO: Starting video analysis loop.", file=sys.stderr)

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}", file=sys.stderr)
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}", file=sys.stderr)
        return []

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    VIDEO_DURATION_SECONDS = total_frames / frame_rate
    print(f"Video details: FPS={frame_rate}, Total Frames={total_frames}, Duration={VIDEO_DURATION_SECONDS:.2f}s", file=sys.stderr)

    # --- Frame Processing Setup ---
    interval_seconds = 5 
    frame_interval = max(1, int(frame_rate * interval_seconds))
    
    # --- VISUAL COMPREHENSION TEST (Skip initial frames for stability) ---
    start_frame_index = frame_interval
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
    ret, frame = cap.read()
    
    if ret:
        print(f"\n--- AI Visual Comprehension Test (Frame {start_frame_index}) ---", file=sys.stderr)
        description = get_visual_description(model, tokenizer, processor, frame)
        print(f"AI Description: {description}", file=sys.stderr)
        print("----------------------------------------------\n", file=sys.stderr)
    else:
        print(f"WARNING: Could not read frame {start_frame_index} for visual comprehension test.", file=sys.stderr)

    # --- Frame Saving Setup ---
    debug_dir = os.path.join(output_dir, "debug_frames")
    os.makedirs(debug_dir, exist_ok=True)
    print(f"DEBUG: Saving debug frames to: {debug_dir}", file=sys.stderr)

    print(f"INFO: Processing video with total frames: {total_frames}, starting from frame {start_frame_index}, with frame interval: {frame_interval}", file=sys.stderr)
    
    scores = {} # {frame_index: score}
    user_prompt = "Rate the current frame for high-action or highlight potential."
    
    # Start the analysis loop after the skipped frames
    for current_frame_index in range(start_frame_index, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
        ret, frame = cap.read()
        
        if not ret:
            print(f"DEBUG: Failed to read frame at index {current_frame_index}.", file=sys.stderr)
            break
        
        # --- DEBUG: Save the frame to inspect ---
        frame_filename = os.path.join(debug_dir, f"frame_{current_frame_index}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        # Get score from the AI model
        score = get_highlight_score(model, tokenizer, processor, frame, user_prompt)
        
        scores[current_frame_index] = score
        
        # Console output for progress
        progress_percent = (current_frame_index / total_frames) * 100
        print(f"Frame {current_frame_index}/{total_frames} ({progress_percent:.1f}%): AI Score={score}", file=sys.stderr)
        
    cap.release()
    print("Video analysis complete. Finding segments...")
    
    # --- 1. Initial Highlight Segmentation Logic (Simple Thresholding) ---
    highlight_segments = []
    min_duration_frames = int(frame_rate * MIN_SEGMENT_DURATION_S)
    
    current_highlight_start = -1
    all_frames = sorted(scores.keys())
    
    for i in range(len(all_frames)):
        frame_idx = all_frames[i]
        score = scores[frame_idx]
        
        if score >= HIGH_SCORE_THRESHOLD:
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

    # --- 2. Smart Segment Merging ---
    final_segments = merge_segments_intelligently(highlight_segments, scores, frame_rate)
    print(f"Found {len(final_segments)} final segments after smart merging.", file=sys.stderr)
    
    # --- 3. Contextual Segment Calculation ---
    time_segments = []
    for segment in final_segments:
        raw_start_time = segment['start_frame'] / frame_rate
        raw_end_time = segment['end_frame'] / frame_rate
        
        # Apply contextual buffers and clamp times
        buffered_start = max(0.0, raw_start_time - CONTEXT_PRE_ROLL_SECONDS)
        buffered_end = min(VIDEO_DURATION_SECONDS, raw_end_time + CONTEXT_POST_ROLL_SECONDS)

        # Calculate the final duration for the cut
        final_duration = buffered_end - buffered_start

        
        if final_duration < MINIMUM_FINAL_DURATION_S:
             # Skip this segment if the buffered duration is too short
             print(f"Skipping segment: final duration {final_duration:.2f}s is less than minimum {MINIMUM_FINAL_DURATION_S}s.", file=sys.stderr)
             continue
        
        # NOTE: No upper duration cap is enforced.
        
        time_segments.append({
            'start': buffered_start,
            'duration': final_duration
        })

    print(f"Found {len(time_segments)} segments (Contextualized): {time_segments}", file=sys.stderr)

    # --- Video Cutting (Actual FFMPEG Execution) & Path Conversion ---
    output_files = []
    ffmpeg_cmd = "ffmpeg" 
    
    for i, segment in enumerate(time_segments):
        start_time = segment['start']
        duration = segment['duration']
        output_file_name = f"highlight_{i+1}.mp4" 
        output_path_abs = os.path.join(output_dir, output_file_name)
        
        command = [
            ffmpeg_cmd,
            "-y", 
            "-ss", str(start_time),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "copy", 
            "-c:a", "copy",
            output_path_abs
        ]
        
        print(f"Executing ffmpeg command for highlight {i+1}: {' '.join(command)}", file=sys.stderr)
        
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            
            print(f"SUCCESS: Highlight {i+1} cut and saved to {output_path_abs}", file=sys.stderr)
            
            # --- Path Conversion for Frontend/UI ---
            # Convert absolute path to URL relative to the public folder
            path_segments = output_path_abs.split(os.sep)
            try:
                # Find the index of 'public' and include everything after it, then prepend /
                public_index = path_segments.index("public")
                relative_path_parts = path_segments[public_index + 1:] 
                relative_path_url = "/" + "/".join(relative_path_parts)
                output_files.append(relative_path_url)
            except ValueError:
                print(f"WARNING: Could not find 'public' in path segments for URL conversion. Sending absolute path.", file=sys.stderr)
                output_files.append(output_path_abs) 

        except subprocess.CalledProcessError as e:
            print(f"FFMPEG ERROR: Failed to cut highlight {i+1} from {start_time:.2f}s for {duration:.2f}s.", file=sys.stderr)
            print(f"Command: {' '.join(e.cmd)}", file=sys.stderr)
            print(f"Stderr: {e.stderr}", file=sys.stderr)
        except FileNotFoundError:
            print(f"CRITICAL ERROR: 'ffmpeg' command not found. Ensure ffmpeg is installed and available in the system PATH.", file=sys.stderr)
            break
        
    # Final output for Next.js API route (MUST use stdout)
    print("---PYTHON-OUTPUT-START---")
    print(json.dumps(output_files)) # This is the main output Next.js expects
    print("---PYTHON-OUTPUT-END---")

    return output_files

# --- Main Script Execution ---

if __name__ == '__main__':
    print("Python script initialized. Waiting for input from Next.js API route.", flush=True)
    
    output_dir = None
    try:
        if len(sys.argv) > 2:
            video_path = sys.argv[1]
            output_dir = sys.argv[2]
            
            analyze_video(video_path, output_dir)
        else:
            print("ERROR: Script requires video_path and output_dir arguments.", file=sys.stderr, flush=True)
            # Ensure the script exits cleanly if arguments is missing
            print("---PYTHON-OUTPUT-START---")
            print("[]")
            print("---PYTHON-OUTPUT-END---")
            
    except Exception as main_error:
        print(f"CRITICAL PYTHON ERROR in main execution block: {main_error}", file=sys.stderr)
        # We still need to print the output-start/end tags for the API to not hang
        print("---PYTHON-OUTPUT-START---")
        print("[]")
        print("---PYTHON-OUTPUT-END---")
        
    finally:
        # Cleanup runs whether try succeeds or fails, as long as output_dir was set
        if output_dir is not None and os.path.exists(output_dir):
             cleanup_directory(output_dir)
