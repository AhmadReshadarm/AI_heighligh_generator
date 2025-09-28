import os
import cv2
import json
import time
import sys
import torch
import numpy as np
import subprocess
import shutil
import gc
from transformers import LlavaForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from PIL import Image

# --- Configuration ---
# The model files are expected to be on your SSD (D: drive).
# model_path = r"D:\hugging_face_ai_model"   # my PC
model_path = r"C:\llava-model" # Using raw string with backslashes

# --- DEFAULT SYSTEM PROMPT (Used if user does not provide one) ---
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert stream analyst. Rate the current video frame based on its **Highlight Potential (1-10)**. "
"Highlight Potential is defined by **AUDITORY AND EMOTIONAL INTENSITY**, which you will infer from **visual cues** alone. A high score indicates a moment likely to be part of an engaging YouTube Short. "
"You **MUST** prioritize high scores for visual indicators of loud events and strong emotional reactions. "
"**Special Rule for VTubers:** Since VTuber avatars have limited emotional range, you **MUST** prioritize the on-screen action, rapid avatar movement, and in-game events as proxies for emotion and intensity. The avatar's expression is a secondary cue. "
"**VISUAL PROXIES FOR AUDIO AND EMOTION:** Look for open mouths, wide eyes, visible shock, fear, or excitement. Also, consider major in-game explosions, rapid screen shaking, and sudden character movements, as these strongly imply loud noises or intense emotional states. "
"You **MUST** use the full range of scores from 1 to 10. "
"**Scoring Criteria Breakdown:** "
"**Score 10: Extreme Intensity** 🤩 The streamer's face (or VTuber's avatar) shows an **extreme** and unmistakable emotional peak, such as screaming, crying tears of laughter, or a look of clear terror. The VTuber avatar is moving erratically and quickly. The on-screen action is at its most chaotic or climactic point, such as a massive, screen-filling explosion, a final boss defeat, or an impossible clutch victory. The camera shakes violently. The visual cues are so strong that a loud, peak event is guaranteed. "
"**Scores 7-9: High Intensity** 😲 The streamer is showing a clear, strong emotional reaction. This includes genuine surprise, intense focus (furrowed brows, biting lips), or a big, toothy grin and excited laughter. The reaction is sudden and visible. For VTubers, this is when their avatar shows a visible emotion (even a limited one) while the on-screen action is intense. The action is rapid and requires significant focus. This might be a fast-paced gunfight, a complex sequence in a rhythm game, or a major in-game event like a building collapsing. "
"**Scores 3-6: Medium Intensity** 🤔 The streamer is engaged in a mild conversation, laughing quietly, or showing a slightly engaged or confused expression. For VTubers, this is the standard 'just playing the game' state with minor avatar movements. There is a moderate level of on-screen activity, such as simple movement, dialogue-heavy scenes, or a slow build-up to an event. This is the **standard 'just playing the game' range.** "
"**Scores 1-2: Low Intensity** 😴 The streamer's face is neutral, idle, or completely static. There is no visible change in their expression. This includes moments where their face is obscured or a VTuber avatar is idle with no on-screen action. The screen is static for more than 5 seconds. This includes menus, scoreboards, inventory screens, simple walking/navigation with no conflict, or a static 'Be Right Back' screen. "
"RESPOND ONLY with a single JSON object containing the numeric score, like this: "
'{"score": 1}'
  # "You are an expert stream analyst. Rate the current video frame based on its **Highlight Potential (1-10)**. "
    # "Highlight Potential is defined by **AUDITORY AND EMOTIONAL INTENSITY**, inferred from visual cues. "
    # "You MUST prioritize high scores for visual indicators of loud events. "
    # "**VISUAL PROXIES FOR AUDIO:** Look for open mouths, visible shock/fear, major on-screen explosions/events, and rapid screen shaking, as these strongly imply screaming or loud game noise. "
    # "You MUST use the full range of scores (1 to 10). "
    # "**CRITICAL RULE FOR LOW SCORES (1-2)**: You MUST score 1 or 2 if the streamer is NOT showing a strong emotional change (e.g., neutral/idle face), or if the screen content is static, shows a menu, a scorecard, or simple navigation/walking for over 5 seconds. Complex *static* overlays (like VTuber backgrounds) must be scored 1 or 2. "
    # "10 = Extreme Intensity (Screaming, clear shock/fear expression, massive in-game explosion/success). "
    # "7-9 = High Intensity (Intense focus, rapid action, visible startle, big smile/laugh). "
    # "3-6 = Medium Intensity (Mild conversation, minor movement, slightly engaged expression). "
    # "1-2 = Low Intensity (Static scene, static scorecard/menu, neutral avatar, idle chat). "
    # "RESPOND ONLY with a single JSON object containing the numeric score, like this: "
    # '{"score": 1}' 
)

# --- Segmentation Constants ---
HIGH_SCORE_THRESHOLD = 7.0 
MERGE_GAP_SCORE_THRESHOLD = 6.0 
MAX_GAP_TO_MERGE_S = 10.0 
MIN_SEGMENT_DURATION_S = 5.0 
CONTEXT_PRE_ROLL_SECONDS = 20  
CONTEXT_POST_ROLL_SECONDS = 10 
MINIMUM_FINAL_DURATION_S = 30.0 

# --- Dynamic Device Detection ---
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"[LOG] INFO: CUDA GPU detected. Using {torch.cuda.get_device_name(0)}.", flush=True)
    LOAD_DTYPE = torch.float16
    USE_QUANTIZATION = True
    DEVICE_MAP_ARG = "auto"
else:
    DEVICE = "cpu"
    print("[LOG] INFO: No CUDA GPU or CUDA environment not enabled. Falling back to CPU.", flush=True)
    LOAD_DTYPE = torch.float32
    USE_QUANTIZATION = False
    DEVICE_MAP_ARG = None

# --- Utility Functions ---

def print_log(message: str):
    """Utility to ensure all output is prefixed for streaming capture."""
    # Print to stdout with a special prefix and immediate flush
    print(f"[LOG] {message}", flush=True)


def cleanup_directory(output_dir):
    """Removes the temporary debug_frames subdirectory."""
    debug_dir = os.path.join(output_dir, "debug_frames")
    if os.path.exists(debug_dir):
        try:
            shutil.rmtree(debug_dir)
            print_log(f"Cleanup: Removed temporary debug directory: {debug_dir}")
        except Exception as e:
            print_log(f"Cleanup Error: Could not remove {debug_dir}: {e}")

def load_ai_model(path: str, device: str, load_dtype: torch.dtype, use_quantization: bool, device_map_arg: str | None):
    """Loads the Llava model, tokenizer, and processor, dynamically configuring for CPU or GPU."""
    print_log(f"Initializing AI model from local path: {path} on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(path)
    processor = AutoProcessor.from_pretrained(path) 

    quantization_config = None
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    try:
        print_log(f"DEBUG: Loading model with dtype={load_dtype} and quantization={use_quantization}...")
        
        model = LlavaForConditionalGeneration.from_pretrained(
            path,
            torch_dtype=load_dtype,
            quantization_config=quantization_config if use_quantization else None,
            device_map=device_map_arg 
        )
        
        if device_map_arg is None:
            model.to(device)

        print_log(f"SUCCESS: Model, Tokenizer, and Processor loaded onto {model.device}.")
        return model, tokenizer, processor
    except Exception as e:
        print_log(f"FATAL ERROR during model initialization: {e}")
        if device == 'cpu':
            print_log("HINT: If this is a memory error on CPU, try a smaller model.")
        else:
             print_log("HINT: If this is a CUDA error, ensure your driver and CUDA toolkit are compatible with your PyTorch installation.")
        return None, None, None

def run_inference(model, tokenizer, processor, image, system_prompt: str, user_prompt: str, is_scoring=True):
    """
    Handles the core inference logic for both scoring and description, using a dynamic system prompt.
    Includes memory and cache cleanup for performance.
    """
    if model is None or tokenizer is None or processor is None:
        return 0 if is_scoring else "Model not initialized."

    # Ensure the image is in the correct format (PIL Image)
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
    else:
        pil_image = image

    output_ids = None # Initialize outside try for cleanup
    inputs = None

    try:
        if pil_image is None:
             return 0 if is_scoring else "Image is None."

        # Construct the LLaVA prompt format using the dynamic system_prompt
        llava_prompt = f"USER: <image>\n{system_prompt}\n{user_prompt}\nASSISTANT:"

        inputs = processor(text=llava_prompt, images=pil_image, return_tensors='pt')
        
        if 'input_ids' not in inputs:
            print_log("FATAL INPUT ERROR: 'input_ids' key is missing.")
            return 0 if is_scoring else "Tokenizer failed."
        
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=256 if not is_scoring else 50,
                pad_token_id=tokenizer.pad_token_id
            )

        if output_ids is None or output_ids.dim() == 0 or output_ids.size(0) == 0:
            raw_response = "Model generation failed."
            score = 0
        else:
            input_len = inputs['input_ids'].size(1)
            raw_response = tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True).strip()

            if not is_scoring:
                score = 0
            else:
                # --- Structured Output Parsing for Scoring ---
                json_start = raw_response.find('{')
                json_end = raw_response.rfind('}') + 1

                if json_start != -1 and json_end != 0:
                    json_str = raw_response[json_start:json_end]
                    try:
                        data = json.loads(json_str)
                        score = int(data.get("score", 0))
                        score = max(1, min(10, score))
                    except json.JSONDecodeError:
                        print_log(f"AI returned invalid JSON: {json_str}")
                        score = 0
                else:
                    try:
                        score = int(raw_response.strip())
                        score = max(1, min(10, score))
                    except ValueError:
                        print_log(f"AI returned non-JSON/non-numeric response: {raw_response[:50]}...")
                        score = 0
        
        return score if is_scoring else raw_response


    except Exception as e:
        print_log(f"Error during AI inference: {e}")
        return 0 if is_scoring else f"Inference Error: {e}"
    finally:
        # CRITICAL Cleanup for GPU Performance
        if inputs is not None:
            del inputs
        if output_ids is not None:
            del output_ids
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 
        gc.collect() 

def get_highlight_score(model, tokenizer, processor, image, system_prompt, prompt):
    """Alias for scoring inference."""
    return run_inference(model, tokenizer, processor, image, system_prompt, prompt, is_scoring=True)

def get_visual_description(model, tokenizer, processor, image, system_prompt):
    """Function to perform the visual comprehension test."""
    prompt = "Describe the video frame in detail. Focus on the streamer avatar, the game interface, and any visible action or emotional state."
    # The system prompt is ignored here, as description uses a simple prompt format, but pass it to satisfy signature.
    return run_inference(model, tokenizer, processor, image, system_prompt, prompt, is_scoring=False)

# ... (merge_segments_intelligently remains unchanged)

def analyze_video(video_path, output_dir, system_prompt: str):
    """Analyzes video for highlights using the Llava AI model."""

    # --- Model Loading (Attempt once at the start) ---
    model, tokenizer, processor = load_ai_model(model_path, DEVICE, LOAD_DTYPE, USE_QUANTIZATION, DEVICE_MAP_ARG)
    if model is None:
        print_log("FATAL: AI Model failed to load. Cannot proceed with analysis.")
        return []

    print_log(f"Analyzing video: {video_path} using local Llava model.")
    print_log(f"INFO: Model is running on device: {model.device}")

    if not os.path.exists(video_path):
        print_log(f"Error: Video file not found at {video_path}")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print_log(f"Error opening video file: {video_path}")
        return []

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    VIDEO_DURATION_SECONDS = total_frames / frame_rate
    print_log(f"Video details: FPS={frame_rate}, Total Frames={total_frames}, Duration={VIDEO_DURATION_SECONDS:.2f}s")

    # --- Frame Processing Setup ---
    interval_seconds = 5 
    frame_interval = max(1, int(frame_rate * interval_seconds))
    
    # --- VISUAL COMPREHENSION TEST (Skip initial frames for stability) ---
    start_frame_index = frame_interval
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
    ret, frame = cap.read()
    
    if ret:
        print_log(f"\n--- AI Visual Comprehension Test (Frame {start_frame_index}) ---")
        description = get_visual_description(model, tokenizer, processor, frame, system_prompt)
        print_log(f"AI Description: {description}")
        print_log("----------------------------------------------\n")
    else:
        print_log(f"WARNING: Could not read frame {start_frame_index} for visual comprehension test.")

    # --- Frame Saving Setup ---
    debug_dir = os.path.join(output_dir, "debug_frames")
    os.makedirs(debug_dir, exist_ok=True)
    print_log(f"DEBUG: Saving debug frames to: {debug_dir}")

    print_log(f"INFO: Processing video with total frames: {total_frames}, starting from frame {start_frame_index}, with frame interval: {frame_interval}")
    
    scores = {} # {frame_index: score}
    user_prompt = "Rate the current frame for high-action or highlight potential."
    
    # --- Pre-loop Cleanup for stable start on GPU ---
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    # --- END Cleanup ---

    # Start the analysis loop after the skipped frames
    for current_frame_index in range(start_frame_index, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
        ret, frame = cap.read()
        
        if not ret:
            print_log(f"DEBUG: Failed to read frame at index {current_frame_index}.")
            break
        
        # --- DEBUG: Save the frame to inspect ---
        frame_filename = os.path.join(debug_dir, f"frame_{current_frame_index}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        # Get score from the AI model
        score = get_highlight_score(model, tokenizer, processor, frame, system_prompt, user_prompt)
        
        scores[current_frame_index] = score
        
        # Console output for progress
        progress_percent = (current_frame_index / total_frames) * 100
        print_log(f"Frame {current_frame_index}/{total_frames} ({progress_percent:.1f}%): AI Score={score}")
        
    cap.release()
    print_log("Video analysis complete. Finding segments...")
    
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
    print_log(f"Found {len(final_segments)} final segments after smart merging.")
    
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
              print_log(f"Skipping segment: final duration {final_duration:.2f}s is less than minimum {MINIMUM_FINAL_DURATION_S}s.")
              continue
        
        time_segments.append({
            'start': buffered_start,
            'duration': final_duration
        })

    print_log(f"Found {len(time_segments)} segments (Contextualized): {time_segments}")

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
        
        print_log(f"Executing ffmpeg command for highlight {i+1}...")
        
        try:
            # Running with check=True to raise exception on non-zero exit code
            subprocess.run(command, check=True, capture_output=True, text=True)
            
            print_log(f"SUCCESS: Highlight {i+1} cut and saved to {output_path_abs}")
            
            # --- Path Conversion for Frontend/UI ---
            path_segments = output_path_abs.split(os.sep)
            try:
                public_index = path_segments.index("public")
                relative_path_parts = path_segments[public_index + 1:] 
                relative_path_url = "/" + "/".join(relative_path_parts)
                output_files.append(relative_path_url)
            except ValueError:
                print_log(f"WARNING: Could not find 'public' in path segments for URL conversion. Sending absolute path.")
                output_files.append(output_path_abs) 

        except subprocess.CalledProcessError as e:
            print_log(f"FFMPEG ERROR: Failed to cut highlight {i+1}.")
            print_log(f"Stderr: {e.stderr}")
        except FileNotFoundError:
            print_log(f"CRITICAL ERROR: 'ffmpeg' command not found. Ensure ffmpeg is installed and available in the system PATH.")
            break
        
    # Final output for Next.js API route (MUST use stdout, without the [LOG] prefix)
    print("---PYTHON-OUTPUT-START---")
    print(json.dumps(output_files)) # This is the main output Next.js expects
    print("---PYTHON-OUTPUT-END---")

    return output_files

# --- Main Script Execution ---

if __name__ == '__main__':
    print_log(f"Python script initialized. Running on {DEVICE}.")
    
    output_dir = None
    try:
        # Expect 3 arguments now: video_path, output_dir, custom_prompt (optional)
        if len(sys.argv) >= 3:
            video_path = sys.argv[1]
            output_dir = sys.argv[2]
            
            # Get the user-provided prompt or use the default
            custom_prompt = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3].strip() else DEFAULT_SYSTEM_PROMPT
            
            if custom_prompt == DEFAULT_SYSTEM_PROMPT:
                print_log("INFO: Using default system prompt.")
            else:
                print_log("INFO: Using custom system prompt provided by user.")

            analyze_video(video_path, output_dir, custom_prompt)
        else:
            print_log("ERROR: Script requires video_path and output_dir arguments.")
            print("---PYTHON-OUTPUT-START---")
            print("[]")
            print("---PYTHON-OUTPUT-END---")
            
    except Exception as main_error:
        print_log(f"CRITICAL PYTHON ERROR in main execution block: {main_error}")
        print("---PYTHON-OUTPUT-START---")
        print("[]")
        print("---PYTHON-OUTPUT-END---")
        
    finally:
        if output_dir is not None and os.path.exists(output_dir):
              cleanup_directory(output_dir)
