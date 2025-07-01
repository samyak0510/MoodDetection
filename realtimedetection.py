"""
Real-time Facial Emotion Detection System

This module implements a comprehensive real-time facial emotion detection system using 
Convolutional Neural Networks (CNNs) and computer vision techniques. The system supports 
both webcam feeds and video file analysis with an interactive GUI interface.

Author: Samyak Bijal Shah
Project: Crowd Emotion Detection
Institution: Indus University
Date: April 2024
"""

import threading
import cv2
from keras.models import model_from_json
import tkinter as tk
from tkinter import filedialog
import numpy as np
import random
import time
from PIL import Image, ImageTk
import logging
import os

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load pre-trained CNN model for emotion detection
try:
    # Load model architecture from JSON file
    json_file = open("emotiondetector.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    
    # Load trained weights
    model.load_weights("emotiondetector.h5")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# Initialize Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Verify cascade classifier loaded properly
if face_cascade.empty():
    logger.error("Error loading Haar cascade classifier")
    raise ValueError("Failed to load Haar cascade classifier")


def extract_features(image):
    """
    Extract and preprocess features from facial image for CNN input.
    
    This function converts the input image to the required format for the CNN model:
    - Reshapes to (1, 48, 48, 1) for single grayscale image batch
    - Normalizes pixel values to [0, 1] range for optimal neural network performance
    
    Args:
        image (numpy.ndarray): Input facial image (48x48 grayscale)
        
    Returns:
        numpy.ndarray: Preprocessed image array ready for CNN inference
    """
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)  # Reshape for CNN input: (batch_size, height, width, channels)
    return feature / 255.0  # Normalize pixel values to [0, 1] range

def select_video_file():
    """
    Handle video file selection and initiate emotion detection.
    
    Opens a file dialog for user to select video files (MP4, AVI formats).
    Sets up video capture and starts emotion detection process.
    """
    global cap, video_file_path, a
    a = 1  # Flag to indicate video file mode (vs webcam mode)
    
    # Open file dialog with video file filters
    video_file_path = filedialog.askopenfilename(
        title="Select video file", 
        filetypes=[("Video files", "*.mp4 *.avi")]
    )
    
    if video_file_path:
        try:
            cap = cv2.VideoCapture(video_file_path)
            if not cap.isOpened():
                logger.error(f"Unable to open video file: {video_file_path}")
                return
            logger.info(f"Video file loaded: {video_file_path}")
            detect_emotions()
        except Exception as e:
            logger.error(f"Error opening video file: {e}")
    else: 
        logger.info("No file selected")
    


def start_webcam():
    """
    Initialize webcam capture and start real-time emotion detection.
    
    Sets up webcam (default camera index 0) and begins emotion detection process.
    Falls back to alternative camera indices if default fails.
    """
    global cap, a
    a = 0  # Flag to indicate webcam mode (vs video file mode)
    
    try:
        cap = cv2.VideoCapture(0)  # Try default camera
        if not cap.isOpened():
            # Try alternative camera indices
            for i in range(1, 4):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    logger.info(f"Using camera index: {i}")
                    break
            else:
                logger.error("No available camera found")
                return
        
        logger.info("Webcam initialized successfully")
        detect_emotions()
    except Exception as e:
        logger.error(f"Error initializing webcam: {e}")   
            

def detect_emotions():
    """
    Core emotion detection function for processing video frames.
    
    Processes video frames in real-time to detect faces and classify emotions.
    Records emotional states with timestamps for analysis and saves results to file.
    
    Features:
    - Real-time face detection using Haar cascades
    - CNN-based emotion classification
    - Timestamp logging for emotion tracking
    - Dynamic display with bounding boxes and labels
    """
    global cap, average_emotion, a
    
    # Emotion label mapping (indices correspond to CNN output classes)
    labels = {
        0: "angry", 
        1: "fear", 
        2: "happy", 
        3: "neutral", 
        4: "sad", 
        5: "surprise"
    }
    
    # Initialize emotion tracking variables
    recorded_emotions = []           # Stores emotion indices for statistical analysis
    recorded_emotions_with_time = [] # Stores emotions with timestamps for detailed logging
    start_time = time.time()         # Reference time for timestamp calculation
    
    logger.info("Starting emotion detection...")

    # Main video processing loop
    while True:
        ret, im = cap.read()
        if not ret:
            logger.warning("Unable to capture frame - ending detection")
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using Haar cascade classifier
        # Parameters: scaleFactor=1.3, minNeighbors=5 for balanced accuracy/speed
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Process each detected face
        for (p, q, r, s) in faces:
            # Extract face region from grayscale image
            face_region = gray[q:q + s, p:p + r]
            
            # Draw bounding box around detected face
            cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
            
            # Resize face image to model input size (48x48)
            face_resized = cv2.resize(face_region, (48, 48))
            
            # Preprocess image for CNN inference
            processed_face = extract_features(face_resized)
            
            # Predict emotion using trained CNN model
            emotion_predictions = model.predict(processed_face, verbose=0)
            predicted_emotion_idx = emotion_predictions.argmax()
            prediction_label = labels[predicted_emotion_idx]
            
            # Record emotion data for analysis (skip neutral for more meaningful data)
            if predicted_emotion_idx != 3:  # Skip neutral emotions for cleaner data
                recorded_emotions.append(predicted_emotion_idx)
                elapsed_time = round(time.time() - start_time, 2)
                recorded_emotions_with_time.append((predicted_emotion_idx, " at time: ", elapsed_time))
            
            # Display emotion label above face
            cv2.putText(im, prediction_label, (p - 10, q - 10),
                       cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)

        # Display video based on mode (video file vs webcam)
        if a == 1:  # Video file mode
            cv2.namedWindow("Emotion Detection - Video", cv2.WINDOW_NORMAL)
            cv2.imshow("Emotion Detection - Video", im)
        else:  # Webcam mode
            cv2.namedWindow("Emotion Detection - Webcam", cv2.WINDOW_NORMAL)
            cv2.imshow("Emotion Detection - Webcam", im)

        # Exit on ESC key press (key code 27)
        if cv2.waitKey(1) == 27:
            logger.info("Detection stopped by user")
            break     
    # Cleanup resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Process and save emotion data
    logger.info(f"Recorded {len(recorded_emotions_with_time)} emotion detections")
    
    # Save detailed emotion log with timestamps
    try:
        with open("recorded_emotions.txt", "w") as file:
            file.write("# Emotion Detection Log\n")
            file.write("# Format: emotion_index, timestamp_marker, time_seconds\n")
            file.write("# Emotions: 0=angry, 1=fear, 2=happy, 3=neutral, 4=sad, 5=surprise\n")
            for entry in recorded_emotions_with_time:
                emotion_idx = entry[0]
                time_marker = entry[1]
                timestamp = entry[2]
                file.write(f"{emotion_idx},{time_marker},{timestamp}\n")
        logger.info("Emotion data saved to recorded_emotions.txt")
    except Exception as e:
        logger.error(f"Error saving emotion data: {e}")

    # Calculate emotion statistics
    emotion_count = len(recorded_emotions)
    
    if emotion_count > 0:
        # Apply emotion mapping correction (if needed based on training data specifics)
        corrected_emotions = []
        for emotion_idx in recorded_emotions:
            # Note: This mapping appears to be specific to training data peculiarities
            if emotion_idx == 2:      # Happy -> Sad mapping
                corrected_emotions.append(4)
            elif emotion_idx == 4:    # Sad -> Happy mapping  
                corrected_emotions.append(2)
            else:
                corrected_emotions.append(emotion_idx)
        
        # Calculate average emotion score
        emotion_total = sum(corrected_emotions)
        average_emotion = emotion_total / emotion_count
        
        logger.info(f"Average emotion score: {average_emotion:.2f}")
        average_emotion_label.config(text=f"Average Emotion: {average_emotion:.2f}")
        
        # Display emotion distribution
        emotion_counts = {labels[i]: corrected_emotions.count(i) for i in range(6)}
        logger.info(f"Emotion distribution: {emotion_counts}")
    else:
        logger.info("No emotions detected")
        average_emotion_label.config(text="No emotions detected")

def generate_dynamic_background():
    """
    Generate dynamic gradient background for GUI.
    
    Creates smooth color transitions for an aesthetically pleasing interface.
    Runs in a separate thread to avoid blocking the main GUI operations.
    """
    while True:
        try:
            # Generate random starting and ending colors for gradient
            start_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            end_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            # Create smooth color interpolation (101 steps for smooth transition)
            for t in range(101):
                # Linear interpolation between start and end colors
                current_color = tuple(
                    int(start_channel + (end_channel - start_channel) * t / 100) 
                    for start_channel, end_channel in zip(start_color, end_color)
                )
                
                # Create gradient image and update background
                try:
                    gradient = Image.new("RGB", (root.winfo_screenwidth(), root.winfo_screenheight()), current_color)
                    background_image = ImageTk.PhotoImage(gradient)
                    background_label.config(image=background_image)
                    background_label.image = background_image  # Keep reference to prevent garbage collection
                except tk.TclError:
                    # Handle case where GUI is closed during background update
                    return
                
                # Control animation speed (50fps = 0.02s per frame)
                time.sleep(0.02)
        except Exception as e:
            logger.error(f"Background generation error: {e}")
            time.sleep(1)  # Wait before retrying


# GUI Setup and Main Application
def main():
    """
    Initialize and run the main GUI application.
    
    Sets up the graphical user interface with dynamic background, control buttons,
    and information displays for the emotion detection system.
    """
    global root, background_label, average_emotion_label
    
    # Create main application window
    root = tk.Tk()
    root.title("Real-time Facial Emotion Detection System")
    root.geometry("800x600")
    root.resizable(True, True)
    
    # Set window icon and configure
    try:
        root.iconbitmap('icon.ico')  # Add icon if available
    except:
        pass  # Continue without icon if file not found
    
    # Create dynamic background
    background_label = tk.Label(root)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
    
    # Start background animation in separate thread
    background_thread = threading.Thread(target=generate_dynamic_background)
    background_thread.daemon = True  # Thread will close when main program exits
    background_thread.start()
    
    # Define button styling
    button_style = {
        "background": "#2C3E50",
        "foreground": "white",
        "font": ("Helvetica", 12, "bold"),
        "activebackground": "#34495E",
        "activeforeground": "white",
        "relief": "raised",
        "borderwidth": 2,
        "padx": 20,
        "pady": 10
    }
    
    # Create title label
    title_label = tk.Label(
        root, 
        text="Real-time Facial Emotion Detection", 
        font=("Helvetica", 18, "bold"),
        bg="black",
        fg="white"
    )
    title_label.place(relx=0.5, rely=0.1, anchor="center")
    
    # Create emotion mapping information
    info_label = tk.Label(
        root, 
        text="Emotion Classes: 0=Angry, 1=Fear, 2=Happy, 3=Neutral, 4=Sad, 5=Surprise",
        font=("Helvetica", 10),
        bg="black",
        fg="lightgray"
    )
    info_label.place(relx=0.5, rely=0.2, anchor="center")
    
    # Create control buttons
    video_button = tk.Button(
        root, 
        text=" Select Video File", 
        command=select_video_file,
        **button_style
    )
    video_button.place(relx=0.5, rely=0.35, anchor="center")
    
    webcam_button = tk.Button(
        root, 
        text="Start Webcam", 
        command=start_webcam,
        **button_style
    )
    webcam_button.place(relx=0.5, rely=0.5, anchor="center")
    
    # Create results display
    average_emotion_label = tk.Label(
        root, 
        text="Average Emotion: Not calculated",
        font=("Helvetica", 12),
        bg="black",
        fg="yellow"
    )
    average_emotion_label.place(relx=0.5, rely=0.7, anchor="center")
    
    # Create instructions
    instructions = tk.Label(
        root,
        text="Instructions: Select video file or start webcam. Press ESC to stop detection.",
        font=("Helvetica", 10),
        bg="black",
        fg="lightblue"
    )
    instructions.place(relx=0.5, rely=0.85, anchor="center")
    
    # Create footer
    footer = tk.Label(
        root,
        text="Developed by Samyak Bijal Shah | Indus University | 2024",
        font=("Helvetica", 8),
        bg="black",
        fg="gray"
    )
    footer.place(relx=0.5, rely=0.95, anchor="center")
    
    logger.info("GUI initialized successfully")
    
    # Start the main GUI event loop
    try:
        root.mainloop()
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"GUI error: {e}")
    finally:
        # Cleanup
        try:
            if 'cap' in globals() and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
        except:
            pass

if __name__ == "__main__":
    main()