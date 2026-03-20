import cv2
import numpy as np
import argparse
import time
import json
import csv
from pose_module import PoseDetector
from utils import calculate_angle, calculate_vertical_angle, draw_text

def main():
    parser = argparse.ArgumentParser(description="Sports Pose Analysis Pipeline")
    parser.add_argument("--input", type=str, default="input.mp4", help="Path to input video")
    parser.add_argument("--output", type=str, default="output_video.mp4", help="Path to output video")
    parser.add_argument("--csv", type=str, default="output_metrics.csv", help="Path to output CSV")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.input}")
        return

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # Pose Detector (OpenPose MobileNet)
    detector = PoseDetector(conf_threshold=0.2)

    # Data logging
    metrics_data = []
    
    frame_count = 0
    start_time = time.time()

    print(f"Processing video: {args.input}...")

    while True:
        success, img = cap.read()
        if not success:
            break
            
        frame_count += 1
        
        # 1. Pose Detection
        # OpenPose expects blob input, which the module handles
        # However, for speed with OpenPose MobileNet on CPU, we might want to resize input? 
        # The module resizes to (368, 368) internally for inference, but draws on original image.
        
        img = detector.findPose(img, draw=True)
        lmList = detector.findPosition()
        
        # OpenPose MobileNet Keypoints Mapping:
        # 2: RShoulder, 3: RElbow, 4: RWrist
        # 5: LShoulder, 6: LElbow, 7: LWrist
        # 8: RHip, 9: RKnee, 10: RAnkle
        # 11: LHip, 12: LKnee, 13: LAnkle
        # 1: Neck (Common top point)
        
        # Check if enough points are detected
        if len(lmList) > 13:
            # 2. Determine Side (Left or Right)
            # We check which shoulder is more 'visible'. 
            # In our module, visibility is 1.0 or 0.0
            
            vis_right_arm = lmList[2]['visibility'] + lmList[3]['visibility'] + lmList[4]['visibility']
            vis_left_arm = lmList[5]['visibility'] + lmList[6]['visibility'] + lmList[7]['visibility']
            
            # Simple heuristic: assume side analysis based on which arm is more detected, 
            # or default to Right Handed Batter if ambiguous?
            # Let's say:
            if vis_left_arm > vis_right_arm:
                side = "Left"
                shoulder_idx, elbow_idx, wrist_idx = 5, 6, 7
                hip_idx, knee_idx, ankle_idx = 11, 12, 13
            else:
                side = "Right"
                shoulder_idx, elbow_idx, wrist_idx = 2, 3, 4
                hip_idx, knee_idx, ankle_idx = 8, 9, 10

            # Required points must be visible to calculate metrics
            required_indices = [shoulder_idx, elbow_idx, wrist_idx, hip_idx, knee_idx, ankle_idx]
            all_visible = all(lmList[i]['visibility'] > 0 for i in required_indices)
            
            if all_visible:
                # Extract Coordinates
                shoulder = [lmList[shoulder_idx]['x'], lmList[shoulder_idx]['y']]
                elbow = [lmList[elbow_idx]['x'], lmList[elbow_idx]['y']]
                wrist = [lmList[wrist_idx]['x'], lmList[wrist_idx]['y']]
                hip = [lmList[hip_idx]['x'], lmList[hip_idx]['y']]
                knee = [lmList[knee_idx]['x'], lmList[knee_idx]['y']]
                ankle = [lmList[ankle_idx]['x'], lmList[ankle_idx]['y']]
                
                # 3. Calculate Body Metrics
                # Metric 1: Elbow Flexion Angle
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                
                # Metric 2: Knee Flexion Angle
                knee_angle = calculate_angle(hip, knee, ankle)
                
                # Metric 3: Trunk Inclination (Forward Lean)
                # Using Neck if available for top of trunk? Or Shoulder?
                # OpenPose has Neck (1). Let's use Neck -> Hip midpoint.
                # If neck is not available, use shoulder.
                # Actually, strictly "trunk" is usually Shoulder to Hip.
                trunk_angle = calculate_vertical_angle(shoulder, hip)
                
                # Store Data
                metrics_data.append({
                    'frame': frame_count,
                    'time_sec': frame_count / fps,
                    'side': side,
                    'elbow_angle': elbow_angle,
                    'knee_angle': knee_angle,
                    'trunk_angle': trunk_angle
                })
                
                # 4. Visualization
                # Overlay Info Box
                cv2.rectangle(img, (10, 10), (300, 150), (255, 255, 255), -1) 
                cv2.rectangle(img, (10, 10), (300, 150), (0, 0, 0), 2)
                
                draw_text(img, f"Side: {side}", (20, 40), (0, 0, 0), scale=0.7)
                draw_text(img, f"Elbow Ang: {int(elbow_angle)}", (20, 70), (0, 0, 255), scale=0.7)
                draw_text(img, f"Knee Ang: {int(knee_angle)}", (20, 100), (0, 255, 0), scale=0.7)
                draw_text(img, f"Trunk Ang: {int(trunk_angle)}", (20, 130), (255, 0, 0), scale=0.7)
                
                # Visual feedback
                cv2.circle(img, (int(elbow[0]), int(elbow[1])), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (int(knee[0]), int(knee[1])), 10, (0, 255, 0), cv2.FILLED)

        # Write Frame
        out.write(img)

    # Cleanup
    cap.release()
    out.release()
    
    # Save CSV
    if metrics_data:
        keys = metrics_data[0].keys()
        with open(args.csv, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(metrics_data)
        print(f"Metrics saved to {args.csv}")
        
    print(f"Video saved to {args.output}")
    print("Done.")

if __name__ == "__main__":
    main()
