# AI/ML Sports Pose Analysis Output

## 1. Approach
This project implements a computer vision pipeline to analyze cricket player movements (batting/bowling) from a side-on video perspective. 

**Pipeline Steps:**
1.  **Input**: Loads a standard video file.
2.  **Pose Estimation**: Utilizes **OpenCV DNN** with **OpenPose MobileNet** (TensorFlow) to detect 18 skeletal landmarks.
3.  **Side Detection**: Automatically determines the active side (Left/Right) based on arm visibility.
4.  **Metric Calculation**: Computes biomechanical angles using vector geometry.
5.  **Visualization**: Overlays a skeleton and real-time metric values on the video.
6.  **Output**: Generates an annotated video (`output_video.mp4`) and a CSV of frame-by-frame metrics (`output_metrics.csv`).

## 2. Model Used & Why
**Model:** OpenPose MobileNet (TensorFlow) via OpenCV DNN.
**Why:**
-   **Robustness:** Works reliably in environments where complex dependencies (like full TensorFlow or MediaPipe) face compatibility issues.
-   **Standard:** Uses the standard COCO keypoint format (18 points).
-   **Portability:** Runs with just `opencv-python` and `numpy`.

## 3. Metrics Defined
We extracted three core metrics relevant to cricket performance:

1.  **Elbow Flexion Angle**:
    -   *Definition*: Angle formed by the Shoulder-Elbow-Wrist.
    -   *Relevance*: Critical for checking "chucking" (illegal bowling action) where the arm must not straighten by more than 15 degrees. In batting, it indicates the backlift lever.
    
2.  **Knee Flexion Angle**:
    -   *Definition*: Angle formed by Hip-Knee-Ankle.
    -   *Relevance*: Measures the depth of the crease stance in batting or the landing impact absorption in bowling delivery stride.
    
3.  **Trunk Inclination (Body Lean)**:
    -   *Definition*: Angle of the torso relative to the vertical axis.
    -   *Relevance*: Important for balance. Bowlers need forward lean for momentum; batters need a stable upright or slightly forward posture.

## 4. Observations & Limitations (Self-Critique)
During testing, the following issues are commonly observed with off-the-shelf models on sports data:

-   **Motion Blur**: High-speed movements (like a bowling arm) often cause landmarks to jitter or drift.
-   **Occlusion**: In a side-on view, one limb often hides the other.
-   **Processing Speed**: CPU inference with DNN can be slower than lightweight optimizations like MediaPipe, but is consistent.

## 5. Improvement Plan
**If given more time & data:**

1.  **Accuracy**: Implement a Kalman Filter to smooth trajectories and reduce jitter.
2.  **Adaptation**: Fine-tune a top-down model (like ViTPose or HRNet) specifically on a dataset of cricket players.
3.  **Data Collection**: Collect high-framerate (60fps+) video from multiple angles.
4.  **Evaluation**: Compare the estimated angles against a gold-standard Vicon motion capture system.

## 6. How to Run
1.  Install dependencies: `pip install -r requirements.txt` (Mainly `opencv-python` and `numpy`)
2.  Place your video as `input.mp4`.
3.  Run: `python main.py --input input.mp4`
    - The script will automatically download the required model file (`graph_opt.pb`) on the first run.
