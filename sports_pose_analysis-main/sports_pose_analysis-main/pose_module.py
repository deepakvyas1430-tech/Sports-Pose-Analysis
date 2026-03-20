import cv2
import numpy as np
import urllib.request
import os

class PoseDetector:
    def __init__(self, model_path="graph_opt.pb", conf_threshold=0.2):
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        self.nPoints = 18
        self.BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                            "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                            "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                            "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }
        self.POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                            ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                            ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                            ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                            ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
        
        # Download model if not exists
        if not os.path.exists(self.model_path):
            print("Model not found. Downloading graph_opt.pb (MobileNet OpenPose)...")
            url = "https://raw.githubusercontent.com/quanhua92/human-pose-estimation-opencv/master/graph_opt.pb"
            try:
                urllib.request.urlretrieve(url, self.model_path)
                print("Download complete.")
            except Exception as e:
                print(f"Failed to download model: {e}")
                raise

        self.net = cv2.dnn.readNetFromTensorflow(self.model_path)

    def findPose(self, img, draw=True):
        frameWidth = img.shape[1]
        frameHeight = img.shape[0]
        
        self.net.setInput(cv2.dnn.blobFromImage(img, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = self.net.forward()
        out = out[:, :19, :, :]
        
        self.points = []
        for i in range(self.nPoints):
            # confidence map of corresponding body's part.
            probMap = out[0, i, :, :]
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            
            if prob > self.conf_threshold:
                self.points.append((int(x), int(y)))
            else:
                self.points.append(None)
                
        if draw:
            for pair in self.POSE_PAIRS:
                partA = pair[0]
                partB = pair[1]
                idA = self.BODY_PARTS[partA]
                idB = self.BODY_PARTS[partB]

                if self.points[idA] and self.points[idB]:
                    cv2.line(img, self.points[idA], self.points[idB], (0, 255, 0), 3)
                    cv2.circle(img, self.points[idA], 5, (0, 0, 255), cv2.FILLED)
                    cv2.circle(img, self.points[idB], 5, (0, 0, 255), cv2.FILLED)
        
        return img

    def findPosition(self):
        # Return simple list format compatible with main.py structure
        # We return a list of dicts to match previous structure slightly, or just list of points
        # To minimize main.py changes, let's keep the list of dicts with 'id', 'x', 'y'
        # BUT note the IDs are different. main.py needs to know THIS IS OPENPOSE IDs.
        
        lmList = []
        for id, point in enumerate(self.points):
            if point:
                lmList.append({
                    'id': id,
                    'x': point[0],
                    'y': point[1],
                    'visibility': 1.0
                })
            else:
                lmList.append({
                    'id': id,
                    'x': 0,
                    'y': 0,
                    'visibility': 0.0
                })
        return lmList
