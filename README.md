#cosc448-ar-emotion-detection
Directed studies project (COSC 448 at UBC Okanagan) supervised by Dr. Khalad Hasan.

#project overview
This project builds an augmented reality system that detects facial expressions and overlays emotional cues during a face-to-face conversation between users wearing Meta Quest Pro headsets. The seven emotions we are looking for are happiness, sadness, anger, fear, surprise, disgust, and contempt.

#how it works
The Meta Quest Pro uses its inward-facing infrared sensors to read 63 FACS (Facial Action Coding System) blendshape values per frame through the OVRFaceExpressions API. These values are classified into one of the seven target emotions. The planned architecture we are going to follow is user A's headset reads their facial expression data and sends it over the network to user B's headset, which displays user A's detected emotion as an AR overlay in user B's view.

#current status
Week 5 - real-time emotion detection pipeline on a single Quest Pro. Detected emotion is displayed as a text label in the user's view.

#tech stack
* Unity (Meta XR Core SDK, Movement SDK)
* C# for the detection pipeline
* Meta Quest Pro

#repository structure
* `/Assets/Scripts` - C# scripts for face tracking and emotion classification
* Unity project files added once development begins in the lab
