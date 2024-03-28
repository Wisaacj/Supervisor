# Supervisor

Real-time object detection on a live video feed, streamed from my phone; served in a Flask application.

## Plan
- [x] Implement real-time object detection on a live video feed streamed from a phone
- [ ] Fine-tune an object detection & classification model on my housemates' faces 
- [ ] Implement facial recognition
- [ ] Use facial recognition as a biometric entry system to our uni house
- [ ] Integrate a LLM + TTS engine to greet people as they enter the house
- [ ] Integrate an ASR model for bilateral communication between guests and a LLM

### Facial Recogition
1. **Object Detection:** First, utilize YOLOv8 to detect people in the frame. This will generate output bounding boxes around people.

2. **Face Validation:** Once the people bounding boxes are available, you can use these regions as inputs to your facenet model to verify if the detected person is indeed a face.

3. **Face Recognition:** If a valid face is detected, the facenet model can then recognise who the person is by comparing the facial features with your pre-trained facenet model.

4. **Labeling:** Now, you have the person's face and the name associated with the face. This name can be plotted on top of the bounding box in real-time.

[Source](https://github.com/ultralytics/ultralytics/issues/4187#issuecomment-1666790428)

### Optimisations
One approach you can try to reduce the computational complexity is tracking. You can use a simple Byte Tracker to track faces and run facenet only once for every tracked face box with a unique id. Of course, depending on how many people are present and how they are moving, there can be a loss in accuracy.

[Source](https://github.com/ultralytics/ultralytics/issues/4187#issuecomment-1672018346)


## Resources
[Face Detection with YOLOv8](https://github.com/ultralytics/ultralytics/issues/4187)