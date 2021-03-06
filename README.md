# KCF-Pro
KCF-Pro (kernel correlation filters pro) is an object tracking algorithm for OpenCV built on top of the existing KCF algorithm and uses Linear Regression to handle object overlapping which would cause other object tracking algorithms to fail. 

## Usage
First, place the KCFPro.py file in the directory with your working file, then import it.
```python
import KCFPro
```
Next, Initialize the model just like how you would with cv2 tracking algorithms, except include the video object, and the number of frames to predict ahead when overlap occurs. The value of this depends on how large the objects overlapping are, as well as their speed, which effects the amount of time they are overlapping. For most cases ~20 frames should work. For smaller or quicker objects, lower this number. For larger or slower objects, increase it. 
```python
tracker = KCFPro.TrackerKCFPro_create(video, 20)
```

Finally, intialize the model to a frame and box to track just like you would with the cv2 tracking algorithms.
```python
tracker.init(frame, box)
```

Now, to track an object, use the update method to predict the object at the next frame just like the built in algorithms, or in case of overlap, predict the object at the time where the overlap should finish.
```python
ret, box = tracker.update(frame)
```

## Demo
Below is a demo of a difficult case where there are overlapping objects, as well as rapid change in object profiles (front angle to side angle) and how KCF handles it (first clip), opposed to how KCF-Pro handles it (second clip)


https://user-images.githubusercontent.com/80224102/178884843-b0547648-cbe4-430a-89b1-a5a230081983.mp4

