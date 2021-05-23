# Object Daddy
A simple, python-based object recognizer, basically an implementation of yolo-v3-tiny. This package is used for personal projects; I don't maintain it, so use at your own risk, and feel free to fork.


## Usage
Basic usage is as follows. This project is built to work with rtsparty (also on pypi).

You can get and recognize one frame using the following:
```python
from rtsparty import Stream
from objectdaddy import Daddy


stream = Stream()
daddy = Daddy()

frame = stream.get_frame()
results, frame = daddy.process_frame(frame)
for detection in results:
    detection.identify()
```

For continuous recognition, use the following code.
```python
from rtsparty import Stream
from objectdaddy import Daddy


stream = Stream()
daddy = Daddy()


try:
    while True:
        frame = stream.get_frame()
        if stream.is_frame_empty(frame):
            continue
        detector.process_frame(frame, bounding_boxes)
        for detection in detector.get_current_detections():
            if not detection.has_been_processed_downstream:
                detection.identify()
                detection.has_been_processed_downstream = True
except KeyboardInterrupt:
    pass
```

The code makes the attempt to recognize objects across multiple frames, and keep a list of objects in memory accessed with the get_current_detections() function.


## Notes
- This package supports only CPU-based inference.
