import time
import cv2


class Detection():
    """Class for representing a detection in a particular image"""

    def __init__(self, frame, label, confidence, x, y, w, h):
        self.frame = frame
        self.label = label
        self.confidence = confidence
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.last_spotted = int(time.time())
        self.has_been_processed_downstream = False

    def get_original_frame(self):
        """Returns original frame"""
        return self.frame

    def get_frame_with_bounding_box(self, frame=None):
        """Returns the frame with a bounding box"""
        if type(frame) == type(None):
            frame = self.frame
        colour = (255, 0, 0)
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), colour, 2)
        text = "{}: {:.4f}".format(self.label, self.confidence)
        font_scale = 1.5
        cv2.putText(frame, text, (self.x, self.y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, colour, 3)
        return frame

    def identify(self):
        """Identify the detection"""
        print('{} detected, {:.4f} confident'.format(self.label, self.confidence))

    def is_person(self):
        return self.label == 'person'

    def is_vehicle(self):
        vehicles = [
            'car',
            'truck',
            'motorbike',
            'bicycle',
            'bus',
        ]
        if self.label in vehicles:
            return True
        return False

    def get_image_file(self):
        """Returns the image file for saving of the frame"""
        ret, buf = cv2.imencode('.jpg', self.frame)
        return buf.tobytes()

    def are_bounding_boxes_similar(self, detection, margin_of_movement=0.2):
        """Compares bounding boxes of self to provided object"""
        frame_height, frame_width = self.frame.shape[:2]
        x_margin_of_error = int(frame_width * margin_of_movement)
        y_margin_of_error = int(frame_height * margin_of_movement)
        if abs(self.x - detection.x) > x_margin_of_error:
            return False
        if abs(self.y - detection.y) > y_margin_of_error:
            return False
        return True

    def is_like_other_detections(self, other_detections, margin_of_movement):
        """Determines if this detection is like other detections"""
        suitable_candidates = [x for x in other_detections if x.label == self.label]
        if not suitable_candidates:
            return False
        for candidate in suitable_candidates:
            if self.are_bounding_boxes_similar(candidate, margin_of_movement):
                return candidate
        return False

    def update_last_spotted(self):
        self.last_spotted = int(time.time())

    def get_frame_cropped(self):
        """Returns the cropped frame of the detection"""
        return self.frame[self.y:self.y+self.h, self.x:self.x+self.w]
