import numpy as np
import time
import cv2
import os
from .detection import Detection


class ObjectDetector():
    """Class for running yolov3-tiny implementation"""

    def __init__(self):
        self._set_up()

    def __del__(self):
        self._clean_up()

    def _set_up(self):
        """Sets up resources"""
        self._set_defaults()
        self._load_labels()
        self._load_model_to_memory()

    def _set_defaults(self):
        this_files_dir = os.path.dirname(os.path.realpath(__file__))
        self.network = None
        self.confidence_threshold = 0.5
        self.object_has_vanished_timeout_seconds = 5
        self.non_maxima_supression_threshold = 0.3
        self.object_margin_of_movement = 0.3
        self.model_weights_path = os.path.join(this_files_dir, 'mlmodels/yolov3-tiny.weights')
        self.model_config_path = os.path.join(this_files_dir, 'mlmodels/yolov3-tiny.cfg')
        self.labels_path = os.path.join(this_files_dir, 'mlmodels/coco.names')
        self.current_detections = []
        self.callback_object_detected = None
        self.callback_object_expired = None

    def _clean_up(self):
        """Cleans up resources"""
        try:
            if self.video_writer:
                self.video_writer.release()
            if self.video_stream:
                self.video_stream.release()
        except Exception:
            print('Error when shutting down')

    def _load_labels(self):
        """Loads labels for model"""
        self.labels = open(self.labels_path).read().strip().split('\n')

    def _load_model_to_memory(self):
        """Load the ML model into memory"""
        self.network = cv2.dnn.readNetFromDarknet(self.model_config_path, self.model_weights_path)
        self.layer_names = [self.network.getLayerNames()[i[0] - 1] for i in self.network.getUnconnectedOutLayers()]

    def _is_within_acceptable_confidence(self, confidence):
        """Determines if prediction is within acceptable confidence"""
        return confidence >= self.confidence_threshold

    def _apply_non_maxima_supression(self, box_list, confidence_list):
        """Applies non_maxima supression to results"""
        return cv2.dnn.NMSBoxes(
            box_list,
            confidence_list,
            self.confidence_threshold,
            self.non_maxima_supression_threshold
        )

    def _calculate_dimensions(self, detection, frame_height, frame_width):
        """Calculates dimensions around the current prediction"""
        box = detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
        (center_x, center_y, prediction_width, prediction_height) = box.astype("int")
        # get top left corner
        x = int(center_x - (prediction_width / 2))
        y = int(center_y - (prediction_height / 2))
        return [x, y, int(prediction_width), int(prediction_height)]

    def process_frame(self, frame, draw_bounding_boxes=False):
        """Process and return a frame and results"""
        box_list = []
        confidence_list = []
        class_id_list = []
        results_list = []
        frame_height, frame_width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.network.setInput(blob)
        start_time = time.time()
        outputs = self.network.forward(self.layer_names)
        end_time = time.time()
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if self._is_within_acceptable_confidence(confidence):
                    box_list.append(self._calculate_dimensions(detection, frame_height, frame_width))
                    confidence_list.append(float(confidence))
                    class_id_list.append(class_id)
        frame_processing_time = end_time - start_time
        results = self._apply_non_maxima_supression(box_list, confidence_list)
        if len(results) > 0:
            for i in results.flatten():
                label = self.labels[class_id_list[i]]
                confidence = confidence_list[i]
                detection = Detection(
                    frame=frame,
                    label=label,
                    confidence=confidence,
                    x=box_list[i][0],
                    y=box_list[i][1],
                    w=box_list[i][2],
                    h=box_list[i][3]
                )
                results_list.append(detection)
                if draw_bounding_boxes:
                    frame = detection.get_frame_with_bounding_box(frame)
        self.update_current_detections(results_list)
        return results_list, frame

    def get_current_detections(self):
        """Returns the list of current detections"""
        return self.current_detections

    def add_new_detection(self, detection):
        """Append a new detection to the list"""
        self.current_detections.append(detection)
        if self.callback_object_detected:
            self.callback_object_detected(detection)

    def update_current_detections(self, raw_detections):
        """Updates the current detections list from the latest frame
        Compares objects to determine if they are novel or existing
        """
        current_detections_labels = [x.label for x in self.current_detections]
        for raw_detection in raw_detections:
            if raw_detection.label in current_detections_labels:
                # This object has potentially been detected before
                # and warrants further investigation
                existing_detection = raw_detection.is_like_other_detections(
                    self.current_detections, self.object_margin_of_movement)
                if existing_detection:
                    # This object has been seen before
                    existing_detection.update_last_spotted()
                else:
                    # This object is too different than what we have, add it to the list.
                    self.add_new_detection(raw_detection)
            else:
                # This object is new to us, add it to the list.
                self.add_new_detection(raw_detection)
        for detection in self.current_detections:
            if int(time.time()) > (detection.last_spotted + self.object_has_vanished_timeout_seconds):
                if self.callback_object_expired:
                    self.callback_object_expired(detection)
                self.current_detections.remove(detection)
    
    def set_callbacks(self, object_found_callback, object_expired_callback):
        """Set callbacks for an object being found, and expiring"""
        self.callback_object_detected = object_found_callback
        self.callback_object_expired = object_expired_callback
