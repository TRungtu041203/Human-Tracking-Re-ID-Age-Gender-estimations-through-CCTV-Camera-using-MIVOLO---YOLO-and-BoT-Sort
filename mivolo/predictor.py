from collections import defaultdict
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
import tqdm
from mivolo.model.mi_volo import MiVOLO
from mivolo.model.yolo_detector import Detector
from mivolo.structures import AGE_GENDER_TYPE, PersonAndFaceResult
import argparse
import json
import os


class TemporalSmoother:
    """Applies temporal smoothing to age and gender predictions for tracked objects."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.age_history: Dict[int, List[float]] = defaultdict(list)
        self.gender_history: Dict[int, List[float]] = defaultdict(list)  # 0=female, 1=male
    
    def update_and_smooth(self, track_id: int, age: float, gender_prob: float) -> Tuple[float, float]:
        """Update history and return smoothed predictions.
        
        Args:
            track_id: Unique track ID
            age: Predicted age
            gender_prob: Gender probability (0-1, where 1=male, 0=female)
            
        Returns:
            Tuple of (smoothed_age, smoothed_gender_prob)
        """
        # Update age history
        self.age_history[track_id].append(age)
        if len(self.age_history[track_id]) > self.window_size:
            self.age_history[track_id].pop(0)
            
        # Update gender history
        self.gender_history[track_id].append(gender_prob)
        if len(self.gender_history[track_id]) > self.window_size:
            self.gender_history[track_id].pop(0)
            
        # Smooth using moving average
        smoothed_age = np.mean(self.age_history[track_id])
        smoothed_gender = np.mean(self.gender_history[track_id])
        
        return float(smoothed_age), float(smoothed_gender)
    
    def clear_old_tracks(self, active_track_ids: set):
        """Remove history for tracks that are no longer active."""
        all_track_ids = set(self.age_history.keys()) | set(self.gender_history.keys())
        old_track_ids = all_track_ids - active_track_ids
        
        for track_id in old_track_ids:
            if track_id in self.age_history:
                del self.age_history[track_id]
            if track_id in self.gender_history:
                del self.gender_history[track_id]


class GlobalTemporalSmoother:
    """Collects all predictions for each track across the entire video and provides final aggregated results."""
    
    def __init__(self):
        self.all_age_predictions: Dict[int, List[float]] = defaultdict(list)
        self.all_gender_predictions: Dict[int, List[float]] = defaultdict(list)  # 0=female, 1=male
        self.track_final_results: Dict[int, Tuple[float, str]] = {}  # track_id -> (final_age, final_gender)
        self.frame_predictions: List[Dict[int, Tuple[float, str]]] = []  # Store predictions per frame
        self.progressive_results: Dict[int, Tuple[float, str]] = {}  # Running average results
    
    def add_predictions(self, track_id: int, age: float, gender_prob: float):
        """Add a prediction for a track ID."""
        self.all_age_predictions[track_id].append(age)
        self.all_gender_predictions[track_id].append(gender_prob)
        
        # Update progressive results (running average)
        current_age = float(np.mean(self.all_age_predictions[track_id]))
        current_gender_prob = float(np.mean(self.all_gender_predictions[track_id]))
        current_gender = "male" if current_gender_prob > 0.5 else "female"
        self.progressive_results[track_id] = (current_age, current_gender)
    
    def get_progressive_results(self, track_id: int) -> Tuple[float, str]:
        """Get current progressive smoothed results for a track."""
        if track_id in self.progressive_results:
            return self.progressive_results[track_id]
        return None, None
    
    def finalize_all_tracks(self) -> Dict[int, Tuple[float, str]]:
        """Compute final aggregated predictions for all tracks."""
        for track_id in self.all_age_predictions.keys():
            # Aggregate age using mean
            final_age = float(np.mean(self.all_age_predictions[track_id]))
            
            # Aggregate gender using majority vote (based on average probability)
            avg_gender_prob = float(np.mean(self.all_gender_predictions[track_id]))
            final_gender = "male" if avg_gender_prob > 0.5 else "female"
            
            self.track_final_results[track_id] = (final_age, final_gender)
        
        return self.track_final_results
    
    def get_track_stats(self, track_id: int) -> Dict:
        """Get statistics for a specific track."""
        if track_id not in self.all_age_predictions:
            return {}
        
        ages = self.all_age_predictions[track_id]
        genders = self.all_gender_predictions[track_id]
        
        return {
            "track_id": track_id,
            "num_detections": len(ages),
            "age_mean": float(np.mean(ages)),
            "age_std": float(np.std(ages)),
            "age_min": float(np.min(ages)),
            "age_max": float(np.max(ages)),
            "gender_confidence": float(np.mean(genders)),
            "final_age": self.track_final_results.get(track_id, (None, None))[0],
            "final_gender": self.track_final_results.get(track_id, (None, None))[1],
        }
    
    def save_results(self, output_path: str):
        """Save final results to a JSON file."""
        import json
        
        results = {
            "final_predictions": {
                str(track_id): {
                    "age": age,
                    "gender": gender,
                    "num_detections": len(self.all_age_predictions[track_id])
                }
                for track_id, (age, gender) in self.track_final_results.items()
            },
            "track_statistics": {
                str(track_id): self.get_track_stats(track_id)
                for track_id in self.all_age_predictions.keys()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Global temporal smoothing results saved to: {output_path}")


class ConfidenceBasedSelector:
    """Keeps the best prediction based on confidence scores for tracked objects."""
    
    def __init__(self, min_detection_conf: float = 0.5, min_gender_conf: float = 0.6):
        # Store best predictions per track ID
        self.best_predictions: Dict[int, Dict] = {}  # track_id -> {age, gender, detection_conf, gender_conf, frame_idx}
        self.min_detection_conf = min_detection_conf
        self.min_gender_conf = min_gender_conf
    
    def update_and_select(self, track_id: int, age: float, gender: str, 
                         detection_conf: float, gender_conf: float, 
                         frame_idx: int) -> Tuple[float, str]:
        """Update with new prediction and return best prediction so far.
        
        Args:
            track_id: Unique track ID
            age: Predicted age
            gender: Predicted gender ("male"/"female")
            detection_conf: YOLO detection confidence (0-1)
            gender_conf: Gender prediction confidence (0-1)
            frame_idx: Current frame index
            
        Returns:
            Tuple of (best_age, best_gender)
        """
        # Calculate composite confidence score
        # We weight detection confidence higher as it indicates object quality
        composite_conf = (detection_conf * 0.7) + (gender_conf * 0.3)
        
        # Only consider predictions above minimum thresholds
        if detection_conf < self.min_detection_conf or gender_conf < self.min_gender_conf:
            # Return existing best if available, otherwise current
            if track_id in self.best_predictions:
                best = self.best_predictions[track_id]
                return best['age'], best['gender']
            else:
                # Store this as first prediction even if low confidence
                self.best_predictions[track_id] = {
                    'age': age, 'gender': gender, 'detection_conf': detection_conf,
                    'gender_conf': gender_conf, 'composite_conf': composite_conf,
                    'frame_idx': frame_idx
                }
                return age, gender
        
        # Check if this is better than existing best
        if track_id not in self.best_predictions:
            # First good prediction for this track
            self.best_predictions[track_id] = {
                'age': age, 'gender': gender, 'detection_conf': detection_conf,
                'gender_conf': gender_conf, 'composite_conf': composite_conf,
                'frame_idx': frame_idx
            }
            return age, gender
        else:
            # Compare with existing best
            current_best = self.best_predictions[track_id]
            if composite_conf > current_best['composite_conf']:
                # New best prediction
                print(f"[DEBUG] Track {track_id}: New best prediction at frame {frame_idx} "
                      f"(conf: {composite_conf:.3f} > {current_best['composite_conf']:.3f}) "
                      f"Age: {current_best['age']:.1f} -> {age:.1f}, "
                      f"Gender: {current_best['gender']} -> {gender}")
                
                self.best_predictions[track_id] = {
                    'age': age, 'gender': gender, 'detection_conf': detection_conf,
                    'gender_conf': gender_conf, 'composite_conf': composite_conf,
                    'frame_idx': frame_idx
                }
                return age, gender
            else:
                # Keep existing best
                return current_best['age'], current_best['gender']
    
    def get_best_prediction(self, track_id: int) -> Tuple[Optional[float], Optional[str]]:
        """Get the current best prediction for a track."""
        if track_id in self.best_predictions:
            best = self.best_predictions[track_id]
            return best['age'], best['gender']
        return None, None
    
    def clear_old_tracks(self, active_track_ids: set):
        """Remove best predictions for tracks that are no longer active."""
        old_track_ids = set(self.best_predictions.keys()) - active_track_ids
        for track_id in old_track_ids:
            del self.best_predictions[track_id]
    
    def get_statistics(self) -> Dict:
        """Get statistics about the confidence-based selections."""
        if not self.best_predictions:
            return {}
        
        detection_confs = [pred['detection_conf'] for pred in self.best_predictions.values()]
        gender_confs = [pred['gender_conf'] for pred in self.best_predictions.values()]
        composite_confs = [pred['composite_conf'] for pred in self.best_predictions.values()]
        
        return {
            'num_tracks': len(self.best_predictions),
            'avg_detection_conf': np.mean(detection_confs),
            'avg_gender_conf': np.mean(gender_confs),
            'avg_composite_conf': np.mean(composite_confs),
            'min_composite_conf': np.min(composite_confs),
            'max_composite_conf': np.max(composite_confs),
        }


class JSONExporter:
    """Exports tracking results to JSON format with frame-by-frame demographics."""
    
    def __init__(self, fps: float = 30.0, save_crops: bool = True, crops_dir: str = "crops"):
        self.fps = fps
        self.save_crops = save_crops
        self.crops_dir = crops_dir
        self.frame_results = []
        self.frame_count = 0
        
        # Create crops directory if saving crops
        if self.save_crops:
            os.makedirs(self.crops_dir, exist_ok=True)
    
    def add_frame_result(self, detected_objects: PersonAndFaceResult, frame: np.ndarray = None):
        """Add detection results for a single frame."""
        self.frame_count += 1
        current_time = self.frame_count / self.fps
        
        # Get tracking results
        current_frame_objs = detected_objects.get_results_for_tracking()
        cur_persons: Dict[int, AGE_GENDER_TYPE] = current_frame_objs[0]
        cur_faces: Dict[int, AGE_GENDER_TYPE] = current_frame_objs[1]
        
        # Combine persons and faces (prioritize person data if both exist for same ID)
        all_detections = {}
        for track_id, (age, gender) in cur_faces.items():
            if age is not None and gender is not None:
                all_detections[track_id] = (age, gender, "face")
        
        for track_id, (age, gender) in cur_persons.items():
            if age is not None and gender is not None:
                all_detections[track_id] = (age, gender, "person")
        
        # Initialize counters
        demographics = {
            "total_people": 0,
            "man": 0,
            "woman": 0,
            "age_teen": 0,      # 13-19
            "age_twenty": 0,    # 20-29
            "age_thirty": 0,    # 30-39
            "age_fourty": 0,    # 40-49
            "age_fifty": 0,     # 50-59
            "age_sisxty": 0,    # 60-69
            "age_seventy": 0    # 70+
        }
        
        people_data = []
        
        for track_id, (age, gender, detection_type) in all_detections.items():
            demographics["total_people"] += 1
            
            # Gender counting
            gender_str = gender.lower() if isinstance(gender, str) else ("man" if gender > 0.5 else "woman")
            if "male" in gender_str or gender_str == "man":
                demographics["man"] += 1
                gender_display = "man"
            else:
                demographics["woman"] += 1
                gender_display = "woman"
            
            # Age categorization
            age_val = float(age)
            if 13 <= age_val < 20:
                demographics["age_teen"] += 1
            elif 20 <= age_val < 30:
                demographics["age_twenty"] += 1
            elif 30 <= age_val < 40:
                demographics["age_thirty"] += 1
            elif 40 <= age_val < 50:
                demographics["age_fourty"] += 1
            elif 50 <= age_val < 60:
                demographics["age_fifty"] += 1
            elif 60 <= age_val < 70:
                demographics["age_sisxty"] += 1
            elif age_val >= 70:
                demographics["age_seventy"] += 1
            
            # Get bounding box location
            location = self._get_bbox_location(detected_objects, track_id, detection_type)
            
            # Save crop if enabled
            crop_path = None
            if self.save_crops and frame is not None:
                crop_path = self._save_crop(detected_objects, frame, track_id, detection_type)
            
            person_data = {
                "id": int(track_id),
                "gender": gender_display,
                "age": round(age_val),
                "cropped_img": crop_path if crop_path else f"/crops/{track_id}.jpg",
                "location": location
            }
            people_data.append(person_data)
        
        # Create frame result
        frame_result = {
            "time": f"{current_time:.1f} sec",
            **demographics,
            "people_data": people_data
        }
        
        self.frame_results.append(frame_result)
    
    def _get_bbox_location(self, detected_objects: PersonAndFaceResult, track_id: int, detection_type: str) -> List[int]:
        """Get the center coordinates of the bounding box for a track ID."""
        # Get appropriate bounding box indices
        if detection_type == "person":
            bbox_inds = detected_objects.get_bboxes_inds("person")
        else:
            bbox_inds = detected_objects.get_bboxes_inds("face")
        
        # Find the bbox for this track ID
        for bbox_ind in bbox_inds:
            if detected_objects._get_id_by_ind(bbox_ind) == track_id:
                bbox = detected_objects.get_bbox_by_ind(bbox_ind)
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    # Return center coordinates
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    return [center_x, center_y]
        
        # Fallback if bbox not found
        return [0, 0]
    
    def _save_crop(self, detected_objects: PersonAndFaceResult, frame: np.ndarray, track_id: int, detection_type: str) -> str:
        """Save cropped image for a detection and return the relative path."""
        # Get appropriate bounding box indices
        if detection_type == "person":
            bbox_inds = detected_objects.get_bboxes_inds("person")
        else:
            bbox_inds = detected_objects.get_bboxes_inds("face")
        
        # Find the bbox for this track ID
        for bbox_ind in bbox_inds:
            if detected_objects._get_id_by_ind(bbox_ind) == track_id:
                bbox = detected_objects.get_bbox_by_ind(bbox_ind)
                if bbox is not None:
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Add some padding
                    padding = 10
                    h, w = frame.shape[:2]
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(w, x2 + padding)
                    y2 = min(h, y2 + padding)
                    
                    # Crop and save
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        crop_filename = f"{track_id}_{self.frame_count}.jpg"
                        crop_path = os.path.join(self.crops_dir, crop_filename)
                        cv2.imwrite(crop_path, crop)
                        return f"/crops/{crop_filename}"
        
        return f"/crops/{track_id}.jpg"
    
    def save_to_file(self, output_path: str):
        """Save all frame results to a JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.frame_results, f, indent=2)
        
        print(f"✅ JSON results saved to: {output_path}")
        print(f"📊 Total frames processed: {len(self.frame_results)}")
        if self.save_crops:
            print(f"🖼️ Cropped images saved to: {self.crops_dir}/")


class Predictor:
    def __init__(self, config, verbose: bool = False):
        tracker_config = getattr(config, 'tracker', 'botsort.yaml')
        self.detector = Detector(config.detector_weights, config.device, verbose=verbose, tracker=tracker_config)
        self.age_gender_model = MiVOLO(
            config.checkpoint,
            config.device,
            half=True,
            use_persons=config.with_persons,
            disable_faces=config.disable_faces,
            verbose=verbose,
        )
        self.draw = config.draw
        
        # Temporal smoothing and confidence-based selection
        self.temporal_smoothing = getattr(config, 'temporal_smoothing', False)
        self.smoothing_window = getattr(config, 'smoothing_window', 10)
        self.global_smoothing = getattr(config, 'global_smoothing', False)
        self.confidence_based = getattr(config, 'confidence_based', False)
        self.min_detection_conf = getattr(config, 'min_detection_conf', 0.5)
        self.min_gender_conf = getattr(config, 'min_gender_conf', 0.6)
        
        # JSON export functionality
        self.save_json = getattr(config, 'save_json', False)
        self.json_output_path = getattr(config, 'json_output_path', None)
        self.save_crops = getattr(config, 'save_crops', True)
        self.crops_dir = getattr(config, 'crops_dir', 'crops')
        
        # Initialize smoother/selector based on configuration priority:
        # 1. Confidence-based selection (highest priority)
        # 2. Global temporal smoothing 
        # 3. Frame-by-frame temporal smoothing
        if self.confidence_based:
            self.smoother = ConfidenceBasedSelector(self.min_detection_conf, self.min_gender_conf)
            if self.global_smoothing:
                print("[WARNING] Both --confidence-based and --global-smoothing specified. Using confidence-based selection (global smoothing disabled).")
        elif self.temporal_smoothing:
            if self.global_smoothing:
                self.smoother = GlobalTemporalSmoother()
            else:
                self.smoother = TemporalSmoother(self.smoothing_window)
        else:
            self.smoother = None
        
        # Initialize JSON exporter if needed
        self.json_exporter = None

    def recognize(self, image: np.ndarray) -> Tuple[PersonAndFaceResult, Optional[np.ndarray]]:
        detected_objects: PersonAndFaceResult = self.detector.predict(image)
        self.age_gender_model.predict(image, detected_objects)

        out_im = None
        if self.draw:
            # plot results on image
            out_im = detected_objects.plot(line_width=2, font_size=7, font="Arial.ttf")

        return detected_objects, out_im

    def recognize_video(self, source: str, skip_frames: int = 0) -> Generator:
        """Recognise age-gender on a video/stream.

        Parameters
        ----------
        source : str
            Path or URL to the video.
        skip_frames : int, default 0
            Process only every ``skip_frames + 1``-th frame. 0 means process every frame.
        """

        video_capture = cv2.VideoCapture(source)
        if not video_capture.isOpened():
            raise ValueError(f"Failed to open video source {source}")

        detected_objects_history: Dict[int, List[AGE_GENDER_TYPE]] = defaultdict(list)

        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) or int(1e9)
        fps = video_capture.get(cv2.CAP_PROP_FPS) or 30.0

        # Initialize JSON exporter if needed
        if self.save_json:
            # Create crops directory path relative to output
            if self.json_output_path:
                output_dir = os.path.dirname(self.json_output_path)
                crops_dir = os.path.join(output_dir, self.crops_dir)
            else:
                crops_dir = self.crops_dir
            
            self.json_exporter = JSONExporter(
                fps=fps, 
                save_crops=self.save_crops, 
                crops_dir=crops_dir
            )

        frame_idx = 0
        processed_frames = []  # Store frames for global smoothing
        
        for _ in tqdm.tqdm(range(total_frames)):
            # grab next frame (decode only when needed)
            if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                # Faster to grab (skip decoding) than read
                if not video_capture.grab():
                    break
                frame_idx += 1
                continue

            ret, frame = video_capture.read()
            if not ret:
                break

            detected_objects: PersonAndFaceResult = self.detector.track(frame)
            self.age_gender_model.predict(frame, detected_objects)

            # Handle different smoothing/selection modes
            if self.confidence_based and self.smoother:
                # Apply confidence-based selection
                self._apply_confidence_based_selection(detected_objects, frame_idx)
            elif self.temporal_smoothing and self.smoother:
                if self.global_smoothing:
                    # For global smoothing, collect predictions and apply progressive smoothing
                    self._collect_global_predictions(detected_objects)
                    self._apply_progressive_global_smoothing(detected_objects)
                else:
                    # For frame-by-frame smoothing, apply immediately
                    self._apply_temporal_smoothing(detected_objects)

            # Get current frame results (potentially smoothed)
            current_frame_objs = detected_objects.get_results_for_tracking()
            cur_persons: Dict[int, AGE_GENDER_TYPE] = current_frame_objs[0]
            cur_faces: Dict[int, AGE_GENDER_TYPE] = current_frame_objs[1]

            # Add current frame results to history (will be smoothed if temporal smoothing is enabled)
            for guid, data in cur_persons.items():
                if None not in data:
                    detected_objects_history[guid].append(data)
            for guid, data in cur_faces.items():
                if None not in data:
                    detected_objects_history[guid].append(data)

            # Clean up old tracks from smoother/selector
            if self.smoother and (self.confidence_based or (self.temporal_smoothing and not self.global_smoothing)):
                active_ids = set(cur_persons.keys()) | set(cur_faces.keys())
                self.smoother.clear_old_tracks(active_ids)

            # Store frame data for global smoothing
            if self.global_smoothing:
                processed_frames.append((detected_objects, frame))
            
            # Only apply tracking-based aggregation if no custom smoothing/selection is enabled
            # (custom methods handle their own aggregation)
            if not self.temporal_smoothing and not self.confidence_based:
                detected_objects.set_tracked_age_gender(detected_objects_history)
            
            # Export to JSON if enabled
            if self.json_exporter:
                self.json_exporter.add_frame_result(detected_objects, frame)

            if self.draw:
                frame = detected_objects.plot(line_width=2, font_size=7, font="Arial.ttf")

            frame_idx += 1
            yield detected_objects_history, frame

        # Post-process for global smoothing
        if self.global_smoothing and self.smoother:
            self._finalize_global_smoothing(processed_frames)
        
        # Save confidence-based results
        if self.confidence_based and self.smoother:
            self._save_confidence_results()
        
        # Save JSON results if enabled
        if self.json_exporter and self.json_output_path:
            self.json_exporter.save_to_file(self.json_output_path)

    def _apply_temporal_smoothing(self, detected_objects: PersonAndFaceResult):
        """Apply temporal smoothing to the detected objects predictions."""
        if not self.smoother:
            return
            
        # Get current predictions for smoothing
        current_frame_objs = detected_objects.get_results_for_tracking()
        cur_persons: Dict[int, AGE_GENDER_TYPE] = current_frame_objs[0]
        cur_faces: Dict[int, AGE_GENDER_TYPE] = current_frame_objs[1]
        
        print(f"[DEBUG] Applying temporal smoothing to {len(cur_persons)} persons and {len(cur_faces)} faces")
        
        # Apply smoothing to persons
        for track_id, (age, gender) in cur_persons.items():
            if age is not None and gender is not None:
                # Convert gender to probability (assume gender is float 0-1 or string)
                if isinstance(gender, str):
                    gender_prob = 1.0 if gender.lower() in ['male', 'm'] else 0.0
                else:
                    gender_prob = float(gender)
                
                smoothed_age, smoothed_gender = self.smoother.update_and_smooth(
                    track_id, float(age), gender_prob
                )
                
                print(f"[DEBUG] Person ID {track_id}: {age:.1f} -> {smoothed_age:.1f}, {gender} -> {'male' if smoothed_gender > 0.5 else 'female'}")
                
                # Find the person index and update with smoothed values
                person_inds = detected_objects.get_bboxes_inds("person")
                for person_ind in person_inds:
                    if detected_objects._get_id_by_ind(person_ind) == track_id:
                        detected_objects.set_age(person_ind, smoothed_age)
                        # Convert probability back to gender string
                        smoothed_gender_str = "male" if smoothed_gender > 0.5 else "female"
                        detected_objects.set_gender(person_ind, smoothed_gender_str, smoothed_gender)
                        break
                
        # Apply smoothing to faces
        for track_id, (age, gender) in cur_faces.items():
            if age is not None and gender is not None:
                # Convert gender to probability
                if isinstance(gender, str):
                    gender_prob = 1.0 if gender.lower() in ['male', 'm'] else 0.0
                else:
                    gender_prob = float(gender)
                
                smoothed_age, smoothed_gender = self.smoother.update_and_smooth(
                    track_id + 10000,  # Offset face IDs to avoid collision with person IDs
                    float(age), gender_prob
                )
                
                print(f"[DEBUG] Face ID {track_id}: {age:.1f} -> {smoothed_age:.1f}, {gender} -> {'male' if smoothed_gender > 0.5 else 'female'}")
                
                # Find the face index and update with smoothed values
                face_inds = detected_objects.get_bboxes_inds("face")
                for face_ind in face_inds:
                    if detected_objects._get_id_by_ind(face_ind) == track_id:
                        detected_objects.set_age(face_ind, smoothed_age)
                        # Convert probability back to gender string
                        smoothed_gender_str = "male" if smoothed_gender > 0.5 else "female"
                        detected_objects.set_gender(face_ind, smoothed_gender_str, smoothed_gender)
                        break

    def _collect_global_predictions(self, detected_objects: PersonAndFaceResult):
        """Collect predictions for global smoothing."""
        if not self.smoother:
            return
            
        # Get current predictions for smoothing
        current_frame_objs = detected_objects.get_results_for_tracking()
        cur_persons: Dict[int, AGE_GENDER_TYPE] = current_frame_objs[0]
        cur_faces: Dict[int, AGE_GENDER_TYPE] = current_frame_objs[1]
        
        # Collect predictions
        for track_id, (age, gender) in cur_persons.items():
            if age is not None and gender is not None:
                # Convert gender to probability
                if isinstance(gender, str):
                    gender_prob = 1.0 if gender.lower() in ['male', 'm'] else 0.0
                else:
                    gender_prob = float(gender)
                self.smoother.add_predictions(track_id, float(age), gender_prob)
                
        for track_id, (age, gender) in cur_faces.items():
            if age is not None and gender is not None:
                # Convert gender to probability
                if isinstance(gender, str):
                    gender_prob = 1.0 if gender.lower() in ['male', 'm'] else 0.0
                else:
                    gender_prob = float(gender)
                self.smoother.add_predictions(track_id + 10000, float(age), gender_prob)

    def _apply_progressive_global_smoothing(self, detected_objects: PersonAndFaceResult):
        """Apply progressive global smoothing to the detected objects using running averages."""
        if not self.smoother or not hasattr(self.smoother, 'get_progressive_results'):
            return
            
        # Get current predictions
        current_frame_objs = detected_objects.get_results_for_tracking()
        cur_persons: Dict[int, AGE_GENDER_TYPE] = current_frame_objs[0]
        cur_faces: Dict[int, AGE_GENDER_TYPE] = current_frame_objs[1]
        
        print(f"[DEBUG] Applying progressive global smoothing to {len(cur_persons)} persons and {len(cur_faces)} faces")
        
        # Apply progressive smoothing to persons
        for track_id, (age, gender) in cur_persons.items():
            if age is not None and gender is not None:
                progressive_age, progressive_gender = self.smoother.get_progressive_results(track_id)
                
                if progressive_age is not None and progressive_gender is not None:
                    print(f"[DEBUG] Person ID {track_id}: {age:.1f} -> {progressive_age:.1f}, {gender} -> {progressive_gender}")
                    
                    # Find the person index and update with progressive values
                    person_inds = detected_objects.get_bboxes_inds("person")
                    for person_ind in person_inds:
                        if detected_objects._get_id_by_ind(person_ind) == track_id:
                            detected_objects.set_age(person_ind, progressive_age)
                            detected_objects.set_gender(person_ind, progressive_gender, 0.9)
                            break
        
        # Apply progressive smoothing to faces
        for track_id, (age, gender) in cur_faces.items():
            if age is not None and gender is not None:
                progressive_age, progressive_gender = self.smoother.get_progressive_results(track_id + 10000)
                
                if progressive_age is not None and progressive_gender is not None:
                    print(f"[DEBUG] Face ID {track_id}: {age:.1f} -> {progressive_age:.1f}, {gender} -> {progressive_gender}")
                    
                    # Find the face index and update with progressive values
                    face_inds = detected_objects.get_bboxes_inds("face")
                    for face_ind in face_inds:
                        if detected_objects._get_id_by_ind(face_ind) == track_id:
                            detected_objects.set_age(face_ind, progressive_age)
                            detected_objects.set_gender(face_ind, progressive_gender, 0.9)
                            break

    def _finalize_global_smoothing(self, processed_frames: List[Tuple[PersonAndFaceResult, np.ndarray]]):
        """Finalize global smoothing and save results."""
        if not self.smoother:
            return
        
        # Only finalize if we have a GlobalTemporalSmoother (not ConfidenceBasedSelector)
        if not hasattr(self.smoother, 'finalize_all_tracks'):
            print("[DEBUG] Skipping global smoothing finalization - confidence-based selection is active")
            return
            
        # Finalize global smoothing
        self.smoother.finalize_all_tracks()
        
        # Save results
        self.smoother.save_results("global_smoothing_results.json")

    def _apply_confidence_based_selection(self, detected_objects: PersonAndFaceResult, frame_idx: int):
        """Apply confidence-based selection to detected objects."""
        if not self.smoother:
            return
            
        # Get current predictions with confidence scores
        current_frame_objs = detected_objects.get_results_for_tracking()
        cur_persons: Dict[int, AGE_GENDER_TYPE] = current_frame_objs[0]
        cur_faces: Dict[int, AGE_GENDER_TYPE] = current_frame_objs[1]
        
        # Get YOLO detection boxes for confidence scores
        yolo_boxes = detected_objects.yolo_results.boxes
        
        print(f"[DEBUG] Applying confidence-based selection to {len(cur_persons)} persons and {len(cur_faces)} faces")
        
        # Apply confidence-based selection to persons
        for track_id, (age, gender) in cur_persons.items():
            if age is not None and gender is not None:
                # Find detection and gender confidence for this person
                detection_conf, gender_conf = self._get_confidence_scores(detected_objects, track_id, "person")
                
                if detection_conf is not None and gender_conf is not None:
                    best_age, best_gender = self.smoother.update_and_select(
                        track_id, float(age), gender, detection_conf, gender_conf, frame_idx
                    )
                    
                    # Update with best prediction
                    person_inds = detected_objects.get_bboxes_inds("person")
                    for person_ind in person_inds:
                        if detected_objects._get_id_by_ind(person_ind) == track_id:
                            detected_objects.set_age(person_ind, best_age)
                            detected_objects.set_gender(person_ind, best_gender, gender_conf)
                            break
        
        # Apply confidence-based selection to faces
        for track_id, (age, gender) in cur_faces.items():
            if age is not None and gender is not None:
                # Find detection and gender confidence for this face
                detection_conf, gender_conf = self._get_confidence_scores(detected_objects, track_id, "face")
                
                if detection_conf is not None and gender_conf is not None:
                    best_age, best_gender = self.smoother.update_and_select(
                        track_id + 10000,  # Offset face IDs to avoid collision
                        float(age), gender, detection_conf, gender_conf, frame_idx
                    )
                    
                    # Update with best prediction  
                    face_inds = detected_objects.get_bboxes_inds("face")
                    for face_ind in face_inds:
                        if detected_objects._get_id_by_ind(face_ind) == track_id:
                            detected_objects.set_age(face_ind, best_age)
                            detected_objects.set_gender(face_ind, best_gender, gender_conf)
                            break

    def _get_confidence_scores(self, detected_objects: PersonAndFaceResult, track_id: int, category: str) -> Tuple[Optional[float], Optional[float]]:
        """Get detection and gender confidence scores for a specific track ID and category."""
        # Find the index of this track ID in the specified category
        category_inds = detected_objects.get_bboxes_inds(category)
        
        for ind in category_inds:
            if detected_objects._get_id_by_ind(ind) == track_id:
                # Get detection confidence from YOLO
                detection_conf = float(detected_objects.yolo_results.boxes[ind].conf.item())
                
                # Get gender confidence score
                gender_conf = detected_objects.gender_scores[ind]
                if gender_conf is not None:
                    gender_conf = float(gender_conf)
                else:
                    gender_conf = 0.5  # Default if not available
                
                return detection_conf, gender_conf
        
        return None, None
    
    def _save_confidence_results(self):
        """Save confidence-based selection results."""
        if not self.smoother or not hasattr(self.smoother, 'get_statistics'):
            return
        
        import json
        
        # Get statistics
        stats = self.smoother.get_statistics()
        
        # Get all best predictions
        results = {
            "confidence_selection_results": {
                str(track_id): {
                    "age": pred_data['age'],
                    "gender": pred_data['gender'],
                    "detection_confidence": pred_data['detection_conf'],
                    "gender_confidence": pred_data['gender_conf'],
                    "composite_confidence": pred_data['composite_conf'],
                    "best_frame": pred_data['frame_idx']
                }
                for track_id, pred_data in self.smoother.best_predictions.items()
            },
            "statistics": stats
        }
        
        output_path = "confidence_based_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Confidence-based selection results saved to: {output_path}")
        print(f"Statistics: {stats}")
