import cv2
import math
import dlib
import numpy as np
import mediapipe as mp
from imutils import face_utils

def dlib_detector(image):
  # Copy an original image
  detecting_image = image.copy()
  gray = cv2.cvtColor(detecting_image, cv2.COLOR_BGR2GRAY)

  # Use 81 landmark detection
  p = "shape_predictor_81_face_landmarks.dat"
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(p)
  
  # Find face
  rects = detector(gray, 0)

  # Collect face landmark
  face_landmark = []
  for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    face_landmark.append(shape)
  
  # Mark point and number of face landmark on the image 
  c = 0 
  for (x, y) in shape: 
    c+=1
    font_size = (image.shape[0])*0.0005
    dot_size = (image.shape[0])*0.002
    face_landmark.append([x,y])
    cv2.circle(detecting_image, (x, y), int(dot_size), (0, 255, 0), -1)
    cv2.putText(detecting_image, str(c), (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), int(dot_size))
  return detecting_image, np.array(face_landmark)


def mediapipe_detector(image):
  mp_drawing = mp.solutions.drawing_utils
  mp_face_mesh = mp.solutions.face_mesh

  # Normalize coordinate to image coordinate
  def _normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height):
    def is_valid_normalized_value(value: float) -> bool:
      return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                        math.isclose(1, value))
    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
      return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

  def create_np_landmark(image_landmark):
    for i, points in enumerate(image_landmark[0].landmark):
      if i == 0: 
        np_keypoint = np.array([[points.x, points.y, points.z]])
      else: 
        np_keypoint = np.concatenate((np_keypoint, np.array([[points.x, points.y, points.z]])), axis=0)
    return np_keypoint

  circleDrawingSpec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2, color=(0,255,0))
  lineDrawingSpec = mp_drawing.DrawingSpec(thickness=2, color=(255,255,255))

  image_landmark = []
  with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.6) as face_mesh:
    # Start landmark detection
    image_rows, image_cols, _ = image.shape
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    annotated_image = image.copy()
    for face_landmarks in results.multi_face_landmarks:
      image_landmark.append(face_landmarks)
      landmark_list = face_landmarks
      idx_to_coordinates = {}
      for idx, landmark in enumerate(landmark_list.landmark):
        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)
        if landmark_px: 
          idx_to_coordinates[idx] = landmark_px

      # Draw dot and number on face landmark
      dot_color = (0, 255, 0)
      text_color = (255, 255, 255)
      face_coor_landmark = []
      for c, landmark_px in enumerate(idx_to_coordinates.values()):
        x, y = landmark_px
        font_size = (image.shape[0])*0.0003
        dot_size = (image.shape[0])*0.001
        face_coor_landmark.append([x, y])
        cv2.circle(annotated_image, landmark_px,  int(dot_size), dot_color)
        cv2.putText(annotated_image, str(c), landmark_px, 
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, 
                    text_color, int(dot_size))
        
      # Draw connection
      connections = mp_face_mesh.FACE_CONNECTIONS
      if connections:
        num_landmarks = len(landmark_list.landmark)
      # Draws the connections if the start and end landmarks are both visible.
      for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
          cv2.line(annotated_image, idx_to_coordinates[start_idx], idx_to_coordinates[end_idx], dot_color, int(dot_size))
      
      face_landmark_3d = create_np_landmark(image_landmark)
      face_landmark_2d = np.array(face_coor_landmark)
  return annotated_image, face_landmark_2d, face_landmark_3d

####### Function Template #######
# def my_function():
#   print("Hello World")

# # Defining our variable
# name = "Nicholas"

# # Defining a class
# class Student:
#   def __init__(self, name, course):
#     self.course = course
#     self.name = name

#   def get_student_details(self):
#     print("Your name is " + self.name + ".")
#     print("You are studying " + self.course)
