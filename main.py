import cv2
import numpy as np
import time
import json
import pafy
from art import tprint
import yt_dlp
import math

def apply_yolo_object_detection(image_to_process):

    height, width, _ = image_to_process.shape
    blob = cv2.dnn.blobFromImage(image_to_process, 1 / 255, (608, 608),
                                 (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)
    class_indexes, class_scores, boxes = ([] for i in range(3))
    objects_count = 0

    # Search
    for out in outs:
        for obj in out:
            scores = obj[5:]
            class_index = np.argmax(scores)
            class_score = scores[class_index]
            if class_score > 0:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                obj_width = int(obj[2] * width)
                obj_height = int(obj[3] * height)
                box = [center_x - obj_width // 2, center_y - obj_height // 2,
                       obj_width, obj_height]
                boxes.append(box)
                class_indexes.append(class_index)
                class_scores.append(float(class_score))

    # Select
    chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)
    for box_index in chosen_boxes:
        box_index = box_index
        box = boxes[box_index]
        class_index = class_indexes[box_index]

        # debugging
        if classes[class_index] in classes_to_look_for:
            objects_count += 1
            image_to_process = draw_object_bounding_box(image_to_process,
                                                        class_index, box)

    final_image = draw_object_count(image_to_process, objects_count)
    return final_image

def draw_object_bounding_box(image_to_process, index, box):

    x, y, w, h = box
    start = (x, y)
    end = (x + w, y + h)
    color = (0, 255, 0)
    width = 2
    final_image = cv2.rectangle(image_to_process, start, end, color, width)

    start = (x, y - 10)
    font_size = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 2
    text = classes[index]
    final_image = cv2.putText(final_image, text, start, font,
                              font_size, color, width, cv2.LINE_AA)

    return final_image

def draw_object_count(image_to_process, objects_count):

    start = (10, 120)
    font_size = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 3
    text = "Detected objects: " + str(objects_count)

    # Text output
    white_color = (255, 255, 255)
    black_outline_color = (0, 0, 0)
    final_image = cv2.putText(image_to_process, text, start, font, font_size,
                              black_outline_color, width * 3, cv2.LINE_AA)
    final_image = cv2.putText(final_image, text, start, font, font_size,
                              white_color, width, cv2.LINE_AA)

    return final_image

def start_video_object_detection(video: str):

    while True:
        try:
            # Capturing a picture
            video_camera_capture = cv2.VideoCapture(video)

            while video_camera_capture.isOpened():
                ret, frame = video_camera_capture.read()
                if not ret:
                    break

                frame = apply_yolo_object_detection(frame)

                frame = cv2.resize(frame, (1920 // 2, 1080 // 2))
                cv2.imshow("Video Capture", frame)
                cv2.waitKey(10)

            video_camera_capture.release()
            cv2.destroyAllWindows()

        except KeyboardInterrupt:
            pass

if __name__ == '__main__':

    tprint("Boat detection")

    net = cv2.dnn.readNetFromDarknet("ML/train_config.cfg",
                                     "ML/boat_model.weights")
    layer_names = net.getLayerNames()
    out_layers_indexes = net.getUnconnectedOutLayers()
    out_layers = [layer_names[index - 1] for index in out_layers_indexes]

    with open("ML/titles.txt") as file:
        classes = file.read().split("\n")

    #url = 'https://www.youtube.com/watch?v=5VoO4DGb3K4'
    #v#ideoPafy = pafy.new(url)
    #best = videoPafy.getbest(preftype="webm")
    #videoy = cv2.VideoCapture(best.url)
    #video = "Result/input/test2.mp4"
    #look_for = "boat"

    def get_stream_url(youtube_url):
      ydl_opts = {
        'format': 'best',
        'quiet': True,
        'noplaylist': True,
        'extract_flat': False,
      }

      with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        video_url = info_dict['url']
        return video_url

    #youtube_url = ''
    #youtube_url = 'https://www.youtube.com/watch?v=UsRzYJZszV4'
    #youtube_url = 'https://www.youtube.com/watch?v=L_beX9ZAhLQ'
    youtube_url = 'https://www.youtube.com/watch?v=5VoO4DGb3K4'
    stream_url = get_stream_url(youtube_url)
    print(f"Прямая ссылка на поток: {stream_url}")

    #video = cv2.VideoCapture(stream_url)

    #video = "test_video/test4.mp4"
    video = stream_url
    look_for = input("detecting: ").split(',')

    list_look_for = []
    for look in look_for:
        list_look_for.append(look.strip())

    classes_to_look_for = list_look_for
    start_video_object_detection(video)

tracker_types = ['MIL', 'CSRT']

def create_tracker_by_name(tracker_type):
    if tracker_type == tracker_types[0]:
        tracker = cv2.legacy.TrackerBoosting_create()
    elif tracker_type == tracker_types[1]:
        tracker = cv2.legacy.TrackerMIL_create()
    elif tracker_type == tracker_types[2]:
        tracker = cv2.legacy.TrackerKCF_create()
    elif tracker_type == tracker_types[3]:
        tracker = cv2.legacy.TrackerTLD_create()
    elif tracker_type == tracker_types[4]:
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif tracker_type == tracker_types[5]:
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == tracker_types[6]:
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        tracker = None
        print('Invalid name! Available trackers: ')
        for t in tracker_types:
            print(t)

    return tracker

video = cv2.VideoCapture('Videos/test2.MP4')
if not video.isOpened():
    print('Error while loading the video!')
    sys.exit()
ok, frame = video.read()

bboxes = []
colors = []

while True:
    bbox = cv2.selectROI('MIL', frame)
    bboxes.append(bbox)
    colors.append((randint(0,255), randint(0,255), randint(0,255)))
    print('Press Q to quit and start tracking')
    print('Press any other key to select the next object')
    k = cv2.waitKey(0) & 0XFF
    if k == 113: # Q - quit
        break

print(bboxes)
print(colors)

tracker_type = 'CSRT'
multi_tracker = cv2.legacy.MultiTracker_create()
for bbox in bboxes:
    multi_tracker.add(create_tracker_by_name(tracker_type), frame, bbox)

while video.isOpened():
    ok, frame = video.read()
    if not ok:
        break

    ok, boxes = multi_tracker.update(frame)

    for i, new_box in enumerate(boxes):
        (x, y, w, h) = [int(v) for v in new_box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors[i], 2)

    cv2.imshow('MultiTracker', frame)
    if cv2.waitKey(1) & 0XFF == 47: # esc
        break


class Vehicle(object):
  def __init__(self, carid, position, start_frame):
    self.id = carid

    self.positions = [position]
    self.frames_since_seen = 0
    self.counted = False
    self.start_frame = start_frame

    self.speed = None

    self.color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))

  @property
  def last_position(self):
    return self.positions[-1]

  def add_position(self, new_position):
    self.positions.append(new_position)
    self.frames_since_seen = 0

  def draw(self, output_image):

    for point in self.positions:
      cv2.circle(output_image, point, 2, self.color, -1)
      cv2.polylines(output_image, [np.int32(self.positions)], False, self.color, 1)

    if self.speed:
      cv2.putText(output_image, ("%1.2f" % self.speed), self.last_position, cv2.FONT_HERSHEY_PLAIN, 0.7,
                  (127, 255, 255), 1)


class VehicleCounter(object):
  def __init__(self, shape, road, fps, samples=0):
    self.height, self.width = shape
    self.divider = road['divider']
    self.is_horizontal = road['divider_horizontal']
    self.pass_side = road['divider_pass_side']
    self.vector_angle_min = road['vector_angle_min']
    self.vector_angle_max = road['vector_angle_max']

    self.vehicles = []
    self.next_vehicle_id = 0
    self.vehicle_count = 0

    self.max_unseen_frames = 10

    self.sample_num = samples

    if samples == 0:
      print('DISTANCE MODE')
      self.distance = road['distance']
      self.fps = fps
    else:
      print('AVERAGE MODE')
      self.samples = []
      self.average_speed = -1
      self.average_threshold = 0.3

      self.average_distance = -1
      self.distances = []

  @staticmethod
  def get_vector(a, b):
    dx = float(b[0] - a[0])
    dy = float(b[1] - a[1])

    distance = math.sqrt(dx ** 2 + dy ** 2)

    if dy > 0:
      angle = math.degrees(math.atan(-dx / dy))
    elif dy == 0:
      if dx < 0:
        angle = 90.0
      elif dx > 0:
        angle = -90.0
      else:
        angle = 0.0
    else:
      if dx < 0:
        angle = 180 - math.degrees(math.atan(dx / dy))
      elif dx > 0:
        angle = -180 - math.degrees(math.atan(dx / dy))
      else:
        angle = 180.0

    return distance, angle

  @staticmethod
  def is_valid_vector(a, angle_min, angle_max):
    distance, angle = a
    return (distance <= 60 and angle > angle_min and angle < angle_max)

  def is_past_divider(self, centroid):
    x, y = centroid

    if self.is_horizontal:
      if self.pass_side == -1:
        return y < self.divider
      else:
        return y > self.divider

    else:
      if self.pass_side == -1:
        return x < self.divider
      else:
        return x > self.divider

  def update_vehicle(self, vehicle, matches):
    # Find if any of the matches fits this vehicle
    for i, match in enumerate(matches):
      contour, centroid = match

      vector = self.get_vector(vehicle.last_position, centroid)
      if self.is_valid_vector(vector, self.vector_angle_min, self.vector_angle_max):
        print('Angle: %s' % vector[1])
        vehicle.add_position(centroid)

        return i

    # No matches fit
    # print('No matches found for vehicle %s' % vehicle.id)
    vehicle.frames_since_seen += 1

    return None

  def update_count(self, matches, frame_number, output_image=None):

    # Update existing vehicles
    for vehicle in self.vehicles:
      i = self.update_vehicle(vehicle, matches)
      if i is not None:
        del matches[i]

    for match in matches:
      contour, centroid = match

      if not self.is_past_divider(centroid):
        new_vehicle = Vehicle(self.next_vehicle_id, centroid, frame_number)
        self.next_vehicle_id += 1
        self.vehicles.append(new_vehicle)

    for vehicle in self.vehicles:
      if not vehicle.counted and self.is_past_divider(vehicle.last_position):
        if self.sample_num == 0:

          time_alive = (frame_number - vehicle.start_frame) / self.fps

          time_alive = time_alive / 60 / 60

          vehicle.speed = self.distance / time_alive

        # print(self.distance, time_alive)

        else:

          distance = self.get_vector(vehicle.last_position, vehicle.positions[0])[0]

          speed = distance / (frame_number - vehicle.start_frame)
          print(f"SPEED: {speed}")

          if len(self.samples) < self.sample_num:
            # Add to samples

            self.samples.append(speed)
            self.distances.append(distance)

            # Should we take the average now?
            if len(self.samples) == self.sample_num:
              self.average_speed = sum(self.samples) / len(self.samples)
              self.average_distance = sum(self.distances) / len(self.distances)

              print(f"AVERAGE SPEED: {self.average_speed}")

          else:

            speed_diff = (speed - self.average_speed) / self.average_speed
            vehicle.speed = speed_diff

            if speed_diff >= self.average_threshold:
              print(f"{vehicle.id} is SPEEDING: {speed_diff}")

        self.vehicle_count += 1
        vehicle.counted = True

    # Draw the vehicles (optional)
    if output_image is not None:
      for vehicle in self.vehicles:
        vehicle.draw(output_image)

      cv2.putText(output_image, ("%02d" % self.vehicle_count), (0, 0), cv2.FONT_HERSHEY_PLAIN, 0.7, (127, 255, 255), 1)
DIVIDER_COLOR = (255, 255, 0)
BOUNDING_BOX_COLOR = (255, 0, 0)
CENTROID_COLOR = (0, 0, 255)

# For cropped rectangles
ref_points = []
ref_rects = []

def nothing(x):
	pass

def click_and_crop (event, x, y, flags, param):
	global ref_points

	if event == cv2.EVENT_LBUTTONDOWN:
		ref_points = [(x,y)]

	elif event == cv2.EVENT_LBUTTONUP:
		(x1, y1), x2, y2 = ref_points[0], x, y

		ref_points[0] = ( min(x1,x2), min(y1,y2) )

		ref_points.append ( ( max(x1,x2), max(y1,y2) ) )

		ref_rects.append( (ref_points[0], ref_points[1]) )

def save_cropped():
	global ref_rects

	with open('settings.json', 'r+') as f:
		data = json.load(f)
		data[road_name]['cropped_rects'] = ref_rects

		f.seek(0)
		json.dump(data, f, indent=4)
		f.truncate()

	print('Saved ref_rects to settings.json!')

# Load any saved cropped rectangles
def load_cropped ():
	global ref_rects

	ref_rects = road['cropped_rects']

	print('Loaded ref_rects from settings.json!')

# Remove cropped regions from frame
def remove_cropped (gray, color):
	cropped = gray.copy()
	cropped_color = color.copy()

	for rect in ref_rects:
		cropped[ rect[0][1]:rect[1][1], rect[0][0]:rect[1][0] ] = 0
		cropped_color[ rect[0][1]:rect[1][1], rect[0][0]:rect[1][0] ] = (0,0,0)


	return cropped, cropped_color

def filter_mask (mask):

	kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
	kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
	kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

	opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close)

	dilation = cv2.dilate(closing, kernel_dilate, iterations = 2)

	return dilation

def get_centroid (x, y, w, h):
	x1 = w // 2
	y1 = h // 2

	return(x+x1, y+y1)

def detect_vehicles (mask):

	MIN_CONTOUR_WIDTH = 10
	MIN_CONTOUR_HEIGHT = 10

	contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	matches = []

	for (i, contour) in enumerate(contours):
		x, y, w, h = cv2.boundingRect(contour)
		contour_valid = (w >= MIN_CONTOUR_WIDTH) and (h >= MIN_CONTOUR_HEIGHT)

		if not contour_valid or not hierarchy[0,i,3] == -1:
			continue

		centroid = get_centroid(x, y, w, h)

		matches.append( ((x,y,w,h), centroid) )

	return matches

def process_frame(frame_number, frame, bg_subtractor, car_counter):
	processed = frame.copy()

	gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

	# remove specified cropped regions
	cropped, processed = remove_cropped(gray, processed)

	if car_counter.is_horizontal:
		cv2.line(processed, (0, car_counter.divider), (frame.shape[1], car_counter.divider), DIVIDER_COLOR, 1)
	else:
		cv2.line(processed, (car_counter.divider, 0), (car_counter.divider, frame.shape[0]), DIVIDER_COLOR, 1)

	fg_mask = bg_subtractor.apply(cropped)
	fg_mask = filter_mask(fg_mask)

	matches = detect_vehicles(fg_mask)

	for (i, match) in enumerate(matches):
		contour, centroid = match

		x,y,w,h = contour

		cv2.rectangle(processed, (x,y), (x+w-1, y+h-1), BOUNDING_BOX_COLOR, 1)
		cv2.circle(processed, centroid, 2, CENTROID_COLOR, -1)

	car_counter.update_count(matches, frame_number, processed)

	cv2.imshow('Filtered Mask', fg_mask)

	return processed

def lane_detection (frame):
	gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

	cropped = remove_cropped(gray)

def main ():
	bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)
	car_counter = None

	load_cropped()

	cap = cv2.VideoCapture(road['stream_url'])
	cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

	cv2.namedWindow('Source Image')
	cv2.setMouseCallback('Source Image', click_and_crop)

	frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

	frame_number = -1

	while True:
		frame_number += 1
		ret, frame = cap.read()

		if not ret:
			print('Frame capture failed, stopping...')
			break

		if car_counter is None:
			car_counter = VehicleCounter(frame.shape[:2], road, cap.get(cv2.CAP_PROP_FPS), samples=10)

		processed = process_frame(frame_number, frame, bg_subtractor, car_counter)

		cv2.imshow('Source Image', frame)
		cv2.imshow('Processed Image', processed)

		key = cv2.waitKey(WAIT_TIME)

		if key == ord('s'):
			# save rects!
			save_cropped()
		elif key == ord('q') or key == 27:
			break

		# Keep video's speed stable

		time.sleep( 1.0 / cap.get(cv2.CAP_PROP_FPS) )

	print('Closing video capture...')
	cap.release()
	cv2.destroyAllWindows()
	print('Done.')
