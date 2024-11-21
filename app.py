import cv2
import numpy as np
import socket
import pickle
from art import tprint

# URL камеры с аутентификацией
username = 'root'
password = 'katran2024'
url = f"http://{username}:{password}@192.168.2.43/mjpeg"

# Настройки сокета для управления камерой
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('192.168.2.37', 2000)

# Порог для погрешности
deviation_threshold = 10

""">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  Boat detection start
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""

def apply_boat_object_detection(image_to_process):
  height, width, _ = image_to_process.shape
  blob = cv2.dnn.blobFromImage(image_to_process, 1 / 255, (608, 608),
                               (0, 0, 0), swapRB=True, crop=False)
  net.setInput(blob)
  outs = net.forward(out_layers)
  class_indexes, class_scores, boxes = ([] for i in range(3))
  objects_count = 0

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

  chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)
  for box_index in chosen_boxes:
    box_index = box_index
    box = boxes[box_index]
    class_index = class_indexes[box_index]

    if classes[class_index] in classes_to_look_for:
      objects_count += 1
      image_to_process = draw_object_box(image_to_process,
                                                  class_index, box)

  final_image = draw_count(image_to_process, objects_count)
  return final_image

def draw_object_box(image_to_process, index, box):

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

def draw_count(image_to_process, objects_count):

  start = (10, 120)
  font_size = 1.5
  font = cv2.FONT_HERSHEY_SIMPLEX
  width = 3
  text = "Detected objects: " + str(objects_count)

  white_color = (255, 255, 255)
  black_outline_color = (0, 0, 0)
  final_image = cv2.putText(image_to_process, text, start, font, font_size,
                            black_outline_color, width * 3, cv2.LINE_AA)
  final_image = cv2.putText(final_image, text, start, font, font_size,
                            white_color, width, cv2.LINE_AA)

  return final_image

def start_video_board_detection(video: str):
  while True:
    try:
      video_camera_capture = cv2.VideoCapture(video)

      while video_camera_capture.isOpened():
        ret, frame = video_camera_capture.read()
        if not ret:
          break

        frame = apply_boat_object_detection(frame)

        frame = cv2.resize(frame, (1920 // 2, 1080 // 2))
        cv2.imshow("Video Capture", frame)
        cv2.waitKey(1)

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

  #video = url
  video = './v-tests/test2.mp4'
  look_for = input("What will detect?: ").split(',')

  list_detect = []
  for look in look_for:
    list_detect.append(look.strip())

  classes_to_look_for = list_detect

  start_video_board_detection(video)

""">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  Boat detection end
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""

# Функция для расчета значения axes[0] на основе deviation_x
def calculate_axes_x(deviation_x):
    if abs(deviation_x) <= deviation_threshold:
        return 0.0  # Остановка, когда объект в центре с допустимой погрешностью
    # Преобразуем deviation_x в диапазон от 1.0 до -1.0
    axes_value = np.clip(-deviation_x / 600, -1.0, 1.0)
    return axes_value


# Функция для расчета значения axes[3] на основе deviation_y
def calculate_axes_y(deviation_y):
    if abs(deviation_y) <= deviation_threshold:
        return 0.0  # Остановка, когда объект в центре с допустимой погрешностью
    # Преобразуем deviation_y в диапазон от 1.0 до -1.0
    axes_value = np.clip(-deviation_y / 600, -1.0, 1.0)
    return axes_value

# Запускаем видеопоток
cap = cv2.VideoCapture(url)

# Проверяем, открылся ли видеопоток
if not cap.isOpened():
    print("Ошибка при открытии видеопотока")
    exit()

# Основной цикл обработки кадров
while True:
    # Читаем кадр
    ret, frame = cap.read()
    if not ret:
        print("Ошибка при чтении кадра")
        break

    # Преобразуем кадр в HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Задаем диапазон цвета для обнаружения белой бутылки
    lower_white = np.array([160, 100, 100])
    upper_white = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Находим контуры
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Если контуры найдены
    if contours:
        # Находим самый большой контур
        largest_contour = max(contours, key=cv2.contourArea)
        # Получаем координаты ограничивающего прямоугольника
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Вычисляем центр найденного объекта
        center_x = x + w // 2
        center_y = y + h // 2

        # Находим центр кадра
        frame_height, frame_width, _ = frame.shape
        center_frame_x = frame_width // 2
        center_frame_y = frame_height // 2

        # Вычисляем отклонение от центра
        deviation_x = center_x - center_frame_x
        deviation_y = center_y - center_frame_y

        # Вычисляем значения для поворота камеры на основе отклонения
        axes_value_x = calculate_axes_x(deviation_x)
        axes_value_y = calculate_axes_y(deviation_y)
        axes = [axes_value_x, 0.0, 0.0, axes_value_y, 0.0]  # axes[3] для Y
        buttons = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

        # Подготовка и отправка данных
        data = {'axes': axes, 'buttons': buttons}
        serialized_data = pickle.dumps(data)
        sock.sendto(serialized_data, server_address)

        # Отображаем результат
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
        cv2.putText(frame, f'Deviation: ({deviation_x}, {deviation_y})', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Находим центр кадра
    frame_height, frame_width, _ = frame.shape
    center_frame_x = frame_width // 2
    center_frame_y = frame_height // 2

    # Рисуем крестик в центре кадра
    line_length = 20  # длина линий крестика
    color = (0, 0, 255)  # цвет (синий)
    thickness = 2  # толщина линий

    # Горизонтальная линия
    cv2.line(frame, (center_frame_x - line_length, center_frame_y),
             (center_frame_x + line_length, center_frame_y), color, thickness)
    # Вертикальная линия
    cv2.line(frame, (center_frame_x, center_frame_y - line_length),
             (center_frame_x, center_frame_y + line_length), color, thickness)

    # Показываем кадр
    cv2.imshow('Frame', frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
