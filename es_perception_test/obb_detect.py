from ultralytics import YOLO
import cv2
import numpy as np
import sys

MODEL_WEIGHT_PATH = 'best_obb1.pt'
model = YOLO(MODEL_WEIGHT_PATH)

# Global counter for bounding box IDs
bbox_id_counter = 1

def get_rotated_box_points(x, y, width, height, angle):
    rectangle = np.array([[-width / 2, -height / 2], [width / 2, -height / 2],
                          [width / 2, height / 2], [-width / 2, height / 2]])
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    rotated_rectangle = np.dot(rectangle, rotation_matrix) + np.array([x, y])
    return np.int0(rotated_rectangle)

def empty_detect(img: cv2.Mat, empty_spot=0, conf_threshold=0.70):
    global model, bbox_id_counter
    results = model(img)
    bboxes = []
    for box, conf in zip(results[0].obb, results[0].obb.conf):
        class_id = int(box.cls[0].item())
        confidence = float(conf.item())
        if class_id == empty_spot and confidence >= conf_threshold:
            x, y, w, h, r = box.xywhr[0].tolist()
            bboxes.append((bbox_id_counter, x, y, w, h, r))
            bbox_id_counter += 1  # Increment ID for each new box
    return bboxes

def process_image(fn):
    global bbox_id_counter
    bbox_id_counter = 1  # Reset counter for each new image
    image = cv2.imread(fn)
    bboxes = empty_detect(image)
    print("Detected", len(bboxes), "empty spot(s)")
    for bb in bboxes:
        id, x, y, w, h, r = bb
        points = get_rotated_box_points(x, y, w, h, -r)
        cv2.polylines(image, [points], isClosed=True, color=(255, 0, 255), thickness=3)
        cv2.putText(image, str(id), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Results', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_webcam():
    global bbox_id_counter
    bbox_id_counter = 1  # Reset counter for each new session
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    print("Press space to exit")
    
    while True:
        _, image = cap.read()
        bboxes = empty_detect(image)
        for bb in bboxes:
            id, x, y, w, h, r = bb
            points = get_rotated_box_points(x, y, w, h, -r)
            cv2.polylines(image, [points], isClosed=True, color=(255, 0, 255), thickness=3)
            cv2.putText(image, str(id), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Tracking', image)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        fn = sys.argv[1]
        if fn.lower() == 'webcam':
            process_webcam()
        else:
            process_image(fn)
    else:
        print("Please provide a filename or 'webcam' as an argument.")
