from ultralytics import YOLO
from PIL import Image
import cv2
import sys

MODEL_WEIGHT_PATH = 'norma_last.pt'
model = YOLO(MODEL_WEIGHT_PATH)

def empty_detect(img: cv2.Mat, empty_spot=0):
    global model
    results = model(img)
    bboxes = []
    print(results)
    print(results[0].obb)
    for box in results[0].obb:
        class_id = int(box.cls[0].item())
        if class_id == empty_spot:
            bboxes.append(box.xywh[0].tolist())
    return bboxes

def main(fn):
    image = cv2.imread(fn)
    bboxes = empty_detect(image)
    print("Detected", len(bboxes), "empty spot(s)")
    for bb in bboxes:
        x, y, w, h = bb
        print("Corner", (x, y), "size", (w, h))
        cv2.rectangle(image, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 255), 3)
    cv2.imshow('Results', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        fn = sys.argv[1]
        main(fn)
    else:
        print("Please provide a filename as an argument.")

