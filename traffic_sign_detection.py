# traffic_sign_detection.py 

import cv2
import numpy as np

def filter_signs_by_color(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower1, upper1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    lower2, upper2 = np.array([170, 70, 50]), np.array([180, 255, 255])
    mask_1 = cv2.inRange(image, lower1, upper1)
    mask_2 = cv2.inRange(image, lower2, upper2)
    mask_r = cv2.bitwise_or(mask_1, mask_2)
    lower3, upper3 = np.array([100, 150, 0]), np.array([140, 255, 255])
    mask_b = cv2.inRange(image, lower3, upper3)
    mask_final = cv2.bitwise_or(mask_r, mask_b)
    return mask_final

def get_boxes_from_mask(mask):
    bboxes = []
    nccomps = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    numLabels, labels, stats, centroids = nccomps
    im_height, im_width = mask.shape[:2]
    for i in range(numLabels):
        x, y, w, h, area = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if w < 20 or h < 20: continue
        if w > 0.8 * im_width or h > 0.8 * im_height: continue
        if w / h > 2.0 or h / w > 2.0: continue
        bboxes.append([x, y, w, h])
    return bboxes


def detect_traffic_signs(img, model, draw=None):
    classes = ['unknown', 'left', 'no_left', 'right',
               'no_right', 'straight', 'stop']
    mask = filter_signs_by_color(img)
    bboxes = get_boxes_from_mask(mask)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    detected_signs = []
    for bbox in bboxes:
        x, y, w, h = bbox
        sub_image = img_rgb[y:y+h, x:x+w]
        if w * h < 1500: continue
        if sub_image.shape[0] < 20 or sub_image.shape[1] < 20: continue
        sub_image_resized = cv2.resize(sub_image, (32, 32))
        sub_image_expanded = np.expand_dims(sub_image_resized, axis=0)
        model.setInput(sub_image_expanded)
        preds = model.forward()[0]
        cls_id = preds.argmax()
        score = preds[cls_id]
        if cls_id == 0 or score < 0.75: continue
        detected_signs.append({
            'class': classes[cls_id],
            'score': score,
            'bbox': (x, y, w, h)
        })

    if detected_signs:
        detected_signs.sort(key=lambda s: s['bbox'][0], reverse=True)

    if draw is not None:
        if detected_signs:
            best_sign = detected_signs[0]
            x, y, w, h = best_sign['bbox']
            text = f"{best_sign['class']} {best_sign['score']:.2f}"
            cv2.rectangle(draw, (x, y), (x+w, y+h), (0, 255, 0), 4)
            cv2.putText(draw, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return detected_signs