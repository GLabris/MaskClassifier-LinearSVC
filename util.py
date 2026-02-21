import cv2


def crop_to_face(img,face_detection):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    H, W, _ = img_rgb.shape

    if out.detections is not None:
        bbox = out.detections[0].location_data.relative_bounding_box

        x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

        x1 = int(x1 * W)
        y1 = int(y1 * H)
        w = int(w * W)
        h = int(h * H)

        return img_rgb[y1:y1 + h, x1:x1 + w]

    return None
