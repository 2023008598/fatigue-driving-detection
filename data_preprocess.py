import os
import cv2
import glob

CLASSES = ['Normal', 'Yawning', 'Microsleep']
CLASS_TO_ID = {cls: i for i, cls in enumerate(CLASSES)}

BASE_DIR = 'f:/shiyan3'
TRAIN_DIR = os.path.join(BASE_DIR, 'Train')
VAL_DIR = os.path.join(BASE_DIR, 'Val')
TEST_DIR = os.path.join(BASE_DIR, 'Test')
YOLO_DIR = os.path.join(BASE_DIR, 'yolo_data')
os.makedirs(YOLO_DIR, exist_ok=True)
FRAME_SKIP = 2

def extract_frames(video_path, output_dir, class_id, video_name):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_SKIP == 0:
            img_name = f"{video_name}_{saved_count:04d}.jpg"
            img_path = os.path.join(output_dir, 'images', img_name)
            cv2.imwrite(img_path, frame)

            lab_name = f"{video_name}_{saved_count:04d}.txt"
            lab_path = os.path.join(output_dir, 'labels', lab_name)
            with open(lab_path, 'w') as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
            saved_count += 1
        frame_count += 1
    cap.release()
    return saved_count

def process_dataset(input_dir, output_dir, set_name):
    images_dir = os.path.join(output_dir, 'images', set_name)
    labels_dir = os.path.join(output_dir, 'labels', set_name)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    file_list = []

    for cls in CLASSES:
        class_dir = os.path.join(input_dir, cls)
        if not os.path.exists(class_dir):
            continue
        cid = CLASS_TO_ID[cls]
        for vp in glob.glob(os.path.join(class_dir, '*.mp4')):
            vname = os.path.splitext(os.path.basename(vp))[0]
            cnt = extract_frames(vp, output_dir, cid, vname)
            for i in range(cnt):
                file_list.append(f"images/{set_name}/{vname}_{i:04d}.jpg")

    with open(os.path.join(output_dir, f"{set_name}.txt"), 'w') as f:
        f.write('\n'.join(file_list))
    print(f"{set_name}: {len(file_list)} 帧")

def create_data_yaml():
    content = f"""train: {YOLO_DIR}/train.txt
val: {YOLO_DIR}/val.txt
test: {YOLO_DIR}/test.txt
nc: {len(CLASSES)}
names: {CLASSES}"""
    with open(os.path.join(YOLO_DIR, 'data.yaml'), 'w') as f:
        f.write(content)

if __name__ == '__main__':
    process_dataset(TRAIN_DIR, YOLO_DIR, 'train')
    process_dataset(VAL_DIR, YOLO_DIR, 'val')
    process_dataset(TEST_DIR, YOLO_DIR, 'test')
    create_data_yaml()