import os
import shutil

BASE = "F:/shiyan3/yolo_data"


# 先获取 train/val/test 里的所有图片名，用来匹配标签
def get_img_names(mode):
    img_dir = os.path.join(BASE, "images", mode)
    return [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(".jpg")]


# 把标签文件移动到对应文件夹
def organize_labels(mode, img_names):
    target_dir = os.path.join(BASE, "labels", mode)
    os.makedirs(target_dir, exist_ok=True)

    count = 0
    # 遍历 labels 根目录的所有 txt 文件
    for file in os.listdir(os.path.join(BASE, "labels")):
        if file.endswith(".txt"):
            file_name = os.path.splitext(file)[0]
            if file_name in img_names:
                src = os.path.join(BASE, "labels", file)
                dst = os.path.join(target_dir, file)
                shutil.move(src, dst)
                count += 1
    print(f"✅ {mode} 标签文件整理完成，共移动 {count} 个文件")


# 1. 获取所有图片名
train_imgs = get_img_names("train")
val_imgs = get_img_names("val")
test_imgs = get_img_names("test")

# 2. 把标签文件分到对应文件夹
organize_labels("train", train_imgs)
organize_labels("val", val_imgs)
organize_labels("test", test_imgs)

print("\n🎉 所有标签文件已按 YOLO 标准结构整理完成！")