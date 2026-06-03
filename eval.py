from ultralytics import YOLO
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ========== 路径配置 ==========
BASE = "F:/shiyan3"
model_path = f"{BASE}/runs/detect/runs/commodity/commodity_detect5/weights/best.pt"
test_txt = f"{BASE}/yolo_data/test.txt"
output_dir = f"{BASE}/test_results"
os.makedirs(output_dir, exist_ok=True)

# ========== 加载模型和数据 ==========
CLASSES = ['Normal', 'Yawning', 'Microsleep']
model = YOLO(model_path)

df = pd.read_csv(test_txt, header=None, names=["path"])

# ✅ 修正：用最后一个下划线前的内容作为视频ID（类别_人员_片段）
df["video"] = df["path"].apply(lambda x: "_".join(os.path.basename(x).split("_")[:-1]))

# 提取真实标签
df["true"] = df["path"].apply(lambda x: 0 if "Normal" in x else 1 if "Yawning" in x else 2)

print(f"测试集: {len(df)} 帧, {df['video'].nunique()} 个视频")

# ========== 逐帧推理 ==========
preds, confs = [], []
for i, p in enumerate(df["path"]):
    r = model(p, verbose=False)
    if r[0].boxes is not None and len(r[0].boxes) > 0:
        preds.append(int(r[0].boxes.cls[0].item()))
        confs.append(float(r[0].boxes.conf[0].item()))
    else:
        preds.append(0)
        confs.append(0.0)
    if (i+1) % 1000 == 0:
        print(f"  推理进度: {i+1}/{len(df)}")

df["pred"] = preds
df["conf"] = confs

# ========== 逐帧准确率 ==========
frame_acc = (df["true"] == df["pred"]).mean()
print(f"\n📊 逐帧准确率: {frame_acc:.4f} ({frame_acc:.2%})")

# 各类逐帧准确率
for cls_id, cls_name in enumerate(CLASSES):
    cls_df = df[df["true"] == cls_id]
    cls_acc = (cls_df["true"] == cls_df["pred"]).mean()
    print(f"   {cls_name}: {len(cls_df)} 帧, 准确率 {cls_acc:.2%}")

# ========== 视频级投票 ==========
video_df = df.groupby("video").agg({
    "true": "first",
    "pred": lambda x: x.value_counts().idxmax(),
    "conf": "mean"
}).reset_index()

video_df.columns = ["filename", "true_label", "pred_label", "confidence"]
video_df.to_csv(f"{output_dir}/pred_test.csv", index=False)
print(f"\n✅ pred_test.csv 已保存")

# ========== 指标 ==========
y_true = video_df["true_label"]
y_pred = video_df["pred_label"]

rep = classification_report(y_true, y_pred, target_names=CLASSES, digits=4, zero_division=0)
print("\n========== 视频级分类报告 ==========")
print(rep)

with open(f"{output_dir}/metrics.txt", "w", encoding="utf-8") as f:
    f.write(rep)
print("✅ metrics.txt 已保存")

# ========== 混淆矩阵 ==========
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Video-level)")
plt.savefig(f"{output_dir}/confusion_matrix.png", bbox_inches="tight", dpi=150)
print("✅ confusion_matrix.png 已保存")

# ========== 最终准确率 ==========
acc = (y_true == y_pred).mean()
print(f"\n🎯 视频级准确率: {acc:.4f} ({acc:.2%})")

# ========== 错误视频分析 ==========
print("\n========== 预测错误的视频 ==========")
errors = video_df[y_true != y_pred]
for _, row in errors.iterrows():
    true_name = CLASSES[int(row["true_label"])]
    pred_name = CLASSES[int(row["pred_label"])]
    print(f"  {row['filename']}: 真实={true_name}, 预测={pred_name}, 置信度={row['confidence']:.3f}")

print("\n✅ 评估完成！")