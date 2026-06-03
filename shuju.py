import os
import glob

BASE = "F:/shiyan3"

for split in ["Train", "Val", "Test"]:
    path = os.path.join(BASE, split)
    if not os.path.exists(path):
        continue
    print(f"\n{'='*50}")
    print(f"{split} 集:")
    for cls in ["Normal", "Yawning", "Microsleep"]:
        cls_path = os.path.join(path, cls)
        if os.path.exists(cls_path):
            videos = glob.glob(os.path.join(cls_path, "*.mp4"))
            # 提取人员编号（假设文件名格式为 类别+人员_片段.mp4）
            persons = set()
            for v in videos:
                name = os.path.basename(v).replace(".mp4", "")
                # 例如 Microsleep16_001 → 提取人员ID "16"
                # 去掉类别前缀，取第一部分
                for prefix in ["Normal", "Yawning", "Microsleep"]:
                    if name.startswith(prefix):
                        rest = name[len(prefix):]
                        person_id = rest.split("_")[0]
                        persons.add(person_id)
                        break
            print(f"  {cls}: {len(videos)} 个视频, 人员: {sorted(persons, key=int)}")