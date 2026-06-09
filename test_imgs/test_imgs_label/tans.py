import os
import json

# 类别映射：按你的数据集类别修改
# YOLO 中类别必须是数字，从 0 开始
CLASS_MAP = {
    "red": 2,
    "yellow": 1,
    "green": 0,
}


def convert_box_to_yolo(box, img_w, img_h):
    """
    原始框格式：
    xmin, ymin, xmax, ymax

    YOLO 格式：
    x_center, y_center, width, height
    并且全部除以图片宽高归一化
    """
    xmin = box["xmin"]
    ymin = box["ymin"]
    xmax = box["xmax"]
    ymax = box["ymax"]

    x_center = (xmin + xmax) / 2 / img_w
    y_center = (ymin + ymax) / 2 / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h

    return x_center, y_center, width, height


def convert_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = data["size"]["width"]
    img_h = data["size"]["height"]

    objects = data["outputs"]["object"]

    yolo_lines = []

    for obj in objects:
        class_name = obj["name"]

        if class_name not in CLASS_MAP:
            print(f"跳过未知类别：{class_name}，文件：{json_path}")
            continue

        class_id = CLASS_MAP[class_name]
        box = obj["bndbox"]

        x_center, y_center, w, h = convert_box_to_yolo(box, img_w, img_h)

        line = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
        yolo_lines.append(line)

    txt_path = os.path.splitext(json_path)[0] + ".txt"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))

    print(f"转换完成：{json_path} -> {txt_path}")


def main():
    current_dir = os.getcwd()

    for filename in os.listdir(current_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(current_dir, filename)
            convert_json(json_path)


if __name__ == "__main__":
    main()