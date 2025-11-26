import sys
sys.path.insert(0, "./ultralytics")  # 使用当前目录下的本地源码

from ultralytics import YOLO

def main():
    model = YOLO("gshead.yaml")  # 从零构建（也可以用 yolov8n.pt）

    model.train(
        data="data.yaml",
        epochs=500,
        imgsz=640,
        batch=8,
        device=0,  # GPU id，或 "cpu"
        project="results",
        name="baseline",
        exist_ok=True
    )

if __name__ == "__main__":
    main()
