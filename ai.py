import os

import pandas as pd
from ultralytics import YOLO


def train_model(
    weights_path: str = 'best.pt',
    data_config: str = 'data.yaml',
    epochs: int = 20,
    imgsz: int = 640,
    batch: int = 16,
    project: str = '.',
    run_name: str = 'fine_tune',
):
    """
    Load pretrained weights and continue training on the provided dataset.
    After training, overwrite best.pt with the new best weights.
    """
    model = YOLO(weights_path)
    model.train(
        data=data_config,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=run_name,
        exist_ok=True,
    )

    # After training, the new best weights are saved under runs/train/{run_name}/weights/best.pt
    new_best = os.path.join(project, 'runs', 'train', run_name, 'weights', 'best.pt')
    if os.path.exists(new_best):
        os.replace(new_best, weights_path)
        print(f"Updated '{weights_path}' with new best weights.")
    else:
        print(f"Warning: '{new_best}' not found. Keeping original weights.")


def create_submission(
    weights_path: str = 'best.pt',
    test_folder: str = os.path.join('test', 'images'),
    output_csv: str = 'submission.csv',
    imgsz: int = 640,
    conf_threshold: float = 0.25,
):
    """
    Run inference on the test images and produce a submission.csv
    with one prediction (highest confidence) per image.
    """
    model = YOLO(weights_path)
    results = model.predict(
        source=test_folder,
        imgsz=imgsz,
        conf=conf_threshold,
        save=False,
    )

    records = []
    for res in results:
        image_id = os.path.basename(res.path)
        # xywh normalized: center_x, center_y, width, height
        boxes = res.boxes.xywhn.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()

        if len(boxes) == 0:
            # No detection -> empty box
            records.append([image_id, 0, 0.0, 0.0, 0.0, 0.0])
        else:
            # pick the box with highest confidence
            idx = confs.argmax()
            x_center, y_center, w, h = boxes[idx]
            records.append([
                image_id,
                0,
                float(x_center),
                float(y_center),
                float(w),
                float(h),
            ])

    df = pd.DataFrame(records, columns=['ImageId', 'ClassId', 'X', 'Y', 'Width', 'Height'])
    df.to_csv(output_csv, index=False)
    print(f"Saved submission file to '{output_csv}'.")


if __name__ == '__main__':
    # 1) Fine-tune the model (will overwrite best.pt)
    train_model(epochs=20, batch=16)

    # 2) Generate submission.csv on test set
    create_submission(conf_threshold=0.25)
