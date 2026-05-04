import cv2 as cv
from pathlib import Path


def read_image(image_path: str | Path):
    return cv.imread(image_path)


def read_folder(folder_path: str | Path, extensions=None):
    folder = Path(folder_path)
    image_extensions = (
        {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        if extensions is None
        else extensions
    )
    images = []
    for image_path in sorted(folder.iterdir()):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in image_extensions:
            continue
        image = read_image(image_path)
        if image is None:
            continue
        images.append(image)
    return images


def read_video(video_path: str | Path, fps: int = 30):
    cap = cv.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.add(frame)
        if cv.waitKey(1) & 255 == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()
    return frames
