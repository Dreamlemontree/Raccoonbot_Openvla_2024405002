import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw


def select_frames(frames, count):
    if len(frames) <= count:
        return frames
    return [frames[round(i * (len(frames) - 1) / (count - 1))] for i in range(count)]


def make_contact_sheet(frames, out_path, count):
    selected = select_frames(frames, max(2, count))
    images = [Image.open(path).convert("RGB") for path in selected]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    label_height = 28

    canvas = Image.new("RGB", (width * len(images), height + label_height), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, (path, image) in enumerate(zip(selected, images)):
        x = idx * width
        canvas.paste(image, (x, label_height))
        draw.text((x + 6, 7), path.name, fill="black")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def make_video(frames, out_path, fps):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("[WARN] ffmpeg is not installed. Skipping MP4 generation.")
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            list_path = Path(f.name)
            for frame in frames:
                escaped = str(frame.resolve()).replace("'", r"'\''")
                f.write(f"file '{escaped}'\n")
                f.write(f"duration {1.0 / fps:.8f}\n")
            escaped = str(frames[-1].resolve()).replace("'", r"'\''")
            f.write(f"file '{escaped}'\n")

        command = [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-vf",
            f"fps={fps}",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ]
        subprocess.run(command, check=True)
    finally:
        if list_path is not None:
            list_path.unlink(missing_ok=True)
    return True


def main():
    parser = argparse.ArgumentParser(description="Visualize one raw MuJoCo dataset episode.")
    parser.add_argument("--episode_dir", required=True)
    parser.add_argument("--out_png", required=True)
    parser.add_argument("--out_mp4", required=True)
    parser.add_argument("--n_show", type=int, default=6)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    episode_dir = Path(args.episode_dir)
    frames = sorted(episode_dir.glob("frame_*.png"))
    if not frames:
        frames = sorted(episode_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No PNG frames found in {episode_dir}")

    out_png = Path(args.out_png)
    out_mp4 = Path(args.out_mp4)

    make_contact_sheet(frames, out_png, args.n_show)
    video_created = make_video(frames, out_mp4, args.fps)

    print(f"[OK] episode frames: {len(frames)}")
    print(f"[OK] contact sheet: {out_png}")
    if video_created:
        print(f"[OK] mp4: {out_mp4}")


if __name__ == "__main__":
    main()
