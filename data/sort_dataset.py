import argparse
from pathlib import Path

def extract_keys(path):
    path = path.strip()
    parts = path.split('/')
    if len(parts) < 3:
        return ("", "")
    scene = parts[1]
    filename = parts[2]
    return (scene, filename)

def main(args):
    root = Path(args.root)
    file_path = root / Path(args.file)
    with open(file_path, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]
    sorted_lines = sorted(lines, key=extract_keys)
    new_file_path = file_path.parent / f"{file_path.stem}_sorted{file_path.suffix}"

    with open(new_file_path, "w") as f:
        for line in sorted_lines:
            f.write(line + "\n")

    print(f"排序结果已写入: {new_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='lidar Visualization')
    parser.add_argument('root', type=str, default='/EmbodiedOcc/data/occscannet')
    parser.add_argument('file', type=str, default='train_mini_final.txt')
    args = parser.parse_args()
    main(args)