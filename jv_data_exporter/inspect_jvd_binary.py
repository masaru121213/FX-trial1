import argparse
import csv
from pathlib import Path


def printable_ratio(bs: bytes) -> float:
    if not bs:
        return 0.0
    printable = 0
    for b in bs:
        if 32 <= b <= 126:
            printable += 1
    return printable / len(bs)


def hex_preview(bs: bytes, limit: int = 64) -> str:
    return bs[:limit].hex(" ")


def ascii_preview(bs: bytes, limit: int = 64) -> str:
    chars = []
    for b in bs[:limit]:
        if 32 <= b <= 126:
            chars.append(chr(b))
        else:
            chars.append(".")
    return "".join(chars)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in in_dir.rglob("*.jvd"):
        with open(path, "rb") as f:
            idx = 0
            while True:
                chunk = f.read(args.chunk_size)
                if not chunk:
                    break
                idx += 1
                rows.append({
                    "source_file": path.name,
                    "chunk_no": idx,
                    "byte_len": len(chunk),
                    "printable_ratio": round(printable_ratio(chunk), 4),
                    "hex_preview": hex_preview(chunk),
                    "ascii_preview": ascii_preview(chunk),
                })

    with open(out_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source_file", "chunk_no", "byte_len", "printable_ratio", "hex_preview", "ascii_preview"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print("done:", out_file)


if __name__ == "__main__":
    main()
