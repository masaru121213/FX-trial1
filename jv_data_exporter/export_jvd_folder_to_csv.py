import csv
import argparse
from pathlib import Path
from jv_parser import parse_lines_from_jvd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for path in in_dir.rglob("*.jvd"):
        for r in parse_lines_from_jvd(str(path)):
            rows.append(r)

    if not rows:
        print("no records")
        return

    with open(out_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("done:", out_file)


if __name__ == "__main__":
    main()
