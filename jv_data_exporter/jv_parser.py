import os


def try_decode(data):
    encodings = ["cp932", "utf-8-sig", "utf-8"]
    for enc in encodings:
        try:
            return data.decode(enc)
        except Exception:
            pass
    return data.decode("cp932", errors="replace")



def parse_lines_from_jvd(path):
    with open(path, "rb") as f:
        raw = f.read()

    text = try_decode(raw)

    line_no = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        line_no += 1
        yield {
            "source_file": os.path.basename(path),
            "line_no": line_no,
            "record_prefix_2": line[:2],
            "record_prefix_8": line[:8],
            "char_len": len(line),
            "raw_text": line,
        }
