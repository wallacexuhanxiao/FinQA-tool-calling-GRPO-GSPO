import argparse
import json
import re
from pathlib import Path

from parse_output import parse_model_output
from metrics import clamp_bbox, clamp_point


POINT_RE = re.compile(r'"?point_2d"?\s*[:=]\s*\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)')
BBOX_RE = re.compile(
    r'"?bbox_2d"?\s*[:=]\s*\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)'
)


def pixel_point_to_1000(point, width, height):
    return [point[0] / width * 1000, point[1] / height * 1000]


def pixel_bbox_to_1000(bbox, width, height):
    x1, y1, x2, y2 = bbox
    return [
        x1 / width * 1000,
        y1 / height * 1000,
        x2 / width * 1000,
        y2 / height * 1000,
    ]


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def raw_point_bbox(text):
    point_match = POINT_RE.search(text or "")
    bbox_match = BBOX_RE.search(text or "")
    point = [float(v) for v in point_match.groups()] if point_match else None
    bbox = [float(v) for v in bbox_match.groups()] if bbox_match else None
    return point, bbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--coord_mode", choices=["normalized", "pixel"], default="pixel")
    args = parser.parse_args()

    out = []
    for row in read_jsonl(args.input_jsonl):
        parsed = parse_model_output(row.get("raw_response", ""))
        raw_point, raw_bbox = raw_point_bbox(row.get("raw_response", ""))
        point = raw_point if raw_point is not None else parsed["pred_point_1000"]
        bbox = raw_bbox if raw_bbox is not None else parsed["pred_bbox_1000"]
        width = row.get("image_width")
        height = row.get("image_height")

        if args.coord_mode == "pixel" and width and height:
            if point is not None:
                point = clamp_point(pixel_point_to_1000(point, width, height))
            if bbox is not None:
                bbox = clamp_bbox(pixel_bbox_to_1000(bbox, width, height))
        else:
            if point is not None:
                point = clamp_point(point)
            if bbox is not None:
                bbox = clamp_bbox(bbox)

        values = []
        if point is not None:
            values.extend(point)
        if bbox is not None:
            values.extend(bbox)

        row.update(parsed)
        row["pred_point_1000"] = point
        row["pred_bbox_1000"] = bbox
        row["out_of_range"] = any(v < 0 or v > 1000 for v in values)
        row["coord_mode"] = args.coord_mode
        out.append(row)

    write_jsonl(args.output_jsonl, out)
    print(f"wrote {len(out)} rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
