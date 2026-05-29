import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


SHAPE_BY_COLOR = {
    "red": "cylinder",
    "blue": "cube",
    "green": "sphere",
    "yellow": "cylinder",
}


def update_geom(root, geom_name, shape):
    geom = root.find(f".//geom[@name='{geom_name}']")
    if geom is None:
        raise ValueError(f"geom not found: {geom_name}")

    if shape == "cube":
        geom.set("type", "box")
        geom.set("size", "0.010 0.010 0.010")
    elif shape == "sphere":
        geom.set("type", "sphere")
        geom.set("size", "0.010")
    elif shape == "cylinder":
        geom.set("type", "cylinder")
        geom.set("size", "0.0075 0.01")
    else:
        raise ValueError(f"unsupported shape: {shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_xml", default="Raccoon_colored_cylinder.xml")
    parser.add_argument("--out_xml", default="Raccoon_multishape_objects.xml")
    parser.add_argument("--summary_json", default="evidence/multishape_scene_summary.json")
    args = parser.parse_args()

    base_xml = Path(args.base_xml)
    out_xml = Path(args.out_xml)
    summary_json = Path(args.summary_json)

    tree = ET.parse(base_xml)
    root = tree.getroot()

    update_geom(root, "geom_red", "cylinder")
    update_geom(root, "geom_blue", "cube")
    update_geom(root, "geom_green", "sphere")
    update_geom(root, "geom_yellow", "cylinder")

    root.set("model", "Raccoon_multishape_objects")
    tree.write(out_xml, encoding="utf-8", xml_declaration=False)

    summary = {
        "base_xml": str(base_xml),
        "out_xml": str(out_xml),
        "shape_by_color": SHAPE_BY_COLOR,
        "new_object_types": ["cube", "sphere"],
        "note": "blue object is changed to cube, green object is changed to sphere; red/yellow remain cylinders.",
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] wrote {out_xml}")
    print(f"[OK] wrote {summary_json}")


if __name__ == "__main__":
    main()
