"""Index presentation text, notes and image identities without changing decks."""
import argparse
import hashlib
import json
from pathlib import Path
import posixpath
import xml.etree.ElementTree as ET
import zipfile


def extract_deck(path):
    namespace = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    slides = []
    with zipfile.ZipFile(path) as deck:
        names = set(deck.namelist())
        # Use presentation order, not filenames or printed footer numbers.
        rel_ns = "{http://schemas.openxmlformats.org/package/2006/relationships}"
        rels = ET.fromstring(deck.read("ppt/_rels/presentation.xml.rels"))
        targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels.findall(rel_ns + "Relationship")}
        presentation = ET.fromstring(deck.read("ppt/presentation.xml"))
        slide_ids = presentation.findall(".//{http://schemas.openxmlformats.org/presentationml/2006/main}sldId")
        ordered = [posixpath.normpath("ppt/" + targets[item.attrib[
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]]) for item in slide_ids]
        for number, name in enumerate(ordered, 1):
            root = ET.fromstring(deck.read(name))
            text = [node.text or "" for node in root.findall(".//a:t", namespace)]
            relationships = posixpath.dirname(name) + "/_rels/" + posixpath.basename(name) + ".rels"
            notes, figures = [], []
            if relationships in names:
                for rel in ET.fromstring(deck.read(relationships)):
                    if rel.attrib.get("TargetMode") == "External":
                        continue
                    target = rel.attrib["Target"]
                    target = target.lstrip("/") if target.startswith("/") else posixpath.normpath(
                        posixpath.dirname(name) + "/" + target)
                    if target not in names:
                        continue
                    if rel.attrib["Type"].endswith("/notesSlide"):
                        note = ET.fromstring(deck.read(target))
                        notes = [node.text or "" for node in note.findall(".//a:t", namespace)]
                    elif rel.attrib["Type"].endswith("/image"):
                        figures.append({"part": target, "sha256": hashlib.sha256(deck.read(target)).hexdigest()})
            slides.append({"slide": number, "part": name, "text": text, "notes": notes, "images": figures})
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "slides": slides}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decks = [extract_deck(path) for path in sorted(args.directory.glob("*.pptx"))
             if not path.name.startswith("~$")]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "slides.json").write_text(json.dumps(decks, indent=2, ensure_ascii=False) + "\n")
    for deck in decks:
        name = Path(deck["path"]).stem
        content = [f"# {name}", f"Source: {deck['path']}", f"SHA256: {deck['sha256']}"]
        for slide in deck["slides"]:
            content += [f"\n## Slide {slide['slide']}", "\n".join(slide["text"]),
                        "\nNotes:\n" + "\n".join(slide["notes"])]
        (args.output / f"{name}.md").write_text("\n\n".join(content) + "\n")
    print(json.dumps({"decks": len(decks), "slides": sum(len(deck["slides"]) for deck in decks)}))
