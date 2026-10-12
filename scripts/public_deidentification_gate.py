"""Fail closed when distributable files contain private identifiers or metadata."""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree


ENCODED_IDENTIFIERS = (
    "VVNBQ0g=",
    "VW5pdmVyc2lkYWQgZGUgU2FudGlhZ28=",
    "U2FudGlhZ28=",
    "SW5nZW5pZXLDrWEgQ2l2aWwgSW5kdXN0cmlhbA==",
    "SW5nZW5pZXJpYSBDaXZpbCBJbmR1c3RyaWFs",
    "SUNJ",
    "SUNJX1NJTQ==",
    "MTYwMDM=",
    "Q09ESUdPX1VOSVY=",
    "TE9B",
    "UEFDRQ==",
    "QkVB",
    "K01D",
    "QHVzYWNoLmNs",
)
IDENTIFIERS = tuple(
    base64.b64decode(value).decode("utf-8") for value in ENCODED_IDENTIFIERS
)
SHORT_WORDS = frozenset(value.casefold() for value in IDENTIFIERS if value.isalpha() and len(value) <= 4)
PRIVATE_FINGERPRINTS = (
    "".join(("Inputs_Generador_", "Cursos_V2.xlsx")),
    "".join(("Inputs_Generador_", "Trayectorias_V2.xlsx")),
    "".join(("Info_Generador_", "Trayectorias.xlsx")),
    "".join(("candidate_", "manifest.json")),
    "".join(("cal_", "p3b")),
    "".join(("hist", "025")),
    "".join(("b745fa58", "ed28895daf411fd923f924fcf6d03964")),
)
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".csv", ".xml", ".gitignore"}
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
LOCAL_FRAGMENTS = (
    "".join(("c", ":", "\\", "users", "\\")),
    "".join(("/", "users", "/")),
    "".join(("one", "drive")),
)


def token_matches(text: str) -> list[str]:
    folded = text.casefold()
    matches: list[str] = []
    for value in IDENTIFIERS:
        needle = value.casefold()
        if needle.isdigit():
            if re.search(rf"(?<![0-9a-z]){re.escape(needle)}(?![0-9a-z])", folded):
                matches.append(value)
        elif needle in SHORT_WORDS:
            if re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", folded):
                matches.append(value)
        elif needle in folded:
            matches.append(value)
    for fingerprint in PRIVATE_FINGERPRINTS:
        if fingerprint.casefold() in folded:
            matches.append("PRIVATE_FINGERPRINT")
    return sorted(set(matches))


def local_matches(text: str) -> list[str]:
    folded = text.casefold()
    return [fragment for fragment in LOCAL_FRAGMENTS if fragment in folded]


def scan_text(path: Path, label: str, counters: dict[str, list[str]]) -> None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    for match in token_matches(text):
        counters["forbidden"].append(f"{label}:{match}")
    for match in local_matches(text):
        counters["local"].append(f"{label}:{match}")
    for match in EMAIL_PATTERN.findall(text):
        counters["personal"].append(f"{label}:{match}")


def scan_xlsx(path: Path, label: str, counters: dict[str, list[str]]) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        lower_names = [name.casefold() for name in names]
        disallowed_parts = (
            "xl/comments",
            "xl/threadedcomments",
            "xl/persons",
            "xl/externallinks",
            "docprops/custom.xml",
        )
        for name, lower in zip(names, lower_names, strict=True):
            if any(part in lower for part in disallowed_parts):
                counters["structural"].append(f"{label}:{name}")
            if not (lower.endswith(".xml") or lower.endswith(".rels")):
                continue
            text = archive.read(name).decode("utf-8", errors="ignore")
            for match in token_matches(text):
                counters["forbidden"].append(f"{label}!{name}:{match}")
            for match in local_matches(text):
                counters["local"].append(f"{label}!{name}:{match}")
            for match in EMAIL_PATTERN.findall(text):
                counters["personal"].append(f"{label}!{name}:{match}")
            folded = text.casefold()
            if "targetmode=\"external\"" in folded or "<hyperlink" in folded:
                counters["structural"].append(f"{label}!{name}:external-reference")

        if "docprops/core.xml" in names:
            root = ElementTree.fromstring(archive.read("docProps/core.xml"))
            for element in root.iter():
                local_name = element.tag.rsplit("}", 1)[-1]
                if local_name in {"creator", "lastModifiedBy"} and (element.text or "").strip():
                    counters["personal"].append(
                        f"{label}!docProps/core.xml:{local_name}={element.text.strip()}"
                    )


def scan_roots(roots: list[Path]) -> dict[str, object]:
    counters: dict[str, list[str]] = {
        "forbidden": [],
        "personal": [],
        "local": [],
        "structural": [],
    }
    files_scanned = 0
    for root in roots:
        resolved = root.resolve()
        for path in sorted(item for item in resolved.rglob("*") if item.is_file()):
            if ".git" in path.parts or "__pycache__" in path.parts:
                continue
            label = str(path.relative_to(resolved)).replace("\\", "/")
            suffix = path.suffix.casefold()
            if suffix == ".xlsx":
                scan_xlsx(path, label, counters)
                files_scanned += 1
            elif suffix in TEXT_SUFFIXES or path.name in {".gitignore", ".gitattributes"}:
                scan_text(path, label, counters)
                files_scanned += 1
    return {
        "files_scanned": files_scanned,
        "forbidden_token_matches": len(counters["forbidden"]),
        "personal_metadata_matches": len(counters["personal"]),
        "local_path_matches": len(counters["local"]),
        "structural_privacy_matches": len(counters["structural"]),
        "details": counters,
        "passed": not any(counters.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--extra-root", action="append", default=[], type=Path)
    args = parser.parse_args(argv)
    report = scan_roots([args.root, *args.extra_root])
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
