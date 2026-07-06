from __future__ import annotations

import math
import re
import statistics
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any


SVG_RE = re.compile(r"<svg\b[\s\S]*?</svg>", re.IGNORECASE)
HEX_RE = re.compile(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?(?:[0-9a-fA-F]{2})?\b")
NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")

ALLOWED_TAGS = {
    "svg",
    "defs",
    "g",
    "path",
    "circle",
    "ellipse",
    "rect",
    "polygon",
    "polyline",
    "line",
    "linearGradient",
    "radialGradient",
    "stop",
    "clipPath",
    "mask",
    "filter",
    "feGaussianBlur",
    "feDropShadow",
    "feOffset",
    "feBlend",
    "feColorMatrix",
}
DISALLOWED_TAGS = {"script", "image", "foreignObject", "iframe", "object", "embed", "video", "audio"}
PRIMITIVE_TAGS = {"path", "circle", "ellipse", "rect", "polygon", "polyline", "line"}
GEOMETRY_ATTRS = {
    "x",
    "y",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
    "r",
    "rx",
    "ry",
    "width",
    "height",
}
COLOR_ATTRS = {"fill", "stroke", "stop-color", "color"}
SAFE_COLOR_NAMES = {
    "black",
    "white",
    "red",
    "green",
    "blue",
    "navy",
    "teal",
    "cyan",
    "yellow",
    "orange",
    "gold",
    "purple",
    "pink",
    "brown",
    "gray",
    "grey",
    "silver",
    "none",
    "transparent",
    "currentColor",
}
COLOR_WORDS = {
    "red": {"red", "coral", "crimson", "rose", "pink"},
    "orange": {"orange", "amber", "gold", "golden", "yellow"},
    "yellow": {"yellow", "cream", "warm"},
    "green": {"green", "teal", "mint", "emerald", "leaf"},
    "blue": {"blue", "navy", "cyan", "sky"},
    "purple": {"purple", "violet", "lavender"},
    "black": {"black", "charcoal"},
    "white": {"white"},
    "gray": {"gray", "grey", "silver"},
    "brown": {"brown", "tan", "earth"},
}
SHAPE_TERMS = {
    "circle": {"circle", "round", "circular", "coin", "seal", "dot", "sun", "moon"},
    "ellipse": {"oval", "ellipse"},
    "rect": {"square", "rectangle", "rounded-square", "badge", "card", "frame"},
    "path": {
        "leaf",
        "sprout",
        "plant",
        "wave",
        "mountain",
        "flame",
        "ribbon",
        "heart",
        "star",
        "shield",
        "note",
        "music",
        "curve",
        "arc",
    },
    "line": {"line", "tick", "staff"},
    "polygon": {"triangle", "diamond", "star"},
}


@dataclass
class ParsedSvg:
    text: str
    root: ET.Element | None
    error: str | None
    exact_svg_count: int
    outside_text_len: int


def extract_svg(text: str) -> str:
    match = SVG_RE.search(text or "")
    return match.group(0) if match else (text or "").strip()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _score_range(value: float, low: float, high: float, soft_low: float, soft_high: float) -> float:
    if low <= value <= high:
        return 1.0
    if value < low:
        return _clamp01((value - soft_low) / max(1e-9, low - soft_low))
    return _clamp01((soft_high - value) / max(1e-9, soft_high - high))


def _parse_svg(raw: str) -> ParsedSvg:
    text = (raw or "").strip()
    matches = list(SVG_RE.finditer(text))
    svg_text = matches[0].group(0) if matches else text
    outside_len = 0
    if matches:
        outside = text[: matches[0].start()] + text[matches[0].end() :]
        outside_len = len(outside.strip())
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError as exc:
        return ParsedSvg(svg_text, None, str(exc), len(matches), outside_len)
    return ParsedSvg(svg_text, root, None, len(matches), outside_len)


def _iter_elements(root: ET.Element | None) -> list[ET.Element]:
    return list(root.iter()) if root is not None else []


def _component_validity(parsed: ParsedSvg) -> tuple[float, dict[str, Any]]:
    if parsed.root is None:
        return 0.0, {"xml_valid": False, "error": parsed.error}

    root_name = _local_name(parsed.root.tag)
    view_box = re.sub(r"\s+", " ", parsed.root.attrib.get("viewBox", "")).strip()
    xmlns_ok = "xmlns" in parsed.root.attrib or parsed.root.tag.startswith("{http://www.w3.org/2000/svg}")
    view_ok = view_box == "0 0 256 256"
    single_svg = parsed.exact_svg_count == 1
    no_outside = parsed.outside_text_len == 0
    score = statistics.mean(
        [
            1.0 if root_name == "svg" else 0.0,
            1.0 if xmlns_ok else 0.4,
            1.0 if view_ok else 0.5,
            1.0 if single_svg else 0.3,
            1.0 if no_outside else 0.5,
        ]
    )
    return score, {
        "xml_valid": True,
        "root": root_name,
        "viewBox": view_box,
        "xmlns": xmlns_ok,
        "single_svg": single_svg,
        "outside_text_len": parsed.outside_text_len,
    }


def _component_safety(elements: list[ET.Element]) -> tuple[float, dict[str, Any]]:
    bad_tags: list[str] = []
    unknown_tags: list[str] = []
    bad_attrs: list[str] = []
    external_refs = 0

    for elem in elements:
        tag = _local_name(elem.tag)
        if tag in DISALLOWED_TAGS:
            bad_tags.append(tag)
        if tag not in ALLOWED_TAGS:
            unknown_tags.append(tag)
        for key, value in elem.attrib.items():
            local_key = _local_name(key)
            lower_value = value.lower()
            if local_key.startswith("on"):
                bad_attrs.append(local_key)
            if "javascript:" in lower_value or "data:" in lower_value or "http://" in lower_value or "https://" in lower_value:
                external_refs += 1
            if "url(" in lower_value and "#" not in lower_value:
                external_refs += 1

    penalties = len(bad_tags) * 0.5 + len(unknown_tags) * 0.08 + len(bad_attrs) * 0.4 + external_refs * 0.3
    score = _clamp01(1.0 - penalties)
    return score, {
        "bad_tags": sorted(set(bad_tags)),
        "unknown_tags": sorted(set(unknown_tags)),
        "bad_attrs": sorted(set(bad_attrs)),
        "external_refs": external_refs,
    }


def _numeric_values(elements: list[ET.Element]) -> list[float]:
    nums: list[float] = []
    for elem in elements:
        for key, value in elem.attrib.items():
            local_key = _local_name(key)
            if local_key in GEOMETRY_ATTRS or local_key in {"d", "points", "transform"}:
                for raw in NUMBER_RE.findall(value):
                    try:
                        number = float(raw)
                    except ValueError:
                        continue
                    if math.isfinite(number):
                        nums.append(number)
    return nums


def _component_geometry(elements: list[ET.Element]) -> tuple[float, dict[str, Any]]:
    nums = _numeric_values(elements)
    if not nums:
        return 0.25, {"numeric_count": 0}
    moderate = sum(-32.0 <= value <= 288.0 for value in nums) / len(nums)
    not_extreme = sum(abs(value) <= 1024.0 for value in nums) / len(nums)
    spread = max(nums) - min(nums)
    spread_score = _score_range(spread, 48.0, 360.0, 0.0, 2048.0)
    score = statistics.mean([moderate, not_extreme, spread_score])
    return score, {
        "numeric_count": len(nums),
        "within_soft_viewbox": round(moderate, 4),
        "not_extreme": round(not_extreme, 4),
        "spread": round(spread, 3),
    }


def _component_structure(elements: list[ET.Element]) -> tuple[float, dict[str, Any]]:
    tag_counts: dict[str, int] = {}
    for elem in elements:
        tag = _local_name(elem.tag)
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    primitive_count = sum(tag_counts.get(tag, 0) for tag in PRIMITIVE_TAGS)
    diversity = sum(1 for tag in PRIMITIVE_TAGS if tag_counts.get(tag, 0) > 0)
    count_score = _score_range(primitive_count, 3.0, 48.0, 0.0, 96.0)
    diversity_score = _score_range(diversity, 2.0, 5.0, 0.0, 8.0)
    defs_ok = 1.0
    if tag_counts.get("linearGradient", 0) or tag_counts.get("radialGradient", 0) or tag_counts.get("filter", 0):
        defs_ok = 1.0 if tag_counts.get("defs", 0) else 0.5
    score = statistics.mean([count_score, diversity_score, defs_ok])
    return score, {"primitive_count": primitive_count, "primitive_diversity": diversity, "tag_counts": tag_counts}


def _component_colors(elements: list[ET.Element]) -> tuple[float, dict[str, Any]]:
    values: list[str] = []
    invalid = 0
    for elem in elements:
        for key, value in elem.attrib.items():
            if _local_name(key) not in COLOR_ATTRS:
                continue
            clean = value.strip()
            if not clean:
                continue
            values.append(clean)
            lower = clean.lower()
            is_ref = lower.startswith("url(#") and lower.endswith(")")
            is_rgb = lower.startswith(("rgb(", "rgba(", "hsl(", "hsla("))
            is_hex = bool(HEX_RE.fullmatch(clean))
            is_named = clean in SAFE_COLOR_NAMES or lower in {name.lower() for name in SAFE_COLOR_NAMES}
            if not (is_ref or is_rgb or is_hex or is_named):
                invalid += 1

    hexes = [color.lower() for color in HEX_RE.findall(" ".join(values))]
    unique_hexes = sorted(set(hexes))
    palette_score = _score_range(len(unique_hexes), 2.0, 8.0, 0.0, 16.0)
    attr_score = _score_range(len(values), 3.0, 80.0, 0.0, 160.0)
    valid_score = 1.0 if not values else 1.0 - invalid / len(values)
    score = statistics.mean([palette_score, attr_score, valid_score])
    return score, {"color_attr_count": len(values), "unique_hex_colors": unique_hexes, "invalid_color_count": invalid}


def _prompt_terms(prompt: str) -> tuple[set[str], set[str], set[str]]:
    lower = (prompt or "").lower()
    shape_needs = {shape for shape, terms in SHAPE_TERMS.items() if any(term in lower for term in terms)}
    color_needs = {family for family, terms in COLOR_WORDS.items() if any(term in lower for term in terms)}
    hex_needs = {value.lower() for value in HEX_RE.findall(prompt or "")}
    return shape_needs, color_needs, hex_needs


def _component_grounding(elements: list[ET.Element], svg_text: str, prompt: str | None) -> tuple[float, dict[str, Any]]:
    if not prompt:
        return 1.0, {"used": False}

    shape_needs, color_needs, hex_needs = _prompt_terms(prompt)
    present_tags = {_local_name(elem.tag) for elem in elements}
    shape_hits = 0
    for shape in shape_needs:
        if shape in present_tags:
            shape_hits += 1
        elif shape == "rect" and any("rx" in elem.attrib for elem in elements):
            shape_hits += 1
        elif shape == "path" and "path" in present_tags:
            shape_hits += 1
    shape_score = 1.0 if not shape_needs else shape_hits / len(shape_needs)

    lower_svg = svg_text.lower()
    hex_hits = sum(1 for value in hex_needs if value in lower_svg)
    hex_score = 1.0 if not hex_needs else hex_hits / len(hex_needs)

    color_hits = 0
    for family in color_needs:
        if any(term in lower_svg for term in COLOR_WORDS[family]):
            color_hits += 1
    color_score = 1.0 if not color_needs else color_hits / len(color_needs)

    score = statistics.mean([shape_score, hex_score, color_score])
    return score, {
        "used": True,
        "shape_needs": sorted(shape_needs),
        "shape_score": round(shape_score, 4),
        "hex_needs": sorted(hex_needs),
        "hex_score": round(hex_score, 4),
        "color_needs": sorted(color_needs),
        "color_score": round(color_score, 4),
    }


def _component_degeneracy(text: str, elements: list[ET.Element]) -> tuple[float, dict[str, Any]]:
    length_score = _score_range(len(text), 160.0, 5000.0, 0.0, 9000.0)
    tokens = re.findall(r"[A-Za-z_#./:-]+|\d+(?:\.\d+)?", text)
    if len(tokens) < 20:
        token_score = 0.4
    else:
        token_score = _clamp01(len(set(tokens)) / len(tokens) / 0.35)
    path_ds = [elem.attrib.get("d", "") for elem in elements if _local_name(elem.tag) == "path" and elem.attrib.get("d")]
    repeated_paths = len(path_ds) - len(set(path_ds))
    repeat_score = _clamp01(1.0 - repeated_paths * 0.12)
    markdown_penalty = 0.6 if "```" in text else 1.0
    score = statistics.mean([length_score, token_score, repeat_score, markdown_penalty])
    return score, {
        "length": len(text),
        "token_unique_ratio": round(len(set(tokens)) / max(1, len(tokens)), 4),
        "repeated_paths": repeated_paths,
        "contains_markdown_fence": "```" in text,
    }


def score_svg(svg: str, prompt: str | None = None) -> dict[str, Any]:
    parsed = _parse_svg(extract_svg(svg))
    elements = _iter_elements(parsed.root)

    validity, validity_info = _component_validity(parsed)
    if parsed.root is None:
        components = {
            "validity": validity,
            "safety": 0.0,
            "geometry": 0.0,
            "structure": 0.0,
            "colors": 0.0,
            "grounding": 0.0 if prompt else 1.0,
            "degeneracy": 0.0,
        }
        total = 0.25 * validity
        return {"score": round(total, 6), "components": components, "details": {"validity": validity_info}}

    safety, safety_info = _component_safety(elements)
    geometry, geometry_info = _component_geometry(elements)
    structure, structure_info = _component_structure(elements)
    colors, colors_info = _component_colors(elements)
    grounding, grounding_info = _component_grounding(elements, parsed.text, prompt)
    degeneracy, degeneracy_info = _component_degeneracy(parsed.text, elements)

    components = {
        "validity": validity,
        "safety": safety,
        "geometry": geometry,
        "structure": structure,
        "colors": colors,
        "grounding": grounding,
        "degeneracy": degeneracy,
    }
    weights = {
        "validity": 0.25,
        "safety": 0.15,
        "geometry": 0.15,
        "structure": 0.15,
        "colors": 0.10,
        "grounding": 0.10,
        "degeneracy": 0.10,
    }
    total = sum(components[key] * weights[key] for key in weights)
    return {
        "score": round(_clamp01(total), 6),
        "components": {key: round(value, 6) for key, value in components.items()},
        "details": {
            "validity": validity_info,
            "safety": safety_info,
            "geometry": geometry_info,
            "structure": structure_info,
            "colors": colors_info,
            "grounding": grounding_info,
            "degeneracy": degeneracy_info,
        },
    }


def reward(svg: str, prompt: str | None = None) -> float:
    return float(score_svg(svg, prompt=prompt)["score"])

