import argparse
import asyncio
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
REPO_ROOT = Path("/data/home/yxzhou/jydai")
IDF_ROOT = REPO_ROOT / "work" / "IDF_T"
VERL_ROOT = REPO_ROOT / "verl"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _discover_images(sample_dir: Path) -> list[Path]:
    images = [p for p in sorted(sample_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return images


def _import_class(class_path: str):
    # Example: tools.TF.SRM_Filter.SRMFilterTool
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return {"raw_text": text}


async def _run_one_tool_on_one_image(
    tool_obj: Any,
    tool_name: str,
    image_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "tool": tool_name,
        "image": str(image_path),
        "ok": False,
    }

    # Each tool writes its own artifacts under this folder.
    tool_artifact_dir = output_dir / "artifacts" / tool_name / image_path.stem
    tool_artifact_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as img:
        width, height = img.size

    parameters = {
        "image_input": str(image_path),
        "image_path": str(image_path),
        "output_dir": str(tool_artifact_dir),
    }
    # Provide minimal required params for pipeline/helper tools.
    if tool_name == "roi_extractor":
        parameters.update(
            {
                "x": 0,
                "y": 0,
                "w": width,
                "h": height,
                "coord_type": "abs",
                "coordinate_space": "original",
            }
        )
    elif tool_name == "image_resizer":
        parameters.update(
            {
                "scale_factor": 0.75,
                "keep_aspect_ratio": True,
                "interpolation": "nearest",
                "prefer_forensics_mode": "pixel_inspection",
            }
        )

    instance_id = None
    try:
        instance_id, _ = await tool_obj.create(image_input=str(image_path), image_path=str(image_path))
        response, score, extra = await tool_obj.execute(instance_id, parameters=parameters)

        response_payload = None
        if hasattr(response, "text") and isinstance(response.text, str):
            response_payload = _safe_json_loads(response.text)
        else:
            response_payload = str(response)

        tool_success = True
        if isinstance(response_payload, dict):
            tool_success = bool(response_payload.get("success", True))

        result.update(
            {
                "ok": tool_success,
                "score": score,
                "extra": extra,
                "response": response_payload,
            }
        )
    except Exception as exc:
        result.update(
            {
                "ok": False,
                "error": str(exc),
            }
        )
    finally:
        if instance_id is not None:
            try:
                await tool_obj.release(instance_id)
            except Exception:
                pass

    return result


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run IDF_T tools on sample images and save JSON outputs.")
    parser.add_argument(
        "--sample-dir",
        default="/data/home/yxzhou/jydai/work/IDF_T/sample",
        help="Directory containing sample images.",
    )
    parser.add_argument(
        "--output-root",
        default="/data/home/yxzhou/jydai/work/IDF_T/test",
        help="Directory to save test outputs.",
    )
    parser.add_argument(
        "--tool-config",
        default="/data/home/yxzhou/jydai/work/IDF_T/tools/verl_tool_config.yaml",
        help="Tool config YAML path.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=-1,
        help="Optional cap on number of images, -1 for all.",
    )
    parser.add_argument(
        "--tools",
        default="",
        help="Comma-separated tool function names to run, e.g. srm_filter,noise_analyzer. Empty means all.",
    )
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    output_root = Path(args.output_root)
    tool_config = Path(args.tool_config)

    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
    if not tool_config.exists():
        raise FileNotFoundError(f"Tool config not found: {tool_config}")

    # Ensure local modules are importable without editable install.
    for p in (str(REPO_ROOT), str(VERL_ROOT), str(IDF_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)

    images = _discover_images(sample_dir)
    if args.max_images > 0:
        images = images[: args.max_images]

    if not images:
        raise RuntimeError(f"No image files found in {sample_dir}")

    config = _load_yaml(tool_config)
    tool_entries = config.get("tools", [])
    if not isinstance(tool_entries, list) or not tool_entries:
        raise RuntimeError("No tools found in tool config")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    selected_names = {x.strip() for x in args.tools.split(",") if x.strip()} if args.tools else set()

    instantiated_tools: list[tuple[str, Any, dict[str, Any]]] = []
    for item in tool_entries:
        class_path = item.get("class_name")
        cfg = item.get("config", {})
        if not class_path:
            continue
        cls = _import_class(class_path)
        tool_obj = cls(cfg, None)

        schema_name = None
        try:
            schema = tool_obj.get_openai_tool_schema()
            schema_name = schema.function.name
        except Exception:
            schema_name = class_path.rsplit(".", 1)[-1]

        if selected_names and schema_name not in selected_names:
            continue
        instantiated_tools.append((schema_name, tool_obj, {"class_name": class_path, "config": cfg}))

    if not instantiated_tools:
        raise RuntimeError("No tools selected to run. Check --tools names.")

    summary: dict[str, Any] = {
        "sample_dir": str(sample_dir),
        "output_dir": str(run_dir),
        "images": [str(p) for p in images],
        "tools": [t[0] for t in instantiated_tools],
        "results": {},
    }

    for image_path in images:
        image_key = image_path.name
        summary["results"][image_key] = {}
        image_out_dir = run_dir / image_path.stem
        image_out_dir.mkdir(parents=True, exist_ok=True)

        for tool_name, tool_obj, tool_meta in instantiated_tools:
            one = await _run_one_tool_on_one_image(tool_obj, tool_name, image_path, image_out_dir)
            one["tool_meta"] = tool_meta
            summary["results"][image_key][tool_name] = one

            out_file = image_out_dir / f"{tool_name}.json"
            with out_file.open("w", encoding="utf-8") as f:
                json.dump(one, f, ensure_ascii=False, indent=2)

    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] Done. Output dir: {run_dir}")
    print(f"[OK] Summary: {summary_path}")
    print(f"[INFO] Images: {len(images)}")
    print(f"[INFO] Tools : {len(instantiated_tools)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
