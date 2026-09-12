"""Task: prepare-depths — convert EXR depth maps to PNG for nerfstudio/dn-splatter.

Reads a directory of z-depth EXR files, normalises each to uint16, and writes
them as .png into an output directory (typically <dataset>/depths/).  Output
filenames match the corresponding image stems so nerfstudio can pair them.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np

from task_interface.interface import (
    ComputeRequirement,
    ResolvedOutput,
    Stage,
    Task,
    TaskInput,
    TaskInvocation,
    TaskMetadata,
    TaskOutput,
    TaskParameter,
    TaskResult,
    TaskStatus,
)
from task_interface.logging import get_logger
from task_interface.registry import TaskRegistry


@TaskRegistry.register
class PrepareDepths(Task):
    metadata = TaskMetadata(
        name="prepare-depths",
        stage=Stage.PRE_PROCESSING,
        compute=ComputeRequirement.CPU,
        version=1,
        description=(
            "Convert a directory of z-depth EXR images to uint16 PNG files "
            "for use as depth supervision in nerfstudio / dn-splatter training."
        ),
    )

    inputs = [
        TaskInput(
            name="depths",
            type="directory",
            required=True,
            description="Directory containing z-depth EXR files",
        ),
    ]

    outputs = [
        TaskOutput(
            name="depths",
            type="directory",
            description="Directory of converted uint16 PNG depth maps",
        ),
    ]

    parameters = [
        TaskParameter(
            name="depth_scale",
            type="float",
            default=1000.0,
            description=(
                "Scale factor applied to metric depth values before uint16 "
                "conversion (default 1000 = millimetres, matching nerfstudio "
                "convention where 1 unit = 1 mm)"
            ),
        ),
    ]

    def execute(self, invocation: TaskInvocation) -> TaskResult:
        log = get_logger()
        params = invocation.parameter_values

        input_dir = Path(invocation.inputs[0].uri).resolve()
        output_dir = Path(invocation.outputs[0].uri).resolve()
        depth_scale = float(params.get("depth_scale", 1000.0))

        if not input_dir.is_dir():
            log.info(f"Depth directory not found, skipping: {input_dir}")
            return TaskResult(
                status=TaskStatus.SUCCESS,
                outputs_produced=[ResolvedOutput(name="depths", uri=str(output_dir))],
                metrics={"converted": 0, "total": 0, "action": "skipped_no_directory"},
            )

        exr_files = sorted(input_dir.glob("*.exr"))
        if not exr_files:
            log.info(f"No .exr files found in {input_dir}, skipping")
            return TaskResult(
                status=TaskStatus.SUCCESS,
                outputs_produced=[ResolvedOutput(name="depths", uri=str(output_dir))],
                metrics={"converted": 0, "total": 0, "action": "skipped_no_exr_files"},
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        converted = 0
        for exr_path in exr_files:
            depth = imageio.v3.imread(str(exr_path))
            if depth is None:
                log.warning(f"Could not read {exr_path.name}, skipping")
                continue

            # Use first channel if multi-channel
            if depth.ndim == 3:
                depth = depth[:, :, 0]

            depth_mm = (depth * depth_scale).clip(0, 65535).astype(np.uint16)
            out_path = output_dir / (exr_path.stem + ".png")
            cv2.imwrite(str(out_path), depth_mm)
            converted += 1

        log.info(f"Converted {converted}/{len(exr_files)} EXR depth maps to PNG")

        return TaskResult(
            status=TaskStatus.SUCCESS,
            outputs_produced=[ResolvedOutput(name="depths", uri=str(output_dir))],
            metrics={"converted": converted, "total": len(exr_files)},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert EXR depth maps to uint16 PNG for nerfstudio depth supervision")
    parser.add_argument("-i", "--input", required=True, help="Input directory containing .exr depth files")
    parser.add_argument("-o", "--output", required=True, help="Output directory for converted PNG depth maps")
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="Scale factor for depth values (default: 1000 = millimetres)")
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    exr_files = sorted(input_dir.glob("*.exr"))
    if not exr_files:
        print(f"No .exr files found in {input_dir}, nothing to convert")
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    for exr_path in exr_files:
        depth = imageio.v3.imread(str(exr_path))
        if depth is None:
            print(f"Warning: could not read {exr_path.name}, skipping", file=sys.stderr)
            continue
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depth_mm = (depth * args.depth_scale).clip(0, 65535).astype(np.uint16)
        cv2.imwrite(str(output_dir / (exr_path.stem + ".png")), depth_mm)
        converted += 1

    print(f"Converted {converted}/{len(exr_files)} EXR depth maps to PNG in {output_dir}")
    sys.exit(0)
