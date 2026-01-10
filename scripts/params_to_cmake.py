#!/usr/bin/env python3
"""
Parse runtime parameters from a YAML file and emit CMake-friendly compile definitions.
Usage:
  python scripts/params_to_cmake.py --params-file defn/params.yaml --format cmake-defs
Formats:
  cmake-defs : semicolon-separated NAME=VALUE entries safe for target_compile_definitions
  kv         : newline-separated NAME=VALUE entries
  json       : JSON object of key/value pairs
"""
import argparse
import json
import sys
from pathlib import Path

import yaml


def _format_def(name, value):
    if isinstance(value, bool):
        return f"{name}={'1' if value else '0'}"
    if isinstance(value, (int, float)):
        return f"{name}={value}"
    text = str(value)
    # Preserve existing quoting when provided in YAML
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return f"{name}={text}"
    escaped = text.replace('"', '\\"')
    return f'{name}="{escaped}"'


def parse_params(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Params file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    runtime_params = data.get("runtime_params", [])
    pairs = []
    for entry in runtime_params:
        if isinstance(entry, dict):
            for key, value in entry.items():
                pairs.append((str(key), value))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Emit CMake-ready compile definitions from a params YAML file")
    parser.add_argument("--params-file", default="defn/params.yaml", help="Path to params.yaml (default: defn/params.yaml)")
    parser.add_argument("--format", choices=["cmake-defs", "kv", "json"], default="cmake-defs", help="Output format")
    args = parser.parse_args()

    params_path = Path(args.params_file).expanduser().resolve()
    pairs = parse_params(params_path)

    if args.format == "cmake-defs":
        defs = [_format_def(k, v) for k, v in pairs]
        sys.stdout.write(";".join(defs))
    elif args.format == "kv":
        for k, v in pairs:
            print(f"{k}={v}")
    else:  # json
        obj = {k: v for k, v in pairs}
        json.dump(obj, sys.stdout)


if __name__ == "__main__":
    main()
