import json
from pathlib import Path
from typing import Dict, Any, List

def save_features_json(path: Path, columns: List[str], meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"columns": list(columns), **meta}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def load_features_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "columns" not in data or not isinstance(data["columns"], list):
        raise ValueError("features.json inválido: falta 'columns'")
    return data
