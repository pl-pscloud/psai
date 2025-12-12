import os
import json
import shutil
import cloudpickle as pickle
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ExperimentManager:
    def __init__(self, base_dir: str = "psai/experiments"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all saved experiments with metadata."""
        experiments = []
        if not os.path.exists(self.base_dir):
            return []

        for name in os.listdir(self.base_dir):
            exp_path = os.path.join(self.base_dir, name)
            if os.path.isdir(exp_path):
                # Try to read metadata if it exists, or infer from file stats
                meta = {
                    "name": name,
                    "created_at": datetime.fromtimestamp(os.path.getctime(exp_path)).isoformat(),
                    "files": os.listdir(exp_path)
                }
                experiments.append(meta)
        
        # Sort by creation time desc
        return sorted(experiments, key=lambda x: x["created_at"], reverse=True)

    def save_experiment(self, name: str, config: Dict[str, Any], psml_instance: Any = None, feature_code: str = None) -> Dict[str, str]:
        """Save the current experiment state."""
        # Sanitize name
        safe_name = "".join([c for c in name if c.isalnum() or c in ('-', '_')]).strip()
        if not safe_name:
            raise ValueError("Invalid experiment name")

        exp_path = os.path.join(self.base_dir, safe_name)
        os.makedirs(exp_path, exist_ok=True)

        # 1. Save Config
        with open(os.path.join(exp_path, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        # 2. Save Feature Engineering Code
        if feature_code:
            with open(os.path.join(exp_path, "feature_transformer.py"), "w") as f:
                f.write(feature_code)

        # 3. Save PSML Model (Pickle)
        # Note: psML objects might be large or have pickling issues (as seen before).
        # We assume the user has fixed pickling issues or we catch error.
        saved_items = ["config.json"]
        if psml_instance:
            try:
                model_path = os.path.join(exp_path, "model.pkl")
                with open(model_path, "wb") as f:
                    pickle.dump(psml_instance, f)
                saved_items.append("model.pkl")
            except Exception as e:
                logger.error(f"Failed to pickle psML instance: {e}")
                # We don't fail the whole save, but we note it
                pass

        return {"name": safe_name, "path": exp_path, "saved_items": saved_items}

    def load_experiment(self, name: str) -> Dict[str, Any]:
        """Load experiment artifacts."""
        safe_name = "".join([c for c in name if c.isalnum() or c in ('-', '_')]).strip()
        exp_path = os.path.join(self.base_dir, safe_name)
        
        if not os.path.exists(exp_path):
            raise FileNotFoundError(f"Experiment {safe_name} not found")

        result = {}

        # 1. Load Config
        config_path = os.path.join(exp_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                result["config"] = json.load(f)

        # 2. Load Code
        code_path = os.path.join(exp_path, "feature_transformer.py")
        if os.path.exists(code_path):
            with open(code_path, "r") as f:
                result["feature_code"] = f.read()

        # 3. Load Model path (we don't unpickle here, we return path for the API to handle)
        model_path = os.path.join(exp_path, "model.pkl")
        if os.path.exists(model_path):
            result["model_path"] = model_path
        
        return result

    def get_model_path(self, name: str) -> str:
        safe_name = "".join([c for c in name if c.isalnum() or c in ('-', '_')]).strip()
        return os.path.join(self.base_dir, safe_name, "model.pkl")
