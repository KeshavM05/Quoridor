"""
Training dashboard API.
Serves metrics from the training runs directory so the frontend can display them.
"""

import os
import json
from fastapi import APIRouter
from typing import Optional

router = APIRouter(prefix="/dashboard")

METRICS_FILE = os.path.join(os.path.dirname(__file__), 'checkpoints', 'metrics.json')


def save_metrics(metrics):
    """Append metrics to the JSON file."""
    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
    existing = load_metrics()
    existing['iterations'].append(metrics)
    with open(METRICS_FILE, 'w') as f:
        json.dump(existing, f, indent=2)


def load_metrics():
    """Load all metrics from file."""
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f:
            return json.load(f)
    return {'iterations': []}


@router.get("/metrics")
def get_metrics():
    """Return all training metrics."""
    return load_metrics()


@router.get("/status")
def get_status():
    """Return current training status."""
    metrics = load_metrics()
    iterations = metrics['iterations']

    if not iterations:
        return {
            "status": "idle",
            "total_iterations": 0,
            "total_positions": 0,
            "latest_win_rate": None,
            "latest_policy_loss": None,
            "latest_value_loss": None,
        }

    latest = iterations[-1]
    return {
        "status": "trained",
        "total_iterations": len(iterations),
        "total_positions": latest.get('total_positions', 0),
        "latest_win_rate": latest.get('win_rate', None),
        "latest_policy_loss": latest.get('policy_loss', None),
        "latest_value_loss": latest.get('value_loss', None),
        "latest_avg_game_length": latest.get('avg_game_length', None),
        "model_accepted": latest.get('model_accepted', None),
    }
