"""Care Home caregiver demo login + patient roster facade.

Patients are no longer hardcoded — they come from PatientRoster (SQLite).
Demo login remains caregiver / care123.
"""
from __future__ import annotations

from pathlib import Path

from .patient_roster import get_roster

# Demo login only — not secure authentication.
DEMO_CAREGIVER = {
    "id": "cg1",
    "username": "caregiver",
    "password": "care123",
    "name": "caregiver1",
}


def get_caregiver(username: str, password: str) -> dict | None:
    if username == DEMO_CAREGIVER["username"] and password == DEMO_CAREGIVER["password"]:
        return {k: v for k, v in DEMO_CAREGIVER.items() if k != "password"}
    return None


def patients_for(caregiver_id: str) -> list[dict]:
    return get_roster().patients_for(caregiver_id)


def get_patient(patient_id: str) -> dict | None:
    return get_roster().get(patient_id)


def get_live_patient(caregiver_id: str = "cg1") -> dict | None:
    return get_roster().get_live_patient(caregiver_id)


def live_camera_label(caregiver_id: str = "cg1", *, empty: str = "No live patient assigned") -> str:
    return get_roster().live_camera_label(caregiver_id, empty=empty)


def add_patient(
    *,
    name: str,
    room: str,
    caregiver_id: str = "cg1",
    source_type: str = "none",
    video_path: str | None = None,
    note: str = "",
    patient_id: str | None = None,
) -> dict:
    return get_roster().add_patient(
        name=name,
        room=room,
        caregiver_id=caregiver_id,
        source_type=source_type,
        video_path=video_path,
        note=note,
        patient_id=patient_id,
    )


def seed_careview_pack(caregiver_id: str = "cg1") -> list[dict]:
    """Replace roster with Alex (live) + Casey (ready fall demo clip)."""
    return get_roster().seed_careview_pack(caregiver_id=caregiver_id)


def resolve_video_path(patient: dict) -> Path | None:
    raw = patient.get("video_path")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        from . import config

        path = config.ROOT / path
    return path if path.exists() else None
