# Extensions: Inactivity, Zone, Session Summary

Built on top of the five activity labels and the existing **Fall Alert**. Same idea: event overlays for a monitoring demo — not new evaluate classes.

## 1. Inactivity alert

| Piece | Role |
|---|---|
| `src/inactivity_alert.py` | Person in view + low hip speed (or MOG2 ratio) for ~8 s → **INACTIVE** |
| Banner | Amber bar (only if no fall alert) |

**Demo:** Stand or sit still full-body ~10 s → amber **INACTIVE**; walk/wave → clears.

## 2. Zone / framing warning

| Piece | Role |
|---|---|
| `src/zone_alert.py` | **OUT OF VIEW** (no person) or **PARTIAL BODY** (`unknown` / incomplete pose) |
| Banner | Yellow bar (lowest priority) |

**Demo:** Leave the frame → OUT OF VIEW; sit at the desk face-only → PARTIAL BODY.

## 3. Session summary

| Piece | Role |
|---|---|
| `src/session_summary.py` | One-line blurb: mostly X · duration · fall/inactive/zone stats |
| Video Analyze | Headline + Processing Details |
| Live Camera | Updates while streaming; final blurb on **Stop camera** |

**Example:** `Mostly standing (70%) · 24s monitored · 1 inactive alert · 3s partial body`

## Banner priority

On the annotated frame, only **one** bottom banner at a time:

1. **FALL ALERT** (red)  
2. **INACTIVE** (amber)  
3. **ZONE** (yellow)

## Honest line for reviewers

> “These are lightweight monitoring extensions on our pose pipeline — heuristics for the demo, not medical or security products. Accuracy for the five activities is unchanged.”
