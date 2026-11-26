"""Dedicated CareView Camera Station UI (live patient from roster).

Laptop entry point — owns the webcam via server-side OpenCV (not browser
getUserMedia). Does not show the caregiver dashboard.
"""
from __future__ import annotations

import html as html_lib

import gradio as gr

CAMERA_STATION_CSS = """
html,body,.gradio-container{background:#0F172A!important;color:#E2E8F0!important}
.gradio-container{max-width:1100px!important;margin:0 auto!important;padding:16px!important}
.cs-wrap{font-family:"Segoe UI",ui-sans-serif,system-ui,sans-serif}
.cs-header{background:#15233F;border:1px solid #1E3A5F;border-radius:18px;padding:18px 20px;margin-bottom:14px}
.cs-kicker{font-size:11px;font-weight:700;letter-spacing:.08em;color:#93C5FD;text-transform:uppercase}
.cs-title{margin-top:6px;font-size:26px;font-weight:750;color:#F8FAFC}
.cs-sub{margin-top:6px;font-size:14px;color:#94A3B8}
.cs-pills{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
.cs-pill{display:inline-flex;align-items:center;gap:6px;padding:6px 12px;border-radius:999px;font-size:12px;font-weight:650;background:#1E293B;color:#E2E8F0;border:1px solid #334155}
.cs-pill i{width:8px;height:8px;border-radius:50%;background:#64748B;display:inline-block}
.cs-pill.on i{background:#22C55E}
.cs-note{margin:10px 0 14px;padding:12px 14px;border-radius:12px;background:#1E293B;border:1px solid #334155;color:#CBD5E1;font-size:13px;line-height:1.5}
.cs-panel{background:#111827!important;border:1px solid #1F2937!important;border-radius:16px!important}
.cs-panel label{color:#CBD5E1!important}
.cs-actions button{min-height:48px!important;border-radius:12px!important;font-weight:700!important}
.cs-actions .primary,.cs-actions button.primary{background:#2563EB!important;border-color:#2563EB!important;color:#fff!important}
.cs-footer{margin-top:16px;font-size:12px;color:#64748B;text-align:center}
.cs-err{margin:8px 0 0;padding:10px 12px;border-radius:10px;background:#7F1D1D;color:#FEE2E2;font-size:13px;display:none}
@media(max-width:700px){
  .gradio-container{padding:10px!important}
  .cs-title{font-size:22px}
}
"""


def camera_station_header_html(patient_label: str) -> str:
    """Header block showing the live roster patient (Name · Room)."""
    label = html_lib.escape((patient_label or "").strip() or "No live patient assigned")
    return f"""
            <div class="cs-wrap">
              <div class="cs-header">
                <div class="cs-kicker">CareView Camera Station</div>
                <div class="cs-title">{label}</div>
                <div class="cs-sub">Laptop webcam via OpenCV · local demo station (not the caregiver app)</div>
                <div class="cs-pills">
                  <span class="cs-pill on"><i></i> Server owns the camera</span>
                  <span class="cs-pill"><i></i> Publishes to LiveMonitorHub</span>
                  <span class="cs-pill"><i></i> Caregivers view-only on /caregiver</span>
                </div>
              </div>
              <div class="cs-note">
                This page does <b>not</b> use the browser webcam API (which fails on
                <code>http://192.168.x.x</code> without HTTPS). The Python process on this
                laptop opens the physical camera, runs AI, and records a raw session video
                under <code>outputs/recordings/</code>. Caregiver phones open <b>/caregiver</b>
                and never request webcam access. Recording failures do not stop monitoring.
                Assign the live patient in the caregiver app (Add patient → laptop camera).
              </div>
            </div>
            """


def camera_station_footer_html(patient_label: str) -> str:
    label = html_lib.escape((patient_label or "").strip() or "No live patient assigned")
    return (
        f'<div class="cs-footer">Demo Camera Station · {label} · '
        "works on localhost and LAN HTTP because capture is server-side</div>"
    )


def build_camera_station_ui(
    station_start,
    station_stop,
    station_tick,
    *,
    initial_patient_label: str = "No live patient assigned",
):
    """Patient-room interface: server-side Start/Stop for the live webcam patient."""
    with gr.Blocks(title="CareView Camera Station") as demo:
        wtitle = gr.HTML(camera_station_header_html(initial_patient_label))
        cam_index = gr.Dropdown(
            label="Camera source index",
            choices=[0, 1, 2],
            value=0,
            interactive=True,
        )
        with gr.Row():
            with gr.Column(scale=1, min_width=300, elem_classes=["cs-panel"]):
                win = gr.Image(
                    label="Camera Preview",
                    type="numpy",
                    height=420,
                    interactive=False,
                )
            with gr.Column(scale=1, min_width=300, elem_classes=["cs-panel"]):
                wout = gr.Image(
                    label="Annotated Live View",
                    type="numpy",
                    height=420,
                    interactive=False,
                )
        with gr.Row(elem_classes=["cs-actions"]):
            wstart = gr.Button("Start Monitoring", variant="primary", scale=1)
            wstop = gr.Button("Stop Monitoring", scale=1)
        whead = gr.Markdown("Ready. Press **Start Monitoring** to open the laptop webcam.")
        wmon = gr.Markdown("Monitoring status will appear while streaming.")
        wsum = gr.Markdown("Session elapsed time and activity summary appear here.")
        werr = gr.Markdown("")
        wevents = gr.Dataframe(
            label=f"Session events ({initial_patient_label})",
            headers=["id", "time", "type", "status", "action", "snapshot"],
            interactive=False,
        )
        wfooter = gr.HTML(camera_station_footer_html(initial_patient_label))
        # title + footer refresh with patient label; events label updates via gr.update
        outs = [wtitle, win, wout, whead, wmon, wsum, werr, wevents, wfooter]
        wstart.click(station_start, [cam_index], outs)
        wstop.click(station_stop, None, outs)
        # 0.5s + server-side skip when frame unchanged (avoids Gradio re-encode thrash).
        timer = gr.Timer(0.5, active=True)
        timer.tick(station_tick, None, outs)
        demo.load(station_tick, None, outs)
    return demo
