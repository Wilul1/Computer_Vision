"""Dedicated CareView Camera Station UI (live patient from roster).

Laptop entry point — owns the webcam via server-side OpenCV (not browser
getUserMedia). Does not show the caregiver dashboard.
"""
from __future__ import annotations

import html as html_lib

import gradio as gr

CAMERA_STATION_CSS = """
:host{display:block!important;width:100%!important;max-width:none!important;min-height:100vh!important;
  margin:0!important;background:
    radial-gradient(ellipse at 5% 0%,rgba(131,190,200,.32),transparent 34%),
    radial-gradient(ellipse at 96% 7%,rgba(231,190,145,.27),transparent 32%),
    linear-gradient(145deg,#E8F1EF 0%,#F7F5EF 48%,#EAF1EF 100%)!important}
:root,html,body,gradio-app,main,.main,#root,.gradio-container{
  color:#183A40!important;color-scheme:light!important;
  background:
    radial-gradient(ellipse at 5% 0%,rgba(131,190,200,.32),transparent 34%),
    radial-gradient(ellipse at 96% 7%,rgba(231,190,145,.27),transparent 32%),
    linear-gradient(145deg,#E8F1EF 0%,#F7F5EF 48%,#EAF1EF 100%)!important;
  background-attachment:fixed!important}
html,body{width:100%!important;max-width:none!important;min-height:100%!important;margin:0!important}
body{min-height:100vh!important}
gradio-app,main,.main,#root{display:block!important;width:100%!important;max-width:none!important;margin-left:0!important;margin-right:0!important}
:root{--background-fill-primary:rgba(255,255,255,.82)!important;
  --background-fill-secondary:rgba(248,250,248,.8)!important;--block-background-fill:rgba(255,255,255,.72)!important;
  --body-background-fill:transparent!important;--body-text-color:#183A40!important;
  --body-text-color-subdued:#668080!important;--input-background-fill:rgba(255,255,255,.88)!important;
  --input-border-color:#D4E2DD!important;--border-color-primary:#D4E2DD!important;--color-accent:#347C78!important;
  --neutral-950:#183A40!important;--neutral-900:#24494D!important;--neutral-800:#365D60!important;
  --neutral-700:#4E7071!important;--neutral-600:#668080!important;--neutral-500:#829797!important;
  --neutral-400:#B2C3BF!important;--neutral-300:#D4E2DD!important;--neutral-200:#E5EEEA!important;
  --neutral-100:#F2F6F3!important;--neutral-50:#FFFFFF!important}
.gradio-container{--background-fill-primary:rgba(255,255,255,.86)!important;
  --background-fill-secondary:rgba(248,250,248,.82)!important;--block-background-fill:rgba(255,255,255,.76)!important;
  --block-border-color:#D4E2DD!important;--body-background-fill:transparent!important;
  --body-text-color:#183A40!important;--body-text-color-subdued:#668080!important;
  --input-background-fill:#FFFFFF!important;--input-border-color:#D4E2DD!important;
  --border-color-primary:#D4E2DD!important;--color-accent:#347C78!important;
  max-width:none!important;width:100%!important;min-height:100vh!important;margin:0!important;
  padding:clamp(12px,2.5vw,30px)!important;box-sizing:border-box!important;
  font-family:"Segoe UI",ui-sans-serif,system-ui,sans-serif}
.gradio-container .wrap,.gradio-container .form,.gradio-container .block{color:#183A40!important}
.gradio-container label,.gradio-container .label-wrap,
.gradio-container .label-wrap span{color:#456668!important;opacity:1!important}
.gradio-container .prose,.gradio-container .prose p,.gradio-container .prose li,
.gradio-container .prose h1,.gradio-container .prose h2,.gradio-container .prose h3,
.gradio-container .prose h4,.gradio-container .prose strong,
.gradio-container .markdown-body,.gradio-container .markdown-body p,
.gradio-container .markdown-body li,.gradio-container .markdown-body h1,
.gradio-container .markdown-body h2,.gradio-container .markdown-body h3,
.gradio-container .markdown-body strong{color:#24494D!important;opacity:1!important}
.gradio-container .prose code,.gradio-container .markdown-body code{
  background:#E5EEEA!important;color:#24494D!important;border:1px solid #D4E2DD!important;
  border-radius:5px!important;padding:2px 5px!important;box-shadow:none!important}
.cs-wrap{font-family:"Segoe UI",ui-sans-serif,system-ui,sans-serif}
.cs-header{position:relative;isolation:isolate;overflow:hidden;min-height:220px;
  background:radial-gradient(ellipse at 92% 5%,rgba(255,255,255,.88),transparent 34%),
    radial-gradient(ellipse at 75% 110%,rgba(126,177,166,.28),transparent 43%),
    linear-gradient(112deg,rgba(255,255,255,.88),rgba(235,244,239,.77) 55%,rgba(232,239,234,.64))!important;
  border:1px solid rgba(255,255,255,.88);border-radius:28px;padding:25px 30px;margin-bottom:16px;
  box-shadow:0 18px 48px rgba(39,77,76,.11),inset 0 1px 0 rgba(255,255,255,.9);
  backdrop-filter:blur(18px)}
.cs-header:after{content:"";position:absolute;z-index:-1;width:220px;height:220px;right:8%;top:-120px;
  border:1px solid rgba(57,121,115,.22);border-radius:50%;
  box-shadow:0 0 0 24px rgba(57,121,115,.045),0 0 0 50px rgba(57,121,115,.03)}
.cs-kicker{font-size:10px;font-weight:800;letter-spacing:.18em;color:#3F716D;text-transform:uppercase}
.cs-title{position:relative;z-index:1;margin-top:14px;font-family:Georgia,"Times New Roman",serif;
  font-size:clamp(34px,5vw,54px);font-weight:500;letter-spacing:-.045em;line-height:1.05;color:#183C42}
.cs-sub{position:relative;z-index:1;margin-top:9px;font-size:14px;color:#5D7779}
.cs-pills{position:relative;z-index:1;display:flex;flex-wrap:wrap;gap:8px;margin-top:17px}
.cs-pill{display:inline-flex;align-items:center;gap:7px;padding:7px 12px;border-radius:999px;font-size:11px;
  font-weight:650;background:rgba(255,255,255,.58);color:#3C6665;border:1px solid rgba(89,139,133,.18);
  backdrop-filter:blur(10px)}
.cs-pill i{width:7px;height:7px;border-radius:50%;background:#8AA4A0;display:inline-block}
.cs-pill.on i{background:#4E9B7C;box-shadow:0 0 0 3px rgba(78,155,124,.12)}
.cs-note{margin:10px 0 17px;padding:14px 17px;border-radius:16px;
  background:rgba(255,255,255,.60);border:1px solid rgba(255,255,255,.86);color:#4D6C6D;
  font-size:12px;line-height:1.65;box-shadow:0 8px 22px rgba(39,77,76,.055);backdrop-filter:blur(12px)}
.cs-note b{color:#24494D}.cs-note code{padding:2px 5px;border-radius:5px;background:rgba(70,119,113,.09);color:#286965}
.cs-camera-control{margin:0 0 14px!important;padding:10px 14px!important;
  background:rgba(255,255,255,.72)!important;border:1px solid rgba(255,255,255,.92)!important;
  border-radius:16px!important;box-shadow:0 8px 22px rgba(39,77,76,.06)!important}
.cs-camera-control *{background-color:transparent!important;border-color:transparent!important;box-shadow:none!important}
.cs-camera-control .wrap,.cs-camera-control .form,.cs-camera-control .block{
  background:transparent!important;border-color:transparent!important}
.cs-panel{min-width:0!important;background:rgba(255,255,255,.70)!important;
  border:1px solid rgba(255,255,255,.9)!important;border-radius:21px!important;
  box-shadow:0 14px 38px rgba(39,77,76,.09),inset 0 1px 0 rgba(255,255,255,.88)!important;
  backdrop-filter:blur(16px)!important}
.cs-panel,.cs-image-preview,.cs-image-preview>div,.cs-image-preview .wrap,
.cs-image-preview .form,.cs-image-preview .block,.cs-image-preview .container,
.cs-image-preview .svelte-frame{width:100%!important;max-width:none!important;min-width:0!important}
.cs-panel .wrap,.cs-panel .form,.cs-panel .block{background:rgba(255,255,255,.45)!important;
  border-color:rgba(101,143,137,.16)!important;color:#183A40!important}
.cs-panel label,.gradio-container label{color:#456668!important;font-weight:650!important}
.cs-panel .image-container{overflow:hidden!important;border:1px solid rgba(101,143,137,.18)!important;
  border-radius:15px!important;background:#122D35!important}
.gradio-container input,.gradio-container textarea,.gradio-container select{
  background:rgba(255,255,255,.88)!important;color:#183A40!important;border:1px solid #D4E2DD!important;border-radius:11px!important}
.gradio-container input::placeholder,.gradio-container textarea::placeholder{color:#91A5A0!important}
.cs-camera-select,.cs-camera-select *{color:#183A40!important;opacity:1!important}
.cs-camera-select,.cs-camera-select>div,.cs-camera-select .wrap,.cs-camera-select .form,
.cs-camera-select .block,.cs-camera-select .container,.cs-camera-select .svelte-frame{
  background:transparent!important;border-color:transparent!important;box-shadow:none!important}
.cs-camera-select{width:100%!important;max-width:380px!important}
.cs-camera-select>div{width:100%!important;max-width:380px!important}
.cs-camera-select input,.cs-camera-select [role="combobox"],.cs-camera-select button{
  color:#183A40!important;background:#FFFFFF!important;border:1px solid #C9DDD6!important;
  border-radius:11px!important;opacity:1!important;min-height:40px!important}
.cs-camera-select label,.cs-camera-select .label-wrap,.cs-camera-select .label-wrap span{
  color:#365D60!important;background:transparent!important;font-size:12px!important;
  font-weight:700!important;opacity:1!important}
.cs-camera-select svg{color:#456668!important;fill:#456668!important}
.cs-image-preview .wrap,.cs-image-preview .form,.cs-image-preview .block,
.cs-image-preview .label-wrap,.cs-image-preview label,.cs-image-preview button,
.cs-image-preview [data-testid="block-label"],.cs-image-preview [class*="label-wrap"]{
  background:transparent!important;border-color:transparent!important;box-shadow:none!important}
.cs-image-preview .label-wrap,.cs-image-preview .label-wrap span,.cs-image-preview label{
  color:#365D60!important;font-size:13px!important;font-weight:700!important;opacity:1!important}
.cs-image-preview .image-container,.cs-image-preview [class*="image-container"],
.cs-image-preview [class*="image-frame"],.cs-image-preview [data-testid*="image"]{
  background:#EAF2EE!important;border:1px solid #D4E2DD!important;border-radius:14px!important}
.cs-image-preview .image-container,.cs-image-preview [class*="image-container"]{
  display:flex!important;align-items:center!important;justify-content:center!important;
  width:min(100%,420px)!important;max-width:420px!important;height:auto!important;min-height:0!important;
  aspect-ratio:1/1!important;margin-left:auto!important;margin-right:auto!important;overflow:hidden!important}
.cs-image-preview .image-container img,.cs-image-preview [class*="image-container"] img,
.cs-image-preview .image-container video,.cs-image-preview [class*="image-container"] video{
  display:block!important;width:100%!important;height:100%!important;max-width:none!important;max-height:none!important;
  object-fit:contain!important;object-position:center!important}
.gradio-container table{background:rgba(255,255,255,.72)!important;color:#365D60!important;border-radius:12px!important}
.gradio-container th{background:rgba(224,238,232,.82)!important;color:#365D60!important}
.gradio-container td{border-color:rgba(72,116,110,.12)!important}
.cs-actions{gap:10px!important;margin:2px 0 6px!important}
.cs-actions button{min-height:48px!important;border-radius:13px!important;font-weight:700!important;
  box-shadow:0 8px 18px rgba(41,102,96,.12);transition:transform .15s ease,filter .15s ease!important}
.cs-actions button:hover{transform:translateY(-1px);filter:brightness(1.04)}
.cs-actions .primary,.cs-actions button.primary{background:#347C78!important;border-color:#347C78!important;color:#fff!important}
.cs-actions button:not(.primary){background:rgba(255,255,255,.78)!important;border:1px solid #CDDFD9!important;color:#35605F!important}
.cs-wrap~.markdown,.cs-wrap~.prose{color:#365D60!important}
.cs-output{min-width:0!important;padding:12px 15px!important;margin:8px 0!important;
  border:1px solid rgba(255,255,255,.9)!important;border-radius:15px!important;
  background:rgba(255,255,255,.72)!important;box-shadow:0 8px 22px rgba(39,77,76,.06)!important;
  color:#24494D!important;opacity:1!important;backdrop-filter:blur(12px)}
.cs-output .prose,.cs-output .markdown-body{--body-text-color:#24494D!important;
  --body-text-color-subdued:#668080!important;color:#24494D!important;opacity:1!important}
.cs-output .prose p,.cs-output .prose li,.cs-output .prose h1,.cs-output .prose h2,
.cs-output .prose h3,.cs-output .prose h4,.cs-output .prose strong,
.cs-output .markdown-body p,.cs-output .markdown-body li,.cs-output .markdown-body h1,
.cs-output .markdown-body h2,.cs-output .markdown-body h3,.cs-output .markdown-body h4,
.cs-output .markdown-body strong{color:#24494D!important;opacity:1!important}
.cs-output .prose h1,.cs-output .prose h2,.cs-output .prose h3,
.cs-output .markdown-body h1,.cs-output .markdown-body h2,.cs-output .markdown-body h3{
  color:#183A40!important;font-family:Georgia,"Times New Roman",serif!important;font-weight:500!important}
.cs-output .prose code,.cs-output .markdown-body code{background:#E5EEEA!important;color:#24494D!important;
  border:1px solid #D4E2DD!important;border-radius:5px!important;padding:2px 5px!important;box-shadow:none!important}
.cs-headline{padding:7px 13px!important;background:rgba(255,255,255,.58)!important;box-shadow:none!important}
.cs-headline h1,.cs-headline h2,.cs-headline h3,.cs-headline .prose h1,
.cs-headline .prose h2,.cs-headline .prose h3{font-family:"Segoe UI",ui-sans-serif,system-ui,sans-serif!important;
  color:#21494C!important;font-size:15px!important;font-weight:700!important;letter-spacing:0!important}
.cs-events label,.cs-events .label-wrap,.cs-events .label-wrap span,
.cs-events .wrap,.cs-events .wrap span{color:#456668!important;opacity:1!important}
.cs-events .table-wrap,.cs-events table{background:rgba(255,255,255,.82)!important}
#cs-session-events,#cs-session-events *{color:#24494D!important;opacity:1!important;
  font-family:"Segoe UI",ui-sans-serif,system-ui,sans-serif!important}
#cs-session-events{--body-text-color:#24494D!important;--body-text-color-subdued:#456668!important;
  --block-label-text-color:#24494D!important;--block-title-text-color:#183A40!important}
#cs-session-events label,#cs-session-events .label-wrap,#cs-session-events .label-wrap *{
  color:#24494D!important;background:transparent!important;font-size:13px!important;font-weight:700!important}
#cs-session-events table,#cs-session-events .table-wrap{background:#FFFFFF!important;color:#24494D!important}
#cs-session-events th,#cs-session-events [role="columnheader"]{
  color:#183A40!important;background:#DCEBE6!important;text-transform:capitalize;
  font-size:12px!important;font-weight:700!important;opacity:1!important}
#cs-session-events td,#cs-session-events [role="gridcell"]{
  color:#24494D!important;background:#FFFFFF!important;font-size:12px!important;opacity:1!important}
.cs-note,.cs-sub,.cs-footer,.cs-pill{opacity:1!important}
.cs-footer{margin:19px 0 5px;font-size:11px;color:#527371;text-align:center;letter-spacing:.02em}
.cs-err{margin:8px 0 0;padding:11px 13px;border-radius:12px;background:#FEF2F2;color:#B91C1C;
  border:1px solid #FECACA;font-size:13px;display:none}
@media(max-width:700px){
  .gradio-container{padding:11px!important}
  .cs-header{min-height:0;padding:21px 18px;border-radius:22px}
  .cs-header:after{right:-8%;top:-133px}
  .cs-title{font-size:34px}
  .cs-sub{font-size:12px;line-height:1.5}
  .cs-pills{gap:6px;margin-top:14px}
  .cs-pill{font-size:10px;padding:6px 9px}
  .cs-panel{border-radius:17px!important}
}
"""


def camera_station_header_html(patient_label: str) -> str:
    """Header block showing the live roster patient (Name · Room)."""
    label = html_lib.escape((patient_label or "").strip() or "No Live Patient Assigned")
    return f"""
            <div class="cs-wrap">
              <div class="cs-header">
                <div class="cs-kicker">CareView · Health &amp; Safety</div>
                <div class="cs-title">{label}</div>
                <div class="cs-sub">A Calm, Attentive View For This Room · Local Demo Monitoring, Not A Clinical Device</div>
                <div class="cs-pills">
                  <span class="cs-pill on"><i></i> Camera Stays On This Station</span>
                  <span class="cs-pill"><i></i> Shared Live Room View</span>
                  <span class="cs-pill"><i></i> Caregiver View-Only</span>
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
    label = html_lib.escape((patient_label or "").strip() or "No Live Patient Assigned")
    return (
        f'<div class="cs-footer">CareView Health &amp; Safety · {label} · '
        "Works On Localhost And LAN HTTP Because Capture Is Server-Side</div>"
    )


def build_camera_station_ui(
    station_start,
    station_stop,
    station_tick,
    *,
    station_preview_tick=None,
    initial_patient_label: str = "No Live Patient Assigned",
):
    """Patient-room interface: server-side Start/Stop for the live webcam patient."""
    with gr.Blocks(title="CareView Camera Station") as demo:
        wtitle = gr.HTML(camera_station_header_html(initial_patient_label))
        with gr.Group(elem_classes=["cs-camera-control"]):
            cam_index = gr.Dropdown(
                label="Camera Source Index",
                choices=[0, 1, 2],
                value=0,
                interactive=True,
                elem_classes=["cs-camera-select"],
                elem_id="cs-camera-source",
            )
        with gr.Row():
            with gr.Column(scale=1, min_width=300, elem_classes=["cs-panel"]):
                win = gr.Image(
                    label="Camera Preview",
                    type="numpy",
                    height=420,
                    interactive=False,
                    elem_classes=["cs-image-preview"],
                )
            with gr.Column(scale=1, min_width=300, elem_classes=["cs-panel"]):
                wout = gr.Image(
                    label="Annotated Live View",
                    type="numpy",
                    height=420,
                    interactive=False,
                    elem_classes=["cs-image-preview"],
                )
        with gr.Row(elem_classes=["cs-actions"]):
            wstart = gr.Button("Start Monitoring", variant="primary", scale=1)
            wstop = gr.Button("Stop Monitoring", scale=1)
        whead = gr.Markdown(
            "Ready. Press **Start Monitoring** To Open The Laptop Webcam.",
            elem_classes=["cs-output", "cs-headline"],
        )
        wmon = gr.Markdown(
            "Monitoring Status Will Appear While Streaming.",
            elem_classes=["cs-output", "cs-monitor-output"],
        )
        wsum = gr.Markdown(
            "Session Elapsed Time And Activity Summary Will Appear Here.",
            elem_classes=["cs-output", "cs-summary-output"],
        )
        werr = gr.Markdown("", elem_classes=["cs-output", "cs-error-output"])
        wevents = gr.Dataframe(
            label=f"Session Events ({initial_patient_label})",
            headers=["ID", "Time", "Type", "Status", "Action", "Snapshot"],
            interactive=False,
            elem_classes=["cs-events"],
            elem_id="cs-session-events",
        )
        wfooter = gr.HTML(camera_station_footer_html(initial_patient_label))
        # title + footer refresh with patient label; events label updates via gr.update
        outs = [wtitle, win, wout, whead, wmon, wsum, werr, wevents, wfooter]
        status_outs = [wtitle, wout, whead, wmon, wsum, werr, wevents, wfooter]
        wstart.click(station_start, [cam_index], outs)
        wstop.click(station_stop, None, outs)
        # Fast poll; station_tick still skips Gradio update when frame seq unchanged.
        timer = gr.Timer(0.2, active=True)
        timer.tick(station_tick, None, status_outs, show_progress="hidden")
        if station_preview_tick is not None:
            preview_timer = gr.Timer(0.1, active=True)
            preview_timer.tick(station_preview_tick, None, win, show_progress="hidden")
        demo.load(station_tick, None, status_outs, show_progress="hidden")
    return demo
