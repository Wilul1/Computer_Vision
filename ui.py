"""Gradio layout and visual styling for the MotionSense activity monitor."""

import gradio as gr


APP_CSS = """
:root{color-scheme:dark}
html,body,.gradio-container{background:#0b0c0e!important;background-attachment:fixed!important;color:#f3f2ef!important}
body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;background-attachment:fixed!important}
.gradio-container{max-width:1680px!important;width:100%!important;min-height:100vh;margin:0 auto!important;padding:0!important}
.gradio-container *{box-sizing:border-box}
#app-shell{display:flex!important;flex-wrap:nowrap!important;gap:16px;align-items:stretch;min-height:100vh;padding:14px;margin:0!important}
#sidebar{flex:0 0 232px;width:232px;height:calc(100vh - 28px);height:calc(100dvh - 28px);max-height:calc(100vh - 28px);max-height:calc(100dvh - 28px);min-height:0;overflow-y:auto;overflow-x:hidden;overscroll-behavior-y:contain;scrollbar-width:thin;scrollbar-color:#49473a transparent;position:sticky;top:14px;align-self:flex-start;padding:19px 13px;border:1px solid #292821;border-radius:17px;background:linear-gradient(180deg,#121311,#0e0f10)}
#main-panel{flex:1;min-width:0;min-height:100vh;overflow:visible;padding:2px 10px 22px;gap:16px}
.brand{display:flex;align-items:center;gap:11px;padding:2px 8px 24px;color:#f5f2e9;font-size:14px;font-weight:750;letter-spacing:1.3px}
.brand-mark{width:34px;height:34px;display:grid;place-items:center;border:1px solid #78633a;border-radius:10px;background:linear-gradient(145deg,#d8bd78,#8a6a32);color:#17150f;font-size:18px}
.brand span{color:#d9bd78}.brand small{display:block;color:#81817d;font-size:9px;font-weight:600;letter-spacing:1.8px;margin-top:3px}
.nav-caption{padding:0 11px 9px;color:#74756f;font-size:9px;font-weight:750;letter-spacing:1.7px}
#sidebar-nav fieldset{display:flex!important;flex-direction:column!important;gap:5px;width:100%!important;min-width:0;border:0!important;padding:0!important;margin:0!important}
#sidebar-nav label{width:100%!important;min-height:42px;display:flex!important;align-items:center;justify-content:flex-start;white-space:nowrap;overflow:hidden;padding:0 11px;border:1px solid transparent;border-radius:9px;color:#a7a7a1;cursor:pointer;box-sizing:border-box!important;transition:background .16s ease,border-color .16s ease,color .16s ease}
#sidebar-nav label:hover{background:#1b1b19;color:#f5f2e9}
#sidebar-nav label:has(input:checked){background:linear-gradient(90deg,#30291a,#211e17);border-color:#50442a;color:#e5ca83}
#sidebar-nav label input{width:0;height:0;margin:0;opacity:0;position:absolute}
#sidebar-nav label span{font-size:12px;line-height:1.2;width:100%}
#sidebar::-webkit-scrollbar{width:6px}
#sidebar::-webkit-scrollbar-track{background:transparent}
#sidebar::-webkit-scrollbar-thumb{background:#49473a;border-radius:8px}
.sidebar-tip{margin:25px 2px 0;padding:15px 13px;min-height:112px;border:1px solid #302f2a;border-radius:12px;background:#151513}
.sidebar-tip-heading{display:flex;align-items:center;gap:9px;margin:2px 0 9px}.sidebar-tip .tip-icon{color:#d9bd78;font-size:14px;line-height:1}.sidebar-tip strong{display:block;margin:0;color:#f1efe9;font-size:11px;font-weight:650}.sidebar-tip p{margin:0;color:#a5a49e;font-size:10px;line-height:1.55}
.sidebar-foot{margin:19px 8px 0;padding-top:14px;border-top:1px solid #252522;color:#777872;font-size:10px;line-height:1.9}
.topbar{min-height:42px;display:flex;align-items:center;justify-content:space-between;gap:12px;color:#858680;font-size:11px}
.topbar b{color:#d9d8d1;font-weight:600}
.local-badge{padding:7px 10px;border:1px solid #34372d;border-radius:20px;background:#151811;color:#b7c8a6;font-size:10px;white-space:nowrap}
.local-badge i{display:inline-block;width:6px;height:6px;margin-right:7px;border-radius:50%;background:#88b66e}
.page-heading{display:flex;align-items:flex-end;justify-content:space-between;gap:18px;margin:15px 0 19px}
.eyebrow{color:#c4a660;font-size:9px;font-weight:750;letter-spacing:1.5px;margin-bottom:8px}
.page-heading h1,.section-title h1{margin:0;color:#f3f1eb;font-size:clamp(25px,3vw,34px);line-height:1.12;letter-spacing:-.8px}
.page-heading p,.section-title p{max-width:760px;margin:9px 0 0;color:#a3a39e;font-size:13px;line-height:1.6}
.ready-pill{padding:8px 10px;border:1px solid #393a2c;border-radius:20px;background:#181a14;color:#c7d2ac;font-size:10px;white-space:nowrap}
.ready-pill i{display:inline-block;width:6px;height:6px;margin-right:7px;border-radius:50%;background:#9cbd70}
.hero-card{position:relative;overflow:hidden;display:flex;align-items:center;justify-content:space-between;gap:22px;min-height:164px;margin:0 0 14px;padding:25px 28px;border:1px solid #77643c;border-radius:15px;background:radial-gradient(ellipse at 88% 8%,rgba(207,177,101,.15),transparent 35%),linear-gradient(110deg,#282318,#171716 58%,#151515)}
.hero-card:after{content:"";position:absolute;width:185px;height:185px;right:5%;top:-100px;border:1px solid rgba(218,190,120,.24);border-radius:50%;box-shadow:0 0 0 24px rgba(218,190,120,.035),0 0 0 49px rgba(218,190,120,.025)}
.hero-copy{position:relative;z-index:1;max-width:720px}
.hero-card h2{margin:0;color:#f5f1e5;font-size:22px;letter-spacing:-.35px}
.hero-card p{max-width:680px;margin:9px 0 0;color:#c1bcae;font-size:12px;line-height:1.65}
.hero-meta{position:relative;z-index:1;min-width:185px;padding:12px 14px;border:1px solid rgba(218,190,120,.24);border-radius:10px;background:rgba(13,13,12,.32);color:#d8c99f;font-size:10px;line-height:1.65}
.hero-meta b{display:block;color:#f0e7d0;font-size:11px;margin-bottom:3px}
.action-row{display:flex;flex-wrap:nowrap;align-items:stretch;gap:9px;margin:0 0 17px}
.action-row > *{flex:1 1 0;min-width:0}
.action-row button,.action-row .primary-action,.action-row .secondary{width:100%!important;height:44px!important;min-height:44px!important;max-height:44px!important}
.action-row button{display:inline-flex!important;align-items:center!important;justify-content:center!important;padding:0 12px!important;box-sizing:border-box!important;line-height:1.2!important}
.metric-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:15px}
.metric-card,.panel,.content-card{min-width:0;border:1px solid #2d2e2b;border-radius:12px;background:linear-gradient(145deg,#171817,#121313)}
.metric-card{min-height:118px;padding:15px 16px}
.metric-label{color:#a8a8a1;font-size:11px}
.metric-value{margin-top:10px;color:#f2f0e9;font-size:23px;font-weight:700;letter-spacing:-.4px}
.metric-note{margin-top:5px;color:#898a84;font-size:10px;line-height:1.45}
.metric-icon{width:27px;height:27px;display:grid;place-items:center;margin-bottom:9px;border:1px solid #56492d;border-radius:8px;background:#211e17;color:#dfc477;font-size:12px;font-weight:700}
.dashboard-grid{display:grid;grid-template-columns:1.15fr .85fr;gap:13px;margin-bottom:14px}
.panel{padding:17px 18px}
.panel h2{margin:0;color:#efeee8;font-size:14px;font-weight:650}
.panel-sub{color:#969790;font-size:10px;margin:6px 0 14px;line-height:1.5}
.pipeline{display:flex;align-items:stretch;gap:7px;margin-top:14px}
.pipe-step{flex:1;min-width:0;padding:12px 10px;border:1px solid #2d2e2b;border-radius:9px;background:#111212}
.pipe-step b{display:block;color:#e8e6de;font-size:10px;margin-bottom:5px}.pipe-step span{color:#92938d;font-size:9px;line-height:1.5}
.pipe-arrow{align-self:center;color:#c7a957}
.activity-list{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:14px}
.activity-chip{padding:10px 11px;border:1px solid #2d2e2b;border-radius:8px;color:#dfded8;background:#111212;font-size:10px}
.activity-chip i{display:inline-block;width:7px;height:7px;margin-right:8px;border-radius:50%;background:#d9bc76}
.activity-chip:nth-child(2) i{background:#9c8de2}.activity-chip:nth-child(3) i{background:#7195df}.activity-chip:nth-child(4) i{background:#72b6ad}
.activity-chip:nth-child(5) i{background:#d48886}
.quick-actions{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:11px}
.quick-card{min-height:94px;padding:14px;border:1px solid #2d2e2b;border-radius:11px;background:#151615}
.quick-card h3{color:#eceae4;font-size:11px;margin:0 0 6px}.quick-card p{color:#999a93;font-size:10px;line-height:1.55;margin:0}
.section-note{margin-top:14px;padding:12px 14px;border-left:2px solid #8f7844;border-radius:0 8px 8px 0;background:#151512;color:#a2a29a;font-size:10px;line-height:1.65}
.section-title{margin:17px 0 19px}.section-title p{margin-top:7px}
.content-card{padding:15px}
.content-card label{color:#c5c4be!important}
.content-card .wrap,.content-card .form{border-color:#353631!important}
.media-row{align-items:stretch!important;gap:14px!important}
.media-panel{min-width:0!important}
.large-viewer{min-height:520px!important}
.large-viewer .wrap{min-height:520px!important}
.live-camera-viewer .image-container{min-height:420px!important}
.live-camera-viewer img,.live-camera-viewer video{display:block!important;width:100%!important;max-height:420px!important;object-fit:contain!important;object-position:center center!important}
/* Keep Gradio webcam / record controls clickable (do not cover them). */
.live-camera-viewer button,.live-camera-viewer .source-selection,.live-camera-viewer [class*="toolbar"],.live-camera-viewer [class*="icon-button"]{pointer-events:auto!important;z-index:5!important;opacity:1!important;visibility:visible!important}
.primary-action button,button.primary{background:linear-gradient(135deg,#d9bf7c,#b89650)!important;color:#17150f!important;border:1px solid #e1c985!important;font-weight:700!important}
.secondary button,button.secondary{background:#1a1b19!important;border:1px solid #3a3b36!important;color:#e5e3db!important}
button{border-radius:8px!important;transition:filter .15s ease,transform .15s ease!important}
button:hover{filter:brightness(1.08);transform:translateY(-1px)}
.gradio-container input,.gradio-container textarea,.gradio-container select{background:#101110!important;color:#eeede7!important;border-color:#383934!important}
.gradio-container input::placeholder,.gradio-container textarea::placeholder{color:#777871!important}
.gradio-container .wrap,.gradio-container .form{border-color:#33342f!important}
.gradio-container table{background:#111212!important;color:#e0dfd9!important}
.gradio-container th{background:#1c1d1a!important;color:#c5c2b8!important}
.gradio-container td{border-color:#2c2d29!important}
#main-panel .prose h1,#main-panel .prose h2,#main-panel .prose h3{color:#efeee8}
#main-panel .prose p{color:#a5a59e;line-height:1.6}
@media(max-width:1200px){.media-row{flex-direction:column!important}.media-row .media-panel{width:100%!important;max-width:100%!important;flex:1 1 100%!important}}
@media(max-width:1120px){.metric-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.dashboard-grid{grid-template-columns:1fr}}
@media(max-width:760px){#app-shell{flex-direction:column!important;flex-wrap:wrap!important;min-height:100vh;padding:9px;gap:10px}#sidebar{position:static;top:auto;width:100%;height:auto;max-height:42vh;min-height:0;flex:0 0 100%;overflow-y:auto!important;overflow-x:hidden!important;padding:12px}.brand{padding-bottom:13px}.nav-caption,.sidebar-foot{display:none}.sidebar-tip{display:block!important;margin:10px 2px 0;min-height:0}#sidebar-nav fieldset{grid-template-columns:repeat(2,minmax(0,1fr))}#main-panel{height:auto;max-height:none;min-height:0;flex:1 1 100%;overflow:visible!important;padding:2px 2px 15px}.page-heading{align-items:flex-start;flex-direction:column}.hero-card{align-items:flex-start;flex-direction:column;padding:20px}.hero-meta{min-width:0;width:100%}.metric-grid{gap:8px}.metric-card{padding:12px;min-height:105px}.quick-actions{grid-template-columns:1fr}.pipeline{flex-wrap:wrap}.pipe-step{flex-basis:42%}.pipe-arrow{display:none}}
@media(max-width:430px){.metric-grid{grid-template-columns:1fr 1fr}.topbar{align-items:flex-start}.local-badge{font-size:9px}.activity-list{grid-template-columns:1fr 1fr}.live-camera-viewer img{height:320px!important}}
"""


PAGE_CHOICES = [(chr(0x25eb) + "  Overview", "Overview"), (chr(0x25b6) + "  Video Analysis", "Video Analysis"), (chr(0x25a7) + "  Photo Analysis", "Photo Analysis"), (chr(0x25c9) + "  Live Camera", "Live Camera"), (chr(0x2699) + "  About The Model", "About The Model")]
PAGE_NAMES = [value for _, value in PAGE_CHOICES]


def show_page(page):
    return [gr.update(visible=page == name) for name in PAGE_NAMES]


def build_ui(analyse_video, analyse_image, webcam_step, stop_webcam, stop_camera_js, start_camera_js, examples_dir):
    # Gradio 6+: theme/css belong on launch(), not Blocks().
    with gr.Blocks(title="MotionSense | Activity Monitor") as demo:
        with gr.Row(elem_id="app-shell"):
            with gr.Column(elem_id="sidebar", scale=0, min_width=205):
                gr.HTML("""
                <div class="brand">
                  <div class="brand-mark">MS</div>
                  <div>Motion<span>Sense</span><small>Activity Monitor</small></div>
                </div>
                <div class="nav-caption">Analysis</div>
                """)
                nav = gr.Radio(
                    choices=PAGE_CHOICES, value=PAGE_NAMES[0], label=None,
                    show_label=False, elem_id="sidebar-nav",
                )
                gr.HTML("""
                <div class="sidebar-tip">
                  <div class="sidebar-tip-heading"><span class="tip-icon" aria-hidden="true">&#10022;</span><strong>Better Results</strong></div>
                  <p>Keep the camera still and include the person's full body in the frame.</p>
                </div>
                <div class="sidebar-foot">Local Inference<br>Activity labels are estimates.</div>
                """)
            with gr.Column(elem_id="main-panel"):
                gr.HTML("""
                <div class="topbar">
                  <div>MotionSense <span style="color:#55564f">/</span> <b>Activity Workspace</b></div>
                  <div class="local-badge"><i></i>Models Loaded | Processing On This Computer</div>
                </div>
                """)
                with gr.Group(visible=True) as overview_page:
                    gr.HTML("""
                    <div class="page-heading">
                      <div>
                        <div class="eyebrow">Computer Vision | Local Demo</div>
                        <h1>Human Activity Monitor</h1>
                        <p>The system classifies these activities: Standing, Sitting, Walking, Running, and Lying.</p>
                      </div>
                      <div class="ready-pill"><i></i>Models Are Ready</div>
                    </div>
                    <div class="hero-card">
                      <div class="hero-copy">
                        <div class="eyebrow">Recommended Start</div>
                        <h2>Review An Activity Video</h2>
                        <p>Upload a clip to see an annotated preview, how much time each activity appears, and when the predicted activity changes.</p>
                      </div>
                      <div class="hero-meta"><b>Best Setup</b>Fixed Camera | Full Body Visible<br>About 2-5 m from the camera</div>
                    </div>
                    """)
                    with gr.Row(elem_classes=["action-row"]):
                        go_video = gr.Button("Analyze Video", variant="primary", elem_classes=["primary-action"])
                        go_photo = gr.Button("Analyze Photo", variant="primary", elem_classes=["primary-action"])
                        go_live = gr.Button("Open Live Camera", variant="primary", elem_classes=["primary-action"])
                    gr.HTML("""
                    <div class="metric-grid">
                      <div class="metric-card"><div class="metric-icon">05</div><div class="metric-label">Activity Labels</div><div class="metric-note">Standing, Sitting, Walking, Running, Lying</div></div>
                      <div class="metric-card"><div class="metric-icon">03</div><div class="metric-label">Ways To Analyze</div><div class="metric-note">Video | Single Photo | Live Camera</div></div>
                      <div class="metric-card"><div class="metric-icon">Y</div><div class="metric-label">Person Detector</div><div class="metric-value">YOLO11n</div><div class="metric-note">Finds a person in the image</div></div>
                      <div class="metric-card"><div class="metric-icon">33</div><div class="metric-label">Pose Landmarks</div><div class="metric-note">Body points help estimate posture and movement</div></div>
                    </div>
                    <div class="dashboard-grid">
                      <div class="panel">
                        <h2>How The Analysis Works</h2><div class="panel-sub">Each frame moves through these steps.</div>
                        <div class="pipeline">
                          <div class="pipe-step"><b>1 | Find A Person</b><span>The detector locates a person in the frame.</span></div>
                          <div class="pipe-arrow">></div>
                          <div class="pipe-step"><b>2 | Read Body Pose</b><span>Pose landmarks and motion cues are measured.</span></div>
                          <div class="pipe-arrow">></div>
                          <div class="pipe-step"><b>3 | Estimate Activity</b><span>Rules combine the measurements into a label.</span></div>
                        </div>
                      </div>
                      <div class="panel">
                        <h2>Activities This System Recognizes</h2><div class="panel-sub">A person may be labeled Unknown when they are hard to see.</div>
                        <div class="activity-list">
                          <div class="activity-chip"><i></i>Standing</div><div class="activity-chip"><i></i>Sitting</div>
                          <div class="activity-chip"><i></i>Walking</div><div class="activity-chip"><i></i>Running</div>
                          <div class="activity-chip"><i></i>Lying</div>
                        </div>
                      </div>
                    </div>
                    <div class="quick-actions">
                      <div class="quick-card"><h3>Video Analysis</h3><p>Get an annotated clip, a time summary, and an activity timeline.</p></div>
                      <div class="quick-card"><h3>Photo Analysis</h3><p>Estimate posture from one still image. A photo cannot show movement over time.</p></div>
                      <div class="quick-card"><h3>Live Camera</h3><p>Allow camera access to view activity estimates as frames arrive.</p></div>
                    </div>
                    <div class="section-note">This is a project demo. Activity labels are estimates and can be wrong when the person is partly hidden, too far away, or poorly lit. It does not detect falls.</div>
                    """)
                with gr.Group(visible=False) as video_page:
                    gr.HTML("""
                    <div class="section-title"><div class="eyebrow">Video Workspace</div><h1>Analyze A Video</h1>
                    <p>Upload a clip to create an annotated preview and a clear summary of the activities detected in it.</p></div>
                    """)
                    with gr.Row(elem_classes=["media-row"]):
                        with gr.Column(scale=1, min_width=420, elem_classes=["content-card", "media-panel"]):
                            vin = gr.Video(label="Choose A Video", sources=["upload"], height=520, elem_classes=["large-viewer"])
                            gr.Markdown("Use a fixed camera and keep the person's full body in view. Common formats include MP4, AVI, MOV, and MKV.")
                            vbtn = gr.Button("Analyze Video", variant="primary", elem_classes=["primary-action"])
                            with gr.Accordion("Try A Sample Clip", open=False):
                                examples = sorted(str(path) for path in examples_dir.glob("*.mp4"))
                                if examples:
                                    gr.Examples(examples, inputs=vin, label="Project Test Clips")
                                else:
                                    gr.Markdown("No sample clips are available in this project folder.")
                                demo_dir = examples_dir.parent / "demo_tomorrow"
                                demo_clips = sorted(str(path) for path in demo_dir.glob("*.mp4")) if demo_dir.is_dir() else []
                                if demo_clips:
                                    gr.Examples(demo_clips, inputs=vin, label="Tomorrow Demo Pack")
                        with gr.Column(scale=1, min_width=420, elem_classes=["content-card", "media-panel"]):
                            vhead = gr.Markdown("Your annotated result will appear here after analysis.")
                            vout = gr.Video(label="Annotated Video", height=520, elem_classes=["large-viewer"])
                    with gr.Row():
                        with gr.Column(scale=1, min_width=260, elem_classes=["content-card"]):
                            vsum = gr.Dataframe(label="Time By Activity", interactive=False)
                        with gr.Column(scale=1, min_width=260, elem_classes=["content-card"]):
                            vtl = gr.Dataframe(label="Activity Timeline", interactive=False)
                    with gr.Accordion("Technical Details", open=False):
                        vjson = gr.JSON(label="Processing Details")
                    vbtn.click(analyse_video, vin, [vout, vhead, vsum, vtl, vjson])
                with gr.Group(visible=False) as photo_page:
                    gr.HTML("""
                    <div class="section-title"><div class="eyebrow">Photo Workspace</div><h1>Analyze A Photo</h1>
                    <p>Estimate posture in one still image. A photo does not contain enough information to tell whether someone is Walking or Running.</p></div>
                    """)
                    with gr.Row(elem_classes=["media-row"]):
                        with gr.Column(scale=1, min_width=420, elem_classes=["content-card", "media-panel"]):
                            iin = gr.Image(label="Choose A Photo", type="numpy", sources=["upload"], height=520, elem_classes=["large-viewer"])
                            gr.Markdown("For best results, include the person's whole body with clear lighting.")
                            ibtn = gr.Button("Analyze Photo", variant="primary", elem_classes=["primary-action"])
                            with gr.Accordion("Try A Sample Photo", open=False):
                                photos = sorted(str(path) for path in examples_dir.glob("*.jpg"))
                                if photos:
                                    gr.Examples(photos, inputs=iin, label="Project Test Photos")
                                else:
                                    gr.Markdown("No sample photos are available in this project folder.")
                        with gr.Column(scale=1, min_width=420, elem_classes=["content-card", "media-panel"]):
                            ihead = gr.Markdown("Your annotated result will appear here after analysis.")
                            iout = gr.Image(label="Annotated Photo", height=520, elem_classes=["large-viewer"])
                    with gr.Accordion("Pose And Processing Details", open=False):
                        ijson = gr.JSON(label="Technical Details")
                    ibtn.click(analyse_image, iin, [iout, ihead, ijson])
                with gr.Group(visible=False) as live_page:
                    gr.HTML("""
                    <div class="section-title"><div class="eyebrow">Live Workspace</div><h1>Live Camera</h1>
                    <p>Click <b>Start camera</b>, allow the browser permission, then press the orange <b>record</b> button on the preview if it appears. Keep your full body in view. Use <b>Stop camera</b> when finished.</p></div>
                    """)
                    gr.Markdown("The app analyzes incoming frames about four times per second. Keep the camera stationary and stand roughly 2-5 m away.")
                    with gr.Row(elem_classes=["media-row"]):
                        with gr.Column(scale=1, min_width=420, elem_classes=["content-card", "media-panel"]):
                            win = gr.Image(
                                label="Camera Preview",
                                sources=["webcam"],
                                streaming=True,
                                type="numpy",
                                height=480,
                                elem_classes=["large-viewer", "live-camera-viewer"],
                                webcam_options=gr.WebcamOptions(mirror=True),
                            )
                        with gr.Column(scale=1, min_width=420, elem_classes=["content-card", "media-panel"]):
                            wout = gr.Image(label="Live Annotated View", height=480, elem_classes=["large-viewer", "live-camera-viewer"])
                    with gr.Row():
                        wstart = gr.Button("Start camera", variant="primary", elem_classes=["primary-action"], scale=0, min_width=140)
                        wstop = gr.Button("Stop camera", elem_classes=["secondary"], scale=0, min_width=140)
                        whead = gr.Markdown("Click **Start camera**, allow access, then the orange record button if shown.")
                    wstate = gr.State(None)
                    win.stream(webcam_step, [win, wstate], [wout, whead, wstate], stream_every=0.25, concurrency_limit=1)
                    wstart.click(None, None, None, js=start_camera_js)
                    wstop.click(stop_webcam, [wstate], [win, wout, whead, wstate], js=stop_camera_js)
                with gr.Group(visible=False) as model_page:
                    gr.HTML("""
                    <div class="section-title"><div class="eyebrow">System Details</div><h1>About The Model</h1>
                    <p>What the activity monitor uses, what it can report, and when its estimates may be unreliable.</p></div>
                    <div class="dashboard-grid">
                      <div class="panel"><h2>Analysis Components</h2><div class="panel-sub">The models load once when the app starts. Processing runs on the computer hosting this app.</div>
                        <div class="activity-list" style="grid-template-columns:1fr">
                          <div class="activity-chip"><i></i><b>YOLO11n</b> | locates people in each frame</div>
                          <div class="activity-chip"><i></i><b>MediaPipe Pose</b> | estimates 33 body landmarks</div>
                          <div class="activity-chip"><i></i><b>Motion Cues</b> | track movement and background changes</div>
                          <div class="activity-chip"><i></i><b>Rule-Based Classifier</b> | estimates one of five activities</div>
                        </div>
                      </div>
                      <div class="panel"><h2>Getting A Useful Result</h2><div class="panel-sub">The system works best when the person is easy to see.</div>
                        <div class="pipe-step" style="margin-bottom:8px"><b>Camera Position</b><span>Keep the camera stationary.</span></div>
                        <div class="pipe-step" style="margin-bottom:8px"><b>Framing</b><span>Include the person's full body.</span></div>
                        <div class="pipe-step"><b>Distance And Lighting</b><span>Around 2-5 m away, in a well-lit room.</span></div>
                      </div>
                    </div>
                    <div class="section-note">A single photo can estimate posture but cannot show motion. This demo classifies basic activity; it is not a fall detector or medical monitoring system.</div>
                    """)
        pages = [overview_page, video_page, photo_page, live_page, model_page]
        nav.change(show_page, inputs=nav, outputs=pages)
        go_video.click(lambda: PAGE_NAMES[1], outputs=nav).then(show_page, inputs=nav, outputs=pages)
        go_photo.click(lambda: PAGE_NAMES[2], outputs=nav).then(show_page, inputs=nav, outputs=pages)
        go_live.click(lambda: PAGE_NAMES[3], outputs=nav).then(show_page, inputs=nav, outputs=pages)

    return demo
