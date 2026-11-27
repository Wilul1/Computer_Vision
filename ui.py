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

/* ---- CareView product (document scroll — Gradio nested fixed boxes block overflow) ---- */
html,body{height:auto!important;max-height:none!important;overflow-x:hidden!important;overflow-y:scroll!important}
.gradio-container,main,.main,.wrap,.app,.fillable,#root{
  height:auto!important;max-height:none!important;min-height:0!important;overflow:visible!important}
#care-page-wrap{position:relative!important;inset:auto!important;top:auto!important;left:auto!important;right:auto!important;bottom:auto!important;width:100%!important;height:auto!important;min-height:100vh!important;max-height:none!important;z-index:1;margin:0!important;padding:0!important;max-width:none!important;background:#F1F5F9!important;overflow:visible!important;color-scheme:light}
#care-page-wrap > .form,#care-page-wrap > div,#care-page-wrap .bubble-wrap,#care-page-wrap .block,
#cv-layout,#careview-shell,#careview-shell .block,#careview-shell .form{
  height:auto!important;max-height:none!important;min-height:0!important;overflow:visible!important}
#cv-layout{display:flex!important;flex-wrap:nowrap!important;align-items:flex-start;min-height:100vh;gap:0!important;margin:0!important}
#cv-gradio-rail{flex:0 0 232px!important;width:232px!important;min-width:232px!important;max-width:232px!important;background:#15233F!important;background-color:#15233F!important;color:#F8FAFC!important;padding:22px 14px!important;border:0!important;border-radius:0!important;position:sticky;top:0;align-self:flex-start;height:100vh;max-height:100vh;overflow:hidden;z-index:5}
#cv-gradio-rail,#cv-gradio-rail *{--body-text-color:#F8FAFC!important;--block-background-fill:transparent!important;--background-fill-primary:transparent!important;--background-fill-secondary:transparent!important}
#cv-gradio-rail .cv-brand{display:flex;gap:12px;align-items:center;padding:2px 8px 24px}
#cv-gradio-rail .cv-brand-mark{width:38px;height:38px;border-radius:12px;background:#3B82F6;display:grid;place-items:center;color:#fff!important;font-size:17px;font-weight:700}
#cv-gradio-rail .cv-brand-name{font-size:16px;font-weight:750;color:#FFFFFF!important}
#cv-gradio-rail .cv-brand-sub{font-size:11px;color:#A8B4C8!important;margin-top:3px}
#cv-gradio-rail .cv-nav-foot{padding:14px 12px;color:#A8B4C8!important;font-size:12px;font-weight:600;border-top:1px solid rgba(255,255,255,.12);margin-top:16px}
#cv-side-nav,#cv-side-nav .form,#cv-side-nav .block,#cv-side-nav .wrap,
#cv-side-nav .panel-wrapper,#cv-side-nav .svelte-frame,
#cv-gradio-rail .block,#cv-gradio-rail .form,#cv-gradio-rail .wrap{
  background:transparent!important;background-color:transparent!important;
  border:0!important;box-shadow:none!important;padding:0!important;margin:0!important;
  color:#F8FAFC!important}
#cv-side-nav fieldset{display:flex!important;flex-direction:column!important;gap:8px;border:0!important;padding:0!important;margin:0!important;background:transparent!important}
#cv-side-nav label{
  width:100%!important;min-height:44px;display:flex!important;align-items:center;padding:0 14px;
  border-radius:12px;background:#243656!important;background-color:#243656!important;
  color:#FFFFFF!important;cursor:pointer!important;border:1px solid rgba(255,255,255,.12)!important;
  font-size:14px;font-weight:650;opacity:1!important}
#cv-side-nav label:hover{background:#2F4570!important;background-color:#2F4570!important;color:#FFFFFF!important}
#cv-side-nav label:has(input:checked){
  background:#2563EB!important;background-color:#2563EB!important;color:#FFFFFF!important;
  border-color:#2563EB!important}
#cv-side-nav label span,#cv-side-nav label *{color:#FFFFFF!important;opacity:1!important}
#cv-side-nav label input{width:0;height:0;margin:0;opacity:0;position:absolute}
#careview-shell{background:#F1F5F9!important;border:0!important;border-radius:0!important;padding:0!important;padding-bottom:28px!important;color:#0F172A!important;min-height:100%;height:auto;max-width:none!important;flex:1!important;overflow:visible;
  /* Gradio .prose * { color: var(--body-text-color) } inherits the dark-lab cream text;
     force CareView's light-theme body color so board + patient details stay readable. */
  --body-text-color:#334155!important;--body-text-color-subdued:#64748B!important;
  --color-text-body:#334155!important;--color-text-primary:#0F172A!important;
  --color-text-secondary:#64748B!important}
#careview-shell .cv-tools .prose,#careview-shell .cv-tools label,#careview-shell .cv-tools p,#careview-shell .cv-tools span,#careview-shell .cv-main{color:#334155}
#careview-shell .cv-main .cv-hello,#careview-shell .cv-main .cv-stat-n,#careview-shell .cv-main .cv-patient-name,#careview-shell .cv-main .cv-alert-title,#careview-shell .cv-section-head h2{color:#0F172A!important}
#careview-shell .cv-main .cv-sub,#careview-shell .cv-main .cv-stat-l,#careview-shell .cv-main .cv-patient-room,#careview-shell .cv-main .cv-patient-bell,#careview-shell .cv-main .cv-alert-time{color:#64748B!important}
#careview-shell textarea,#careview-shell input,#careview-shell select{background:#fff!important;color:#0F172A!important;border:1px solid #E2E8F0!important;border-radius:10px!important}
/* Add patient: keep form visible (accordion header was black-on-black). */
#careview-shell .cv-add-patient{
  background:#fff!important;border:1px solid #E2E8F0!important;border-radius:14px!important;
  padding:12px 14px!important;margin:8px 0 12px!important;box-shadow:0 1px 2px rgba(15,23,42,.04)}
#careview-shell .cv-add-patient > button,
#careview-shell .cv-add-patient > .label-wrap,
#careview-shell .cv-add-patient summary,
#careview-shell .cv-add-patient .svelte-accordion,
#careview-shell .cv-add-patient [class*="accordion"] > button{
  background:#fff!important;color:#0F172A!important;border:0!important;
  font-weight:700!important;font-size:15px!important;opacity:1!important}
#careview-shell .cv-add-patient label,
#careview-shell .cv-add-patient span,
#careview-shell .cv-add-patient p{color:#334155!important;opacity:1!important}
#careview-shell .cv-add-patient input[type=radio],
#careview-shell .cv-add-patient input[type=checkbox]{
  -webkit-appearance:auto!important;appearance:auto!important;
  width:18px!important;height:18px!important;min-width:18px!important;min-height:18px!important;
  margin:0 8px 0 0!important;opacity:1!important;position:static!important;
  pointer-events:auto!important;cursor:pointer!important;accent-color:#2563EB!important;
  background:transparent!important;border:0!important;border-radius:0!important;flex-shrink:0!important}
#careview-shell .cv-add-patient fieldset{border:0!important;padding:0!important;margin:6px 0 10px!important}
#careview-shell .cv-add-patient fieldset label{
  display:flex!important;align-items:center!important;gap:8px!important;
  min-height:40px!important;padding:8px 10px!important;margin:0 0 6px!important;
  background:#F8FAFC!important;border:1px solid #E2E8F0!important;border-radius:10px!important;
  pointer-events:auto!important;cursor:pointer!important;color:#0F172A!important}
#careview-shell .gr-button,#careview-shell button{border-radius:10px!important}
#careview-shell button.primary,#careview-shell .primary{background:#2563EB!important;border-color:#2563EB!important;color:#fff!important}
#careview-shell .cv-tools .markdown-body,#careview-shell .cv-tools .prose,#careview-shell .cv-tools .prose *{color:#334155!important}
#careview-root,#careview-root *{box-sizing:border-box}
/* Patient details — beat Gradio .prose * { color: var(--body-text-color) }; scoped to #careview-root only */
#careview-shell .prose:has(#careview-root.pd-page),
#careview-shell .prose:has(#careview-root.pd-page) #careview-root.pd-page{--body-text-color:#334155!important;color:#334155!important}
#careview-shell #careview-root.pd-page{background:#F1F5F9!important;color:#0F172A!important}
#careview-shell #careview-root.pd-page .pd-head,
#careview-shell #careview-root.pd-page .pd-sec{background:#FFFFFF!important;border-color:#E2E8F0!important}
#careview-shell #careview-root.pd-page .pd-name,
#careview-shell #careview-root.pd-page .pd-stat-n{color:#0F172A!important}
#careview-shell #careview-root.pd-page .pd-sec h2,
#careview-shell #careview-root.pd-page .pd-row-title,
#careview-shell #careview-root.pd-page .pd-meta b,
#careview-shell #careview-root.pd-page .pd-act-top b,
#careview-shell #careview-root.pd-page .pd-snap-meta b{color:#1E293B!important}
#careview-shell #careview-root.pd-page .pd-meta,
#careview-shell #careview-root.pd-page .pd-act-top,
#careview-shell #careview-root.pd-page .pd-snap-meta{color:#334155!important}
#careview-shell #careview-root.pd-page .pd-sub,
#careview-shell #careview-root.pd-page .pd-muted,
#careview-shell #careview-root.pd-page .pd-row-sub,
#careview-shell #careview-root.pd-page .pd-stat-l,
#careview-shell #careview-root.pd-page .pd-empty{color:#64748B!important}
#careview-shell #careview-root.pd-page .pd-back{color:#2563EB!important}
#careview-shell #careview-root.pd-page .pd-chip{background:#F1F5F9!important;color:#475569!important}
#careview-shell #careview-root.pd-page .pd-chip-rec-ok{background:#DCFCE7!important;color:#166534!important}
#careview-shell #careview-root.pd-page .pd-empty,
#careview-shell #careview-root.pd-page .pd-stat,
#careview-shell #careview-root.pd-page .pd-snap{background:#F8FAFC!important;border-color:#E2E8F0!important}
#careview-shell #careview-root.pd-page .pd-focus{background:#EFF6FF!important}
#careview-shell .pd-player-panel{background:#0F172A;border-radius:14px;overflow:hidden}
#careview-shell .pd-player{display:block;width:100%;max-height:min(56vh,420px);background:#000;aspect-ratio:16/9;object-fit:contain}
#careview-shell .pd-player-info{margin-top:8px;color:#334155!important}
#careview-shell .cv-play-block{background:transparent!important;border:0!important;border-radius:0!important;padding:0!important;margin:8px 0!important;box-shadow:none!important}
#careview-shell .cv-details-tools,#careview-shell .cv-handle-tools,#careview-shell .cv-ac-filter-wrap{margin:0 0 8px}
#careview-shell .cv-tools .label-wrap,#careview-shell .cv-tools .form{border:0!important;box-shadow:none!important}
#careview-shell .cv-toolbar{gap:8px;margin:0 0 6px}
#careview-shell #careview-root.pd-page .pd-pill-safe{background:#DCFCE7!important;color:#16A34A!important}
#careview-shell #careview-root.pd-page .pd-pill-attention{background:#FEF9C3!important;color:#A16207!important}
#careview-shell #careview-root.pd-page .pd-pill-alert{background:#FEE2E2!important;color:#DC2626!important}
.cv-app{display:flex;min-height:100vh;background:#F1F5F9;font-family:"Segoe UI",ui-sans-serif,system-ui,sans-serif}
#careview-shell .cv-rail,.cv-rail{width:232px;flex:0 0 232px;background:#15233F!important;color:#F8FAFC!important;padding:26px 16px;display:flex;flex-direction:column;min-height:100vh}
#careview-shell .cv-rail,#careview-shell .cv-rail *,#careview-shell .prose .cv-rail,#careview-shell .prose .cv-rail *{color:#F8FAFC!important}
.cv-brand{display:flex;gap:12px;align-items:center;padding:2px 8px 32px}
.cv-brand-mark{width:38px;height:38px;border-radius:12px;background:#3B82F6;display:grid;place-items:center;color:#fff!important;font-size:17px;font-weight:700}
.cv-brand-name{font-size:16px;font-weight:750;color:#FFFFFF!important;letter-spacing:-.2px}
.cv-brand-sub{font-size:11px;color:#94A3B8!important;margin-top:3px}
.cv-nav{display:flex;flex-direction:column;gap:8px;flex:1}
.cv-nav-item{padding:12px 14px;border-radius:12px;font-size:14px;color:#E2E8F0!important;font-weight:600;background:rgba(255,255,255,.04)}
.cv-nav-active{background:#2563EB!important;color:#FFFFFF!important}
.cv-nav-badge{margin-left:8px;background:#EF4444!important;color:#fff!important;border-radius:999px;font-size:10px;padding:2px 7px}
.cv-nav-foot{padding:14px 12px;color:#94A3B8!important;font-size:12px;font-weight:600;border-top:1px solid rgba(255,255,255,.12);margin-top:12px}
.cv-main{flex:1;min-width:0;padding:22px 28px 20px;background:#EEF2F6;width:100%;box-sizing:border-box}
.cv-top{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;margin-bottom:20px}
.cv-hello{font-size:30px;font-weight:750;color:#0F172A;letter-spacing:-.6px}
.cv-sub{margin-top:6px;font-size:14px;color:#64748B}
.cv-top-right{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.cv-live-pill{display:inline-flex;align-items:center;gap:7px;padding:8px 14px;border-radius:999px;background:#DCFCE7;color:#166534;font-size:12px;font-weight:650}
.cv-live-pill i{width:7px;height:7px;border-radius:50%;background:#16A34A;display:inline-block}
.cv-user{padding:8px 14px;border-radius:999px;background:#fff;border:1px solid #E2E8F0;font-size:12px;color:#1E293B;font-weight:600}
.cv-stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-bottom:22px}
.cv-stats-compact{gap:8px;margin-bottom:14px}
.cv-stat{background:#fff;border:1px solid #E2E8F0;border-radius:18px;padding:18px 20px;min-height:96px;display:flex;gap:14px;align-items:center;box-shadow:0 1px 2px rgba(15,23,42,.04)}
.cv-stats-compact .cv-stat{min-height:0;padding:12px 10px;border-radius:14px;flex-direction:column;align-items:center;text-align:center;gap:2px}
.cv-stat-icon{width:44px;height:44px;border-radius:14px;display:grid;place-items:center;font-weight:800;font-size:16px}
.cv-stat-safe .cv-stat-icon{background:#DCFCE7;color:#16A34A}
.cv-stat-att .cv-stat-icon{background:#FEF9C3;color:#A16207}
.cv-stat-alert .cv-stat-icon{background:#FEE2E2;color:#DC2626}
.cv-stat-safe .cv-stat-n{color:#16A34A!important}
.cv-stat-att .cv-stat-n{color:#CA8A04!important}
.cv-stat-alert .cv-stat-n{color:#DC2626!important}
.cv-stat-n{font-size:30px;font-weight:750;color:#0F172A;line-height:1}
.cv-stats-compact .cv-stat-n{font-size:20px}
.cv-stat-l{margin-top:4px;font-size:12px;color:#64748B;font-weight:600}
.cv-stats-compact .cv-stat-l{margin-top:0;font-size:11px}
.cv-section-head{display:flex;justify-content:space-between;align-items:baseline;margin:6px 0 12px}
.cv-section-head h2{margin:0;font-size:17px;color:#0F172A;font-weight:700}
.cv-link{font-size:12px;color:#2563EB;font-weight:650}
.cv-muted{font-size:11px;color:#94A3B8}
.cv-patients{display:grid;grid-template-columns:1fr;gap:10px;width:100%;max-width:100%;box-sizing:border-box}
.cv-patient-card{background:#fff;border:1px solid #E2E8F0;border-radius:16px;padding:12px;box-shadow:none;
  width:100%;max-width:100%;box-sizing:border-box;overflow:hidden}
.cv-patient-row{display:flex;align-items:center;gap:12px;min-height:0;padding:0;width:100%;max-width:100%;box-sizing:border-box}
.cv-patient-who{flex:1 1 auto;min-width:0;overflow:hidden}
.cv-patient-top{display:flex;justify-content:space-between;align-items:flex-start}
.cv-avatar{width:42px;height:42px;border-radius:12px;display:grid;place-items:center;background:#EFF6FF;color:#2563EB;font-weight:750;font-size:15px;flex:0 0 42px}
.cv-pill{padding:4px 9px;border-radius:999px;font-size:10px;font-weight:750;letter-spacing:.03em;flex:0 0 auto;text-transform:uppercase;white-space:nowrap}
.cv-patient-name{margin-top:0;font-size:15px;font-weight:750;color:#0F172A}
.cv-patient-room{margin-top:2px;font-size:12px;color:#64748B;line-height:1.35;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:100%}
.cv-open-link{margin-top:10px;font-size:12px;font-weight:700;color:#2563EB}
.cv-patient-act,.cv-patient-bell,.cv-card-hint{display:none}
.cv-clickable{cursor:pointer!important;transition:border-color .15s ease,box-shadow .15s ease,transform .12s ease}
.cv-clickable:hover{border-color:#2563EB!important;box-shadow:0 4px 14px rgba(37,99,235,.16)!important;transform:translateY(-1px)}
.cv-alerts{background:#fff;border:1px solid #E2E8F0;border-radius:18px;overflow:hidden;box-shadow:0 1px 2px rgba(15,23,42,.04)}
.cv-alert-row{display:flex;gap:12px;align-items:center;padding:14px 16px;border-bottom:1px solid #F1F5F9}
.cv-alert-row:last-child{border-bottom:0}
.cv-alert-row.cv-clickable:hover{background:#F8FAFC}
.cv-alert-thumb{width:48px;height:48px;border-radius:12px;background:#FEE2E2;color:#DC2626;display:grid;place-items:center;font-size:11px;font-weight:700}
.cv-alert-title{font-size:13px;font-weight:650;color:#0F172A}
.cv-alert-desc{font-size:12px;color:#DC2626;font-weight:600;margin-top:2px}
.cv-alert-time{font-size:10px;color:#94A3B8;margin-top:2px}
.cv-alert-open{margin-left:auto;padding:8px 14px;border-radius:999px;background:#FEE2E2;color:#B91C1C;font-size:11px;font-weight:700;white-space:nowrap}
.cv-alert-empty{padding:28px 16px;text-align:center;color:#64748B;font-size:13px;line-height:1.45;
  background:#fff;border:1px dashed #E2E8F0;border-radius:16px}
.cv-tools{margin:0;padding:12px 22px 28px;background:#F1F5F9!important;border:0;border-radius:0;border-top:0}
#careview-shell .cv-toolbar label{color:#334155!important;font-weight:600!important}
#careview-shell .cv-exit button{background:#fff!important;color:#1A2B4B!important;border:1px solid #CBD5E1!important}
#careview-shell,#careview-shell .block,#careview-shell .form,#careview-shell .panel-wrapper{background:#F1F5F9!important;color:#0F172A!important;border-color:#E2E8F0!important}
#careview-shell .progress-bar,#careview-shell .progress-level{background:#DBEAFE!important}
#careview-shell .meta-text,#careview-shell .progress-text,#careview-shell .eta-bar{color:#475569!important;background:transparent!important}
/* Kill dark Gradio chrome in Patient actions / Watch recording / Care response */
#care-page-wrap,#careview-shell,.cv-tools{
  --background-fill-primary:#FFFFFF!important;
  --background-fill-secondary:#F8FAFC!important;
  --block-background-fill:#FFFFFF!important;
  --block-border-color:#E2E8F0!important;
  --block-label-text-color:#64748B!important;
  --block-title-text-color:#0F172A!important;
  --body-background-fill:#F1F5F9!important;
  --body-text-color:#0F172A!important;
  --body-text-color-subdued:#64748B!important;
  --border-color-primary:#E2E8F0!important;
  --color-accent:#2563EB!important;
  --input-background-fill:#FFFFFF!important;
  --input-border-color:#E2E8F0!important;
  --input-placeholder-color:#94A3B8!important;
  --neutral-950:#0F172A!important;
  --neutral-900:#1E293B!important;
  --neutral-800:#334155!important;
  --neutral-700:#475569!important;
  --neutral-600:#64748B!important;
  --neutral-500:#94A3B8!important;
  --neutral-400:#CBD5E1!important;
  --neutral-300:#E2E8F0!important;
  --neutral-200:#F1F5F9!important;
  --neutral-100:#F8FAFC!important;
  --neutral-50:#FFFFFF!important;
  color-scheme:light!important}
#careview-shell .cv-details-tools .row,
#careview-shell .cv-details-tools .form,
#careview-shell .cv-details-tools .block,
#careview-shell .cv-details-tools .panel-wrapper,
#careview-shell .cv-details-tools .svelte-group,
#careview-shell .cv-handle-tools .row,
#careview-shell .cv-handle-tools .form,
#careview-shell .cv-handle-tools .block,
#careview-shell .cv-handle-tools .panel-wrapper,
#careview-shell .cv-play-block,
#careview-shell .cv-play-block .form,
#careview-shell .cv-play-block .block,
#careview-shell .cv-toolbar,
#careview-shell .cv-exit{
  background:#FFFFFF!important;background-color:#FFFFFF!important;
  border-color:#E2E8F0!important;box-shadow:none!important;color:#0F172A!important}
#careview-shell .cv-details-tools label,
#careview-shell .cv-handle-tools label,
#careview-shell .cv-play-block label,
#careview-shell .cv-details-tools .label-wrap,
#careview-shell .cv-handle-tools .label-wrap,
#careview-shell .cv-play-block .label-wrap,
#careview-shell .cv-details-tools span,
#careview-shell .cv-play-block span,
#careview-shell .cv-details-tools p,
#careview-shell .cv-play-block p{
  color:#334155!important;opacity:1!important}
#careview-shell .cv-details-tools .container,
#careview-shell .cv-handle-tools .container,
#careview-shell .cv-play-block .container,
#careview-shell .cv-details-tools [class*="wrap"],
#careview-shell .cv-handle-tools [class*="wrap"],
#careview-shell .cv-play-block [class*="wrap"]{
  background:transparent!important;background-color:transparent!important;
  border-color:#E2E8F0!important;color:#0F172A!important}
#careview-shell .cv-details-tools input,
#careview-shell .cv-details-tools textarea,
#careview-shell .cv-details-tools select,
#careview-shell .cv-handle-tools input,
#careview-shell .cv-handle-tools textarea,
#careview-shell .cv-handle-tools select,
#careview-shell .cv-play-block input,
#careview-shell .cv-play-block textarea,
#careview-shell .cv-play-block select,
#careview-shell #cv-session-select input,
#careview-shell #cv-session-select textarea{
  background:#FFFFFF!important;color:#0F172A!important;border:1px solid #E2E8F0!important;
  border-radius:10px!important;caret-color:#0F172A!important}
#careview-shell .cv-exit button,
#careview-shell .cv-details-tools button:not(.primary):not(.cv-cr-resolve){
  background:#FFFFFF!important;color:#1E3A5F!important;border:1px solid #CBD5E1!important}
#careview-shell .cv-details-tools button.primary,
#careview-shell .cv-details-tools .primary{
  background:#2563EB!important;border-color:#2563EB!important;color:#FFFFFF!important}
/* Dropdown option menus (often portaled outside the card) */
#care-page-wrap [role="listbox"],
#careview-shell [role="listbox"],
body > [role="listbox"],
body .options,
body ul.options,
body [class*="options"],
#care-page-wrap ul[role="listbox"],
#careview-shell ul[role="listbox"]{
  background:#FFFFFF!important;color:#0F172A!important;border:1px solid #E2E8F0!important;
  border-radius:12px!important;box-shadow:0 8px 24px rgba(15,23,42,.12)!important}
#care-page-wrap [role="option"],
#careview-shell [role="option"],
body > [role="listbox"] [role="option"],
body .options li,
body ul.options li,
body [class*="options"] li,
#care-page-wrap [role="listbox"] li,
#careview-shell [role="listbox"] li{
  background:#FFFFFF!important;color:#0F172A!important;border:0!important;
  opacity:1!important;font-size:13px!important}
#care-page-wrap [role="option"][aria-selected="true"],
#careview-shell [role="option"][aria-selected="true"],
body > [role="listbox"] [role="option"][aria-selected="true"],
body .options li.selected,
body ul.options li.selected,
body [class*="options"] li[aria-selected="true"]{
  background:#EFF6FF!important;color:#1E40AF!important}
#care-page-wrap [role="option"]:hover,
#careview-shell [role="option"]:hover,
body > [role="listbox"] [role="option"]:hover,
body .options li:hover,
body ul.options li:hover{
  background:#F1F5F9!important;color:#0F172A!important}
/* Light tool cards — Settings-only add form; no dark accordion bars on patient page */
#careview-shell .cv-status-note{
  background:#fff!important;border:1px solid #E2E8F0!important;border-radius:14px!important;
  padding:12px 16px!important;margin:0 0 12px!important;box-shadow:0 1px 2px rgba(15,23,42,.04)}
#careview-shell .cv-status-note .prose,#careview-shell .cv-status-note p,
#careview-shell .cv-status-note *{color:#334155!important}
#careview-shell .cv-details-tools,
#careview-shell .cv-handle-tools,
#careview-shell .cv-settings-tools,
#careview-shell .cv-events-log,
#careview-shell .cv-ac-filter-wrap{
  background:#fff!important;border:1px solid #E2E8F0!important;border-radius:16px!important;
  padding:14px 16px!important;margin:0 0 12px!important;box-shadow:0 1px 2px rgba(15,23,42,.04)}
#careview-shell .cv-section-label{
  display:block;font-size:11px;font-weight:750;letter-spacing:.06em;text-transform:uppercase;
  color:#64748B!important;margin:0 0 10px}
/* Care Response panel (no icons) */
#careview-shell .cv-handle-tools{padding:16px!important}
#careview-shell .cv-cr{margin:0 0 14px}
#careview-shell .cv-cr-head{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin:0 0 12px}
#careview-shell .cv-cr-title{font-size:18px;font-weight:750;color:#1E3A5F!important;letter-spacing:-.2px}
#careview-shell .cv-cr-sub{margin-top:4px;font-size:12px;color:#64748B!important;line-height:1.45}
#careview-shell .cv-cr-badge{
  flex:0 0 auto;padding:6px 12px;border-radius:999px;background:#FEE2E2!important;color:#B91C1C!important;
  font-size:11px;font-weight:750;white-space:nowrap}
#careview-shell .cv-cr-empty{
  padding:14px;border-radius:12px;background:#F8FAFC!important;border:1px solid #E2E8F0!important;
  color:#64748B!important;font-size:13px}
#careview-shell .cv-cr-incident{
  display:grid;grid-template-columns:1.4fr .9fr;gap:14px;align-items:start;
  padding:14px 16px;border-radius:14px;background:#FEF2F2!important;border:1px solid #FECACA!important}
#careview-shell .cv-cr-incident-id{font-size:12px;font-weight:700;color:#B91C1C!important;margin-bottom:4px}
#careview-shell .cv-cr-incident-title{font-size:16px;font-weight:750;color:#7F1D1D!important;line-height:1.25}
#careview-shell .cv-cr-incident-blurb{margin-top:6px;font-size:12px;color:#64748B!important;line-height:1.5}
#careview-shell .cv-cr-incident-side{display:grid;gap:8px;font-size:12px;color:#334155!important}
#careview-shell .cv-cr-incident-side span{display:block;font-size:10px;font-weight:700;color:#94A3B8!important;margin-bottom:2px}
#careview-shell .cv-cr-incident-side b{font-weight:700;color:#1E293B!important}
#careview-shell .cv-cr-wait{
  display:inline-block;margin-top:2px;padding:5px 10px;border-radius:8px;background:#FEE2E2!important;
  color:#B91C1C!important;font-size:11px;font-weight:750;width:fit-content}
#careview-shell .cv-cr-step{display:flex;gap:10px;align-items:flex-start;margin:16px 0 10px}
#careview-shell .cv-cr-step-n{
  width:22px;height:22px;border-radius:999px;background:#2563EB!important;color:#fff!important;
  font-size:11px;font-weight:750;display:grid;place-items:center;flex:0 0 auto;margin-top:1px}
#careview-shell .cv-cr-step-h{font-size:14px;font-weight:750;color:#1E3A5F!important}
#careview-shell .cv-cr-step-s{margin-top:2px;font-size:12px;color:#64748B!important}
#careview-shell .cv-outcome-radio fieldset,
#careview-shell .cv-outcome-radio .wrap{border:0!important;padding:0!important;margin:0!important}
#careview-shell .cv-outcome-radio{margin:0 0 4px!important}
#careview-shell .cv-outcome-radio .form,
#careview-shell .cv-outcome-radio .block{border:0!important;background:transparent!important;box-shadow:none!important;padding:0!important}
#careview-shell .cv-outcome-radio label{
  display:flex!important;flex-direction:column!important;align-items:flex-start!important;justify-content:flex-start!important;
  gap:4px!important;min-height:84px;padding:12px 12px 12px!important;margin:0!important;
  border:1.5px solid #E2E8F0!important;border-radius:12px!important;background:#fff!important;
  box-shadow:none!important;cursor:pointer;position:relative;white-space:pre-line!important;
  color:#1E293B!important;font-size:12px!important;line-height:1.35!important;font-weight:600!important}
#careview-shell .cv-outcome-radio .wrap,
#careview-shell .cv-outcome-radio fieldset{
  display:grid!important;grid-template-columns:repeat(auto-fit,minmax(148px,1fr))!important;gap:10px!important}
#careview-shell .cv-outcome-radio label span{
  white-space:pre-line!important;text-align:left!important;color:#1E293B!important;font-size:12px!important;line-height:1.35!important}
#careview-shell .cv-outcome-radio label input{position:absolute;top:10px;right:10px;margin:0;accent-color:#16A34A}
#careview-shell .cv-outcome-radio label:has(input:checked){
  border-color:#16A34A!important;background:#F0FDF4!important;box-shadow:0 0 0 1px #16A34A inset!important}
#careview-shell .cv-outcome-radio label:has(input:checked) span{color:#166534!important}
#careview-shell .cv-cr-note textarea{
  min-height:96px!important;border-radius:12px!important;border:1px solid #E2E8F0!important;
  background:#fff!important;color:#0F172A!important}
#careview-shell .cv-cr-info{
  margin:12px 0;padding:10px 12px;border-radius:10px;background:#EFF6FF!important;border:1px solid #BFDBFE!important;
  color:#1E40AF!important;font-size:12px;line-height:1.45}
#careview-shell .cv-cr-actions{gap:10px;margin:4px 0 0}
#careview-shell .cv-cr-actions button{
  min-height:52px!important;border-radius:12px!important;font-weight:700!important;
  line-height:1.25!important;white-space:pre-line!important;padding:10px 12px!important}
#careview-shell .cv-cr-keep button,
#careview-shell .cv-cr-keep .secondary,
#careview-shell button.cv-cr-keep{
  background:#fff!important;color:#1E3A5F!important;border:1.5px solid #1E3A5F!important}
#careview-shell .cv-cr-resolve button,
#careview-shell .cv-cr-resolve .primary,
#careview-shell button.cv-cr-resolve{
  background:#15803D!important;color:#fff!important;border-color:#15803D!important}
#careview-shell .cv-cr-event .wrap{margin-bottom:8px}
@media(max-width:700px){
  #careview-shell .cv-cr-incident{grid-template-columns:1fr}
  #careview-shell .cv-outcome-radio .wrap,
  #careview-shell .cv-outcome-radio fieldset{grid-template-columns:1fr!important}
}
#careview-shell .cv-details-tools button.primary,
#careview-shell .cv-details-tools .primary{background:#2563EB!important;border-color:#2563EB!important;color:#fff!important}
/* History list rows — beat Gradio dark button chrome */
#careview-shell .cv-hist2-row,
#careview-shell button.cv-hist2-row,
#careview-shell .cv-hist2-list .cv-hist2-row{
  background:#FFFFFF!important;background-color:#FFFFFF!important;
  color:#0F172A!important;border:1px solid #E2E8F0!important;
  box-shadow:none!important;opacity:1!important}
#careview-shell .cv-hist2-row:hover,
#careview-shell button.cv-hist2-row:hover{
  background:#F8FAFC!important;border-color:#93C5FD!important;color:#0F172A!important}
#careview-shell .cv-hist2-row.is-selected,
#careview-shell button.cv-hist2-row.is-selected{
  background:#EFF6FF!important;border-color:#2563EB!important;color:#0F172A!important;
  box-shadow:0 0 0 1px #2563EB inset!important}
#careview-shell .cv-hist2-row-name{color:#0F172A!important}
#careview-shell .cv-hist2-row-title{color:#B91C1C!important}
#careview-shell .cv-hist2-row-time{color:#64748B!important}
#careview-shell .cv-hist2-pill{background:#DCFCE7!important;color:#166534!important}
#careview-shell .cv-hist2-avatar{background:#DBEAFE!important;color:#1D4ED8!important}
#careview-shell .cv-hist2-panel,#careview-shell .cv-hist2-stat,#careview-shell .cv-hist2-card{
  background:#FFFFFF!important;border-color:#E2E8F0!important;color:#0F172A!important}
#careview-shell .cv-hist2-stat-n,#careview-shell .cv-hist2-panel-h,
#careview-shell .cv-hist2-detail-h,#careview-shell .cv-hist2-card-h,
#careview-shell .cv-hist2-alert-title{color:#0F172A!important}
#careview-shell .cv-hist2-stat.fall .cv-hist2-stat-n{color:#B91C1C!important}
#careview-shell .cv-hist2-stat.ok .cv-hist2-stat-n{color:#15803D!important}
#careview-shell .cv-hist2-alert{background:#F8FAFC!important;border-color:#E2E8F0!important}
#careview-shell .cv-hist2-alert-fall{background:#FEF2F2!important;border-color:#FECACA!important}
#careview-shell .cv-hist2-alert-fall .cv-hist2-alert-title{color:#7F1D1D!important}
#careview-shell .cv-settings-tools .cv-add-patient{
  background:transparent!important;border:0!important;padding:0!important;margin:0!important;box-shadow:none!important}
/* Off-screen hooks so board HTML can trigger real Gradio actions */
/* Off-screen hooks must stay clickable via JS (.click()) — pointer-events:none breaks Gradio. */
#cv-hidden-hooks{position:fixed!important;left:-10000px!important;top:0!important;width:1px!important;height:1px!important;overflow:hidden!important;opacity:0.01!important;pointer-events:auto!important;z-index:-1!important}
/* Desktop caregiver (tablet+): sticky side rail; page/body scrolls */
@media(min-width:769px){
  #cv-mobile-top{display:none!important}
  #cv-gradio-rail{display:flex!important;flex-direction:column!important;
    position:sticky!important;top:0!important;height:100vh!important;
    max-height:100vh!important;align-self:flex-start!important;overflow:hidden!important}
  #careview-shell{flex:1 1 auto!important;min-height:100vh!important;padding-bottom:48px!important}
}
/* Mobile-first caregiver: full-bleed phone shell matching mockup */
@media(max-width:768px){
  html,body{
    -webkit-overflow-scrolling:touch!important;overscroll-behavior-y:auto;
    touch-action:pan-y!important;position:static!important;margin:0!important;padding:0!important;
    background:#EEF2F6!important;overflow-x:hidden!important;max-width:100vw!important}
  .gradio-container,.gradio-container .main,.gradio-container .wrap,.gradio-container .app,
  .gradio-container .fillable,.gradio-container .contain,#root{
    max-width:100%!important;width:100%!important;margin:0!important;padding:0!important;
    border:0!important;border-radius:0!important;box-shadow:none!important;
    background:#EEF2F6!important;overflow-x:hidden!important}
  #care-page-wrap{min-height:100dvh!important;touch-action:pan-y;margin:0!important;padding:0!important;
    width:100%!important;max-width:100vw!important;overflow-x:hidden!important;background:#EEF2F6!important;
    border:0!important;border-radius:0!important;box-shadow:none!important}
  #care-page-wrap > .form,#care-page-wrap > div,
  #care-page-wrap .bubble-wrap,#care-page-wrap .block,
  #care-page-wrap .html-container,#care-page-wrap .prose,
  #cv-layout,#careview-shell,#careview-shell .block,#careview-shell .form{
    flex:none!important;width:100%!important;max-width:100%!important;min-width:0!important;margin:0!important;
    padding:0!important;border:0!important;border-radius:0!important;box-shadow:none!important;
    background:transparent!important;box-sizing:border-box!important;overflow-x:hidden!important}
  #cv-layout{display:flex!important;flex-direction:column!important;flex-wrap:nowrap!important;
    align-items:stretch!important;min-height:0!important;width:100%!important;max-width:100vw!important}
  #cv-layout > *{min-width:0!important;max-width:100%!important;box-sizing:border-box!important}
  /* Kill Gradio HTML/prose padding that pushes cards off-screen */
  #careview-shell .prose,#careview-shell .markdown-body,#careview-shell .html-container,
  #careview-shell .gap,#careview-shell .wrap,#careview-shell .panel-full,
  #careview-shell .panel-wrapper,#careview-shell .svelte-frame{
    padding:0!important;margin:0!important;max-width:100%!important;min-width:0!important;
    overflow-x:hidden!important;border:0!important;box-shadow:none!important;background:transparent!important}
  #careview-shell .prose #careview-root,#careview-shell .prose .cv-main{
    max-width:100%!important;overflow-x:hidden!important;box-sizing:border-box!important}
  /* Header: edge-to-edge navy like mockup */
  #care-page-wrap .html-container:has(#cv-mobile-top),
  #care-page-wrap .prose:has(#cv-mobile-top),
  #care-page-wrap div:has(> #cv-mobile-top){
    padding:0!important;margin:0!important;border:0!important;border-radius:0!important;
    background:transparent!important;max-width:100%!important}
  #cv-mobile-top{display:flex!important;align-items:center;justify-content:space-between;gap:10px;
    position:sticky;top:0;z-index:90;height:56px;padding:0 14px;background:#0B1F3A!important;color:#F8FAFC!important;
    border:0!important;border-radius:0!important;margin:0!important;width:100%!important;
    max-width:100vw!important;box-sizing:border-box}
  #cv-mobile-top .cv-m-brand{display:flex;align-items:center;gap:10px;font-weight:750;font-size:16px;color:#fff!important}
  #cv-mobile-top .cv-m-mark{width:32px;height:32px;border-radius:10px;background:#2563EB;display:grid;place-items:center;color:#fff;font-weight:700}
  #cv-mobile-top .cv-m-meta{display:flex;align-items:center;gap:8px}
  #cv-mobile-top .cv-m-bell{width:36px;height:36px;border-radius:999px;background:transparent;display:grid;place-items:center;color:#FDE68A;font-size:18px;cursor:pointer;border:0;padding:0}
  #cv-mobile-top .cv-m-user{display:none!important}
  /* Slim fixed bottom dock — out of document flow so it cannot widen the page */
  #cv-gradio-rail{position:fixed!important;left:0!important;right:0!important;bottom:0!important;top:auto!important;
    width:100%!important;max-width:100vw!important;min-width:0!important;flex:0 0 0!important;
    height:56px!important;max-height:56px!important;margin:0!important;
    z-index:95!important;padding:4px 6px calc(4px + env(safe-area-inset-bottom,0px))!important;
    background:#0B1F3A!important;border:0!important;border-top:1px solid rgba(255,255,255,.08)!important;
    border-radius:0!important;box-shadow:none!important;box-sizing:border-box!important;overflow:visible!important}
  #cv-gradio-rail .cv-brand,#cv-gradio-rail .cv-nav-foot{display:none!important}
  #cv-side-nav,#cv-side-nav .form,#cv-side-nav .block,#cv-side-nav .wrap{width:100%!important;max-width:100%!important;min-width:0!important}
  #cv-side-nav fieldset{flex-direction:row!important;justify-content:space-between!important;gap:0!important;width:100%!important}
  #cv-side-nav label{
    flex:1 1 0!important;min-width:0!important;min-height:44px!important;max-height:48px!important;
    justify-content:center!important;align-items:center!important;text-align:center!important;
    padding:6px 2px!important;font-size:11px!important;font-weight:700!important;letter-spacing:.02em;
    border-radius:12px!important;line-height:1.1!important;
    background:transparent!important;background-color:transparent!important;
    border:0!important;color:#94A3B8!important}
  #cv-side-nav label span,#cv-side-nav label *{color:inherit!important}
  #cv-side-nav label:has(input:checked){
    background:#2563EB!important;background-color:#2563EB!important;color:#FFFFFF!important}
  #cv-side-nav label:not(:has(input:checked)):hover{
    background:rgba(255,255,255,.06)!important;color:#E2E8F0!important}
  #careview-shell{width:100%!important;max-width:100%!important;min-width:0!important;padding:0!important;
    padding-bottom:calc(72px + env(safe-area-inset-bottom,0px))!important;
    min-height:0!important;height:auto!important;overflow-x:hidden!important;flex:1 1 auto!important;
    background:#EEF2F6!important;border-radius:0!important}
  #careview-shell .cv-main,.cv-main{
    padding:16px 14px 18px!important;width:100%!important;max-width:100%!important;min-width:0!important;
    box-sizing:border-box!important;overflow-x:hidden!important;background:#EEF2F6!important;min-height:0!important}
  #careview-shell #careview-root,#careview-shell #careview-root.cv-app{
    width:100%!important;max-width:100%!important;overflow-x:hidden!important;box-sizing:border-box!important}
  #careview-shell #careview-root.pd-page{padding-bottom:24px!important}
  #cv-live-panel{padding-bottom:12px!important}
  .cv-hello{font-size:22px!important;letter-spacing:-.4px;font-weight:750!important;color:#0F172A!important}
  .cv-sub{font-size:13px!important;margin-top:4px!important;margin-bottom:0!important;color:#64748B!important}
  .cv-top{flex-direction:column!important;align-items:stretch!important;gap:4px!important;margin-bottom:14px!important}
  .cv-top-right{display:none!important}
  .cv-stats,.cv-stats-compact{
    grid-template-columns:repeat(3,minmax(0,1fr))!important;gap:8px!important;margin-bottom:16px!important;
    width:100%!important;max-width:100%!important;box-sizing:border-box!important}
  .cv-stat{
    padding:10px 8px!important;min-height:0!important;border-radius:14px!important;gap:2px!important;
    box-shadow:none!important;width:auto!important;max-width:100%!important;box-sizing:border-box!important;
    flex-direction:column!important;align-items:center!important;text-align:center!important}
  .cv-stat-icon{display:none!important}
  .cv-stat-n{font-size:18px!important;font-weight:750!important}
  .cv-stat-safe .cv-stat-n{color:#16A34A!important}
  .cv-stat-att .cv-stat-n{color:#CA8A04!important}
  .cv-stat-alert .cv-stat-n{color:#DC2626!important}
  .cv-stat-l{font-size:11px!important;font-weight:650!important;color:#64748B!important}
  .cv-section-head{margin:4px 0 10px!important}
  .cv-section-head h2{font-size:15px!important;font-weight:750!important}
  .cv-patients{grid-template-columns:1fr!important;gap:10px!important;width:100%!important;max-width:100%!important}
  .cv-patient-card{
    min-height:0!important;padding:12px!important;border-radius:16px!important;
    width:100%!important;max-width:100%!important;box-sizing:border-box!important;overflow:hidden!important}
  .cv-patient-row{
    min-height:0!important;padding:0!important;width:100%!important;max-width:100%!important;
    box-sizing:border-box!important;overflow:hidden!important}
  .cv-patient-who{flex:1 1 0!important;min-width:0!important}
  .cv-pill{flex:0 0 auto!important}
  .cv-open-link{margin-top:10px!important;font-size:12px!important;font-weight:700!important;color:#2563EB!important}
  .cv-alerts{border-radius:16px!important;width:100%!important;max-width:100%!important;box-sizing:border-box!important;
    background:transparent!important;border:0!important;box-shadow:none!important;overflow:visible!important}
  .cv-alert-empty{margin:0!important}
  .cv-alert-row{padding:12px!important;align-items:flex-start!important;flex-wrap:wrap;
    background:#fff!important;border:1px solid #E2E8F0!important;border-radius:16px!important;margin-bottom:10px}
  .cv-alert-open{margin-left:0!important;margin-top:8px}
  .cv-tools{padding:10px 14px 24px!important;width:100%!important;max-width:100%!important;box-sizing:border-box!important}
  .cv-status-note{display:none!important}
  #careview-shell .cv-exit button,#careview-shell button{min-height:44px!important}
  #careview-shell #careview-root.pd-page{padding:12px 12px 8px!important}
  #careview-shell #careview-root.pd-page .pd-name{font-size:22px!important}
  #careview-shell #careview-root.pd-page .pd-meta{grid-template-columns:1fr!important}
  #careview-shell #careview-root.pd-page .pd-grid3{grid-template-columns:1fr!important}
  #cv-case-preview img, #careview-shell .large-viewer img, #careview-shell .cv-live-image img{
    max-width:100%!important;width:100%!important;height:auto!important;aspect-ratio:4/3;object-fit:contain}
  #cv-live-panel{padding:12px 14px 0!important}
  .cv-lab-link{display:none!important}
  .cv-hist2-grid{display:flex!important;flex-direction:column!important}
  .cv-hist2-list-panel{order:1}
  .cv-hist2-detail-panel{order:2}
}
#cv-live-panel{padding:12px 22px 10px;background:#F1F5F9}
#cv-live-panel .cv-live-image{border-radius:14px;overflow:hidden}
#cv-live-panel .cv-live-empty-compact{min-height:56px!important;max-height:88px!important;padding:12px 14px!important;
  background:#F8FAFC!important;border:1px solid #E2E8F0!important;color:#64748B!important}
/* P4 Alert Center — compact chips, list first */
.cv-ac-chips{display:flex;flex-wrap:nowrap;gap:8px;overflow-x:auto;margin:0 0 12px;padding-bottom:2px;-webkit-overflow-scrolling:touch}
.cv-ac-chip{flex:0 0 auto;background:#fff;border:1px solid #E2E8F0;border-radius:999px;padding:7px 12px;font-size:12px;font-weight:650;color:#64748B}
.cv-ac-chip b{color:#0F172A;font-weight:750;margin-right:4px}
.cv-ac-chip-high{border-color:#FECACA;background:#FEF2F2;color:#B91C1C}
.cv-ac-chip-high b{color:#DC2626}
.cv-ac-chip-med{border-color:#FDE68A;background:#FFFBEB;color:#A16207}
.cv-ac-chip-med b{color:#A16207}
.cv-ac-stats{display:none}
.cv-ac-note{display:none}
.cv-ac-list{display:flex;flex-direction:column;gap:10px}
.cv-ac-card{background:#fff;border:1px solid #E2E8F0;border-radius:16px;overflow:hidden;box-shadow:0 1px 2px rgba(15,23,42,.04)}
.cv-ac-summary{list-style:none;display:flex;gap:12px;align-items:flex-start;padding:12px 14px;cursor:pointer}
.cv-ac-summary::-webkit-details-marker{display:none}
.cv-ac-thumb{width:56px;height:56px;border-radius:12px;object-fit:cover;flex:0 0 56px;background:#FEE2E2;color:#B91C1C;display:grid;place-items:center;font-size:11px;font-weight:700}
.cv-ac-thumb-empty{display:grid}
.cv-ac-main{min-width:0;flex:1}
.cv-ac-top{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.cv-ac-name{font-weight:700;color:#0F172A;font-size:14px}
.cv-ac-room{font-size:12px;color:#64748B}
.cv-pri,.cv-st{font-size:10px;font-weight:750;padding:3px 8px;border-radius:999px;letter-spacing:.2px}
.cv-pri-high{background:#FEE2E2;color:#B91C1C}
.cv-pri-med{background:#FEF9C3;color:#A16207}
.cv-pri-low{background:#DBEAFE;color:#1D4ED8}
.cv-pri-unk{background:#F1F5F9;color:#475569}
.cv-st-open{background:#FFEDD5;color:#C2410C}
.cv-st-ack{background:#E0E7FF;color:#4338CA}
.cv-ac-title{margin-top:6px;font-size:13px;font-weight:650;color:#0F172A}
.cv-ac-sub,.cv-ac-brief,.cv-ac-next{margin-top:3px;font-size:12px;color:#64748B;line-height:1.4}
.cv-ac-brief b,.cv-ac-next b{color:#334155}
.cv-ac-count{font-size:10px;font-weight:750;padding:3px 8px;border-radius:999px;background:#F1F5F9;color:#475569}
.cv-needs-check{background:#7F1D1D;color:#FEE2E2;border-radius:999px;font-size:10px;font-weight:750;
  padding:3px 8px;letter-spacing:.02em}
.cv-ac-detail{padding:0 14px 14px 82px;border-top:1px solid #F1F5F9;color:#334155;font-size:12px;line-height:1.5}
.cv-ac-detail > div{margin-top:6px}
.cv-ac-findings-head{margin-top:10px!important;font-size:12px;color:#0F172A}
.cv-ac-finding{display:flex;gap:10px;align-items:flex-start;margin-top:10px;padding:10px;border:1px solid #E2E8F0;
  border-radius:12px;background:#F8FAFC}
.cv-ac-thumb-sm{width:40px;height:40px;flex:0 0 40px;border-radius:10px;font-size:10px}
.cv-cr-findings{margin:8px 0 0;padding-left:18px;font-size:12px;color:#334155;line-height:1.45}
.cv-ac-actions{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-top:10px}
.cv-ac-open-patient{appearance:none;border:1px solid #2563EB;background:#2563EB;color:#fff!important;border-radius:10px;
  padding:8px 12px;font-size:12px;font-weight:700;cursor:pointer!important;pointer-events:auto!important;
  display:inline-block;text-decoration:none!important;line-height:1.2}
.cv-ac-open-patient:hover{background:#1D4ED8!important;border-color:#1D4ED8!important}
/* Fall alert popup — loops sound until Silence alarm */
.cv-urgent-modal{position:fixed;inset:0;z-index:9999;display:flex;align-items:center;justify-content:center;
  padding:18px;pointer-events:auto}
.cv-urgent-modal-backdrop{position:absolute;inset:0;background:rgba(15,23,42,.62)}
.cv-urgent-modal-card{position:relative;z-index:1;width:min(420px,100%);padding:20px 18px 16px;border-radius:18px;
  background:#7F1D1D;color:#FEE2E2;border:1px solid #B91C1C;box-shadow:0 18px 48px rgba(127,29,29,.55);
  animation:cvUrgentPulse 1.4s ease-in-out infinite}
.cv-urgent-modal.cv-urgent-silenced{display:none!important}
@keyframes cvUrgentPulse{0%,100%{transform:scale(1)}50%{transform:scale(1.015);box-shadow:0 0 0 6px rgba(248,113,113,.28)}}
.cv-urgent-kicker{font-size:11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#FECACA}
.cv-urgent-title{margin-top:8px;font-size:20px;font-weight:750;color:#fff;line-height:1.25}
.cv-urgent-meta{margin-top:6px;font-size:13px;color:#FEE2E2}
.cv-urgent-reason{margin-top:10px;font-size:14px;line-height:1.45;color:#FFE4E6}
.cv-urgent-actions{margin-top:16px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
.cv-urgent-silence{appearance:none;border:1px solid #fff;background:#fff;color:#7F1D1D;border-radius:10px;
  padding:10px 14px;font-size:13px;font-weight:750;cursor:pointer!important}
.cv-urgent-silence:hover{background:#FEE2E2}
.cv-urgent-actions .cv-urgent-open,.cv-urgent-actions .cv-ac-open-patient{
  background:transparent;color:#fff!important;border:1px solid #FECACA;border-radius:10px;
  padding:10px 14px;font-size:13px;font-weight:750;text-decoration:none!important}
.cv-urgent-hint{margin-top:12px;font-size:11px;line-height:1.4;color:#FECACA}
@media(max-width:768px){
  .cv-urgent-modal{padding:12px;align-items:flex-end}
  .cv-urgent-modal-card{border-radius:16px 16px 12px 12px}
}
/* In-page snapshot lightbox — avoids blank data: URL tabs */
.cv-lightbox{position:fixed;inset:0;z-index:10000;display:none;align-items:center;justify-content:center;
  padding:16px;background:rgba(15,23,42,.82)}
.cv-lightbox.is-open{display:flex}
.cv-lightbox img{max-width:min(960px,100%);max-height:min(90vh,100%);border-radius:12px;
  box-shadow:0 18px 48px rgba(0,0,0,.45);background:#0F172A}
.cv-lightbox-close{position:absolute;top:12px;right:12px;appearance:none;border:1px solid #E2E8F0;
  background:#fff;color:#0F172A;border-radius:999px;padding:8px 12px;font-size:13px;font-weight:700;
  cursor:pointer!important}
/* Incident timeline under Watch recording / History video */
.cv-incident-bar{margin-top:10px}
.cv-incident-row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;justify-content:space-between}
.cv-incident-chip{display:inline-flex;align-items:center;padding:6px 10px;border-radius:999px;
  background:#F1F5F9;color:#0F172A;font-size:12px;font-weight:700;line-height:1.2}
.cv-incident-chip.cv-incident-fall{background:#FEF3C7;color:#92400E;border:1px solid #F59E0B}
.cv-incident-jump{appearance:none;border:1px solid #F59E0B;background:#FFFBEB;color:#92400E;
  border-radius:10px;padding:7px 12px;font-size:12px;font-weight:750;cursor:pointer!important}
.cv-incident-jump:hover{background:#FDE68A}
.cv-incident-track{position:relative;margin-top:10px;height:14px;border-radius:999px;background:#E2E8F0;
  overflow:visible}
.cv-incident-mark{appearance:none;position:absolute;top:50%;transform:translate(-50%,-50%);
  width:10px;height:18px;border:0;border-radius:3px;padding:0;background:#94A3B8;
  box-shadow:0 0 0 3px rgba(148,163,184,.25);cursor:pointer!important;z-index:2}
.cv-incident-mark-fall{background:#F59E0B;box-shadow:0 0 0 4px rgba(245,158,11,.35)}
.cv-incident-mark-fall::before{content:"";position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
  width:18px;height:100%;border-radius:4px;background:rgba(245,158,11,.35);pointer-events:none}
.cv-incident-unavailable .cv-incident-chip{background:#F8FAFC;color:#64748B;font-weight:650}
@media(max-width:768px){
  .cv-ac-detail{padding-left:14px}
  .cv-hist2-grid{display:flex!important;flex-direction:column!important}
  .cv-hist2-list-panel{order:1}
  .cv-hist2-detail-panel{order:2}
}
@media(max-width:1100px) and (min-width:769px){
  .cv-patients{grid-template-columns:1fr 1fr}
  .cv-main{padding:18px}
}
"""


# Click My Patients / Recent Alerts / Alert Center HTML → hidden Gradio buttons.
# Force document scroll on every viewport (Gradio wrappers often set overflow:hidden).
CV_BOARD_CLICK_JS = """
() => {
  const unlockPageScroll = () => {
    const html = document.documentElement;
    const body = document.body;
    [html, body].forEach((el) => {
      if (!el) return;
      el.style.setProperty('overflow-y', 'scroll', 'important');
      el.style.setProperty('overflow-x', 'hidden', 'important');
      el.style.setProperty('height', 'auto', 'important');
      el.style.setProperty('max-height', 'none', 'important');
      el.style.setProperty('position', 'static', 'important');
    });
    const wrap = document.getElementById('care-page-wrap');
    if (wrap) {
      wrap.style.setProperty('position', 'relative', 'important');
      wrap.style.setProperty('inset', 'auto', 'important');
      wrap.style.setProperty('top', 'auto', 'important');
      wrap.style.setProperty('left', 'auto', 'important');
      wrap.style.setProperty('right', 'auto', 'important');
      wrap.style.setProperty('bottom', 'auto', 'important');
      wrap.style.setProperty('height', 'auto', 'important');
      wrap.style.setProperty('max-height', 'none', 'important');
      wrap.style.setProperty('overflow-x', 'hidden', 'important');
      wrap.style.setProperty('max-width', '100vw', 'important');
      wrap.style.setProperty('padding', '0', 'important');
      wrap.style.setProperty('margin', '0', 'important');
    }
    document.querySelectorAll(
      '.gradio-container, main, .main, .wrap, .app, .fillable, .contain, #root'
    ).forEach((el) => {
      el.style.setProperty('overflow-x', 'hidden', 'important');
      el.style.setProperty('height', 'auto', 'important');
      el.style.setProperty('max-height', 'none', 'important');
      el.style.setProperty('padding', '0', 'important');
      el.style.setProperty('margin', '0', 'important');
      el.style.setProperty('max-width', '100%', 'important');
    });
    // Phone: Gradio Column min-width pushes patient pills off-screen.
    if (window.matchMedia('(max-width: 768px)').matches) {
      document.querySelectorAll('#cv-layout, #cv-layout > *, #careview-shell, #careview-shell *').forEach((el) => {
        if (el && el.style) {
          el.style.setProperty('min-width', '0', 'important');
        }
      });
      const shell = document.getElementById('careview-shell');
      if (shell) {
        shell.style.setProperty('width', '100%', 'important');
        shell.style.setProperty('max-width', '100%', 'important');
      }
      const rail = document.getElementById('cv-gradio-rail');
      if (rail) {
        rail.style.setProperty('min-width', '0', 'important');
        rail.style.setProperty('max-width', '100vw', 'important');
        rail.style.setProperty('width', '100%', 'important');
      }
    }
  };
  unlockPageScroll();
  window.addEventListener('resize', unlockPageScroll, { passive: true });
  // Debounced — every Gradio HTML refresh used to thrash layout.
  let unlockTimer = null;
  const mo = new MutationObserver(() => {
    if (unlockTimer) return;
    unlockTimer = setTimeout(() => { unlockTimer = null; unlockPageScroll(); }, 120);
  });
  mo.observe(document.documentElement, { childList: true, subtree: true });

  // Bind once on document (survives Gradio remounts of #care-page-wrap).
  if (document.documentElement.dataset.cvClickBound === '1') return [];
  document.documentElement.dataset.cvClickBound = '1';

  const clickHost = (sel) => {
    const host = document.querySelector(sel);
    if (!host) return false;
    const btn = host.matches('button') ? host : host.querySelector('button');
    (btn || host).click();
    return true;
  };
  const setNativeValue = (input, val) => {
    if (!input) return;
    const proto = input.tagName === 'TEXTAREA'
      ? window.HTMLTextAreaElement.prototype
      : window.HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
    setter.call(input, String(val));
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
  };
  const setEventId = (val) => {
    const host = document.querySelector('#cv-hidden-eid');
    if (!host) return;
    setNativeValue(host.querySelector('input'), val);
  };
  const setAlertBundle = (token) => {
    // Select the Care Response dropdown so Resolve uses the full reviewed patient bundle.
    if (!token) return;
    const root = document.querySelector('#careview-shell') || document;
    const hosts = root.querySelectorAll('.cv-cr-event, label');
    let input = null;
    root.querySelectorAll('input, textarea, select').forEach((el) => {
      const wrap = el.closest('.cv-cr-event');
      if (wrap) input = el;
    });
    // Gradio Dropdown often uses a hidden input + visible textbox.
    const dd = root.querySelector('.cv-cr-event input, .cv-cr-event textarea');
    if (dd) {
      setNativeValue(dd, token);
      // Also try selecting matching <option> if present.
      const sel = root.querySelector('.cv-cr-event select');
      if (sel) {
        sel.value = String(token);
        sel.dispatchEvent(new Event('change', { bubbles: true }));
      }
      return;
    }
    if (input) setNativeValue(input, token);
  };
  const setOpenPatientId = (val) => {
    const host = document.querySelector('#cv-open-pid');
    if (!host) return;
    setNativeValue(host.querySelector('input, textarea'), val);
  };
  const clickSoon = (sel, tries) => {
    const n = (tries == null) ? 8 : tries;
    setTimeout(() => {
      if (clickHost(sel)) return;
      if (n > 1) clickSoon(sel, n - 1);
    }, 40);
  };
  const openPatient = (id) => {
    if (!id) return;
    setOpenPatientId(id);
    clickSoon('#cv-btn-open-patient');
  };
  const openAlertEvent = (eid, pid, bundle) => {
    if (bundle) setAlertBundle(bundle);
    if (eid) setEventId(eid);
    if (pid) setOpenPatientId(pid);
    // Review opens the patient case focused on this alert (most reliable path).
    if (eid) clickSoon('#cv-btn-review');
    else openPatient(pid);
  };

  // Endless fall alarm until caregiver presses Silence (localStorage per event id).
  const SILENCE_KEY = 'cv_alert_silenced';
  let alarmTimer = null;
  let alarmCtx = null;
  let alarmEventId = '';

  const readSilenced = () => {
    try { return JSON.parse(localStorage.getItem(SILENCE_KEY) || '{}') || {}; }
    catch (e) { return {}; }
  };
  const isSilenced = (eid) => {
    if (!eid) return false;
    const map = readSilenced();
    const until = Number(map[String(eid)] || 0);
    return until > Date.now();
  };
  const silenceEvent = (eid) => {
    if (!eid) return;
    const map = readSilenced();
    // Keep silence for 12h so reopen/poll does not restart the same alert's sound.
    map[String(eid)] = Date.now() + 12 * 60 * 60 * 1000;
    try { localStorage.setItem(SILENCE_KEY, JSON.stringify(map)); } catch (e) {}
  };
  const stopAlarm = () => {
    if (alarmTimer) { clearInterval(alarmTimer); alarmTimer = null; }
    if (alarmCtx) {
      try { alarmCtx.close(); } catch (e) {}
      alarmCtx = null;
    }
    alarmEventId = '';
    try { if (navigator.vibrate) navigator.vibrate(0); } catch (e) {}
  };
  const playAlarmBurst = () => {
    try {
      if (navigator.vibrate) navigator.vibrate([180, 90, 180, 90, 180, 90, 180]);
    } catch (e) {}
    try {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return;
      if (!alarmCtx || alarmCtx.state === 'closed') {
        alarmCtx = new Ctx();
      }
      if (alarmCtx.state === 'suspended' && alarmCtx.resume) alarmCtx.resume();
      const ctx = alarmCtx;
      const beepAt = (startSec, freq) => {
        const o = ctx.createOscillator();
        const g = ctx.createGain();
        o.type = 'square';
        o.frequency.value = freq;
        g.gain.value = 0.07;
        o.connect(g); g.connect(ctx.destination);
        o.start(ctx.currentTime + startSec);
        o.stop(ctx.currentTime + startSec + 0.2);
      };
      beepAt(0, 880);
      beepAt(0.28, 990);
      beepAt(0.56, 880);
      beepAt(0.84, 990);
    } catch (e) {}
  };
  const startAlarmLoop = (eid) => {
    if (!eid || isSilenced(eid)) {
      stopAlarm();
      return;
    }
    if (alarmEventId === eid && alarmTimer) return;
    stopAlarm();
    alarmEventId = eid;
    playAlarmBurst();
    alarmTimer = setInterval(() => {
      if (isSilenced(eid)) { stopAlarm(); return; }
      const still = document.querySelector(
        '#cv-urgent-watch .cv-urgent-modal[data-alert-loop=\"1\"][data-event-id=\"' + eid + '\"]'
      );
      if (!still) { stopAlarm(); return; }
      playAlarmBurst();
    }, 1600);
  };
  const dismissUrgentModal = (root, eid) => {
    if (eid) silenceEvent(eid);
    stopAlarm();
    const modal = root || document.querySelector('#cv-urgent-watch .cv-urgent-modal');
    if (modal) modal.classList.add('cv-urgent-silenced');
  };
  const applySilenceUi = (root) => {
    if (!root) return;
    const eid = root.getAttribute('data-event-id') || '';
    if (isSilenced(eid)) root.classList.add('cv-urgent-silenced');
    else root.classList.remove('cv-urgent-silenced');
  };
  const watchUrgent = () => {
    const el = document.querySelector('#cv-urgent-watch .cv-urgent-modal[data-alert-loop=\"1\"]');
    if (!el) {
      stopAlarm();
      return;
    }
    applySilenceUi(el);
    const eid = el.getAttribute('data-event-id') || '';
    if (!eid || isSilenced(eid)) {
      stopAlarm();
      return;
    }
    startAlarmLoop(eid);
  };
  const urgentMo = new MutationObserver(() => watchUrgent());
  urgentMo.observe(document.documentElement, { childList: true, subtree: true });
  // Unlock audio on first tap (required by iOS / Messenger WebView).
  document.addEventListener('pointerdown', () => {
    try {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return;
      if (!alarmCtx || alarmCtx.state === 'closed') alarmCtx = new Ctx();
      if (alarmCtx.resume) alarmCtx.resume();
    } catch (e) {}
    watchUrgent();
  }, { passive: true });
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.cv-urgent-silence');
    if (!btn) return;
    e.preventDefault();
    e.stopPropagation();
    const modal = btn.closest('.cv-urgent-modal');
    const eid = btn.getAttribute('data-event-id')
      || (modal && modal.getAttribute('data-event-id'))
      || '';
    dismissUrgentModal(modal, eid);
  }, true);

  // Mobile header bell → Alerts tab
  document.addEventListener('click', (e) => {
    const bell = e.target.closest('#cv-m-bell, .cv-m-bell');
    if (!bell) return;
    e.preventDefault();
    const labels = document.querySelectorAll('#cv-side-nav label');
    for (const lab of labels) {
      const t = (lab.textContent || '').trim();
      if (t === 'Alerts' || t.indexOf('Alerts') === 0) {
        lab.click();
        break;
      }
    }
  }, true);

  // Snapshot enlarge — in-page lightbox (data: URL tabs go blank on phones).
  const ensureLightbox = () => {
    let box = document.getElementById('cv-lightbox');
    if (box) return box;
    box = document.createElement('div');
    box.id = 'cv-lightbox';
    box.className = 'cv-lightbox';
    box.innerHTML = '<button type="button" class="cv-lightbox-close" aria-label="Close">Close</button>'
      + '<img alt="Alert snapshot">';
    document.body.appendChild(box);
    box.addEventListener('click', (ev) => {
      if (ev.target === box || ev.target.closest('.cv-lightbox-close')) {
        box.classList.remove('is-open');
        const img = box.querySelector('img');
        if (img) img.removeAttribute('src');
      }
    });
    return box;
  };
  const openSnapLightbox = (src) => {
    if (!src) return;
    const box = ensureLightbox();
    const img = box.querySelector('img');
    if (img) img.src = src;
    box.classList.add('is-open');
  };
  document.addEventListener('click', (e) => {
    const zoom = e.target.closest('.cv-snap-zoom');
    if (!zoom) return;
    e.preventDefault();
    e.stopPropagation();
    const img = zoom.querySelector('img') || (zoom.tagName === 'IMG' ? zoom : null);
    const src = img && img.getAttribute('src');
    openSnapLightbox(src);
  }, true);

  // Fall / incident timeline: yellow tick + Jump seeks the native <video>.
  const findIncidentVideo = (bar) => {
    if (!bar) return null;
    const panel = bar.closest('.pd-player-panel, .cv-hist2-player, .cv-hist2-card');
    return (panel && panel.querySelector('video')) || null;
  };
  const syncIncidentMark = (bar, duration) => {
    if (!bar || !(duration > 0)) return;
    const offset = parseFloat(bar.getAttribute('data-offset') || '');
    if (!Number.isFinite(offset)) return;
    const pct = Math.max(0, Math.min(100, (offset / duration) * 100));
    bar.setAttribute('data-duration', String(duration));
    bar.setAttribute('data-pct', pct.toFixed(2));
    const mark = bar.querySelector('.cv-incident-mark');
    if (mark) mark.style.left = pct.toFixed(2) + '%';
  };
  const seekIncident = (bar, useJump) => {
    const video = findIncidentVideo(bar);
    if (!video || !bar) return;
    const offset = parseFloat(bar.getAttribute('data-offset') || '');
    const jump = parseFloat(bar.getAttribute('data-jump') || '');
    if (!Number.isFinite(offset)) return;
    let t = (useJump && Number.isFinite(jump)) ? jump : offset;
    const go = () => {
      const dur = Number(video.duration);
      if (Number.isFinite(dur) && dur > 0) {
        syncIncidentMark(bar, dur);
        t = Math.max(0, Math.min(t, dur));
      }
      try { video.currentTime = t; } catch (err) {}
    };
    if (video.readyState >= 1 && Number.isFinite(video.duration)) go();
    else video.addEventListener('loadedmetadata', go, { once: true });
  };
  const wireIncidentBars = () => {
    document.querySelectorAll('.cv-incident-bar[data-offset]').forEach((bar) => {
      if (bar.getAttribute('data-wired') === '1') return;
      bar.setAttribute('data-wired', '1');
      const video = findIncidentVideo(bar);
      if (!video) return;
      const onMeta = () => {
        if (Number.isFinite(video.duration) && video.duration > 0) {
          syncIncidentMark(bar, video.duration);
        }
      };
      video.addEventListener('loadedmetadata', onMeta);
      if (video.readyState >= 1) onMeta();
    });
  };
  document.addEventListener('click', (e) => {
    const jumpBtn = e.target.closest('.cv-incident-jump');
    const markBtn = e.target.closest('.cv-incident-mark');
    if (!jumpBtn && !markBtn) return;
    e.preventDefault();
    e.stopPropagation();
    const bar = (jumpBtn || markBtn).closest('.cv-incident-bar');
    seekIncident(bar, !!jumpBtn);
  }, true);
  const incidentMo = new MutationObserver(() => wireIncidentBars());
  incidentMo.observe(document.documentElement, { childList: true, subtree: true });
  wireIncidentBars();

  document.addEventListener('click', (e) => {
    const inApp = e.target.closest(
      '#care-page-wrap, #careview-shell, #careview-root, .cv-ac-card, .cv-patient-card, .cv-urgent-modal'
    );
    if (!inApp) return;

    const openBtn = e.target.closest('.cv-ac-open-patient');
    if (openBtn) {
      e.preventDefault();
      e.stopPropagation();
      const modal = openBtn.closest('.cv-urgent-modal');
      const card = openBtn.closest('.cv-ac-card');
      const eid = openBtn.getAttribute('data-event-id')
        || (modal && modal.getAttribute('data-event-id'))
        || (card && card.getAttribute('data-event-id'));
      const pid = openBtn.getAttribute('data-patient-id')
        || (modal && modal.getAttribute('data-patient-id'))
        || (card && card.getAttribute('data-patient-id'));
      const bundle = card && card.getAttribute('data-bundle');
      if (modal) dismissUrgentModal(modal, eid);
      openAlertEvent(eid, pid, bundle);
      return;
    }

    // Alert Center card: select the full patient bundle for one-shot Resolve.
    const acCard = e.target.closest('.cv-ac-card[data-event-id]');
    if (acCard && !e.target.closest('a, button, .cv-ac-open-patient')) {
      const bundle = acCard.getAttribute('data-bundle');
      if (bundle) setAlertBundle(bundle);
      setEventId(acCard.getAttribute('data-event-id'));
      return;
    }

    const card = e.target.closest('.cv-patient-card[data-patient-id], .cv-clickable[data-patient-id]');
    if (card) {
      openPatient(card.getAttribute('data-patient-id'));
      return;
    }

    // History: select event → show summary + snapshots (no icons).
    const histRow = e.target.closest('.cv-hist2-row[data-hist-select]');
    if (histRow) {
      e.preventDefault();
      e.stopPropagation();
      const eid = histRow.getAttribute('data-hist-select') || '';
      const root = histRow.closest('#cv-hist2') || document.getElementById('cv-hist2');
      if (root && eid) {
        root.setAttribute('data-hist-active', eid);
        root.querySelectorAll('.cv-hist2-row').forEach((r) => {
          r.classList.toggle('is-selected', r.getAttribute('data-hist-select') === eid);
        });
        root.querySelectorAll('.cv-hist2-detail').forEach((d) => {
          d.classList.toggle('is-active', d.getAttribute('data-hist-detail') === eid);
        });
      }
      if (eid) setEventId(eid);
      return;
    }

    const row = e.target.closest('.cv-alert-row[data-event-id]');
    if (row) {
      openAlertEvent(
        row.getAttribute('data-event-id'),
        row.getAttribute('data-patient-id'),
        row.getAttribute('data-bundle'),
      );
    }
  }, true);
  return [];
}
"""


# CareView is the product home. Lab tools stay available under the exit button.
PAGE_CHOICES = [
    (chr(0x2665) + "  CareView", "Care Home"),
    (chr(0x25c9) + "  Live Camera", "Live Camera"),
    (chr(0x25b6) + "  Video Analysis", "Video Analysis"),
    (chr(0x25a7) + "  Photo Analysis", "Photo Analysis"),
    (chr(0x2699) + "  About The Model", "About The Model"),
]
PAGE_NAMES = [value for _, value in PAGE_CHOICES]
CARE_PAGE = "Care Home"
LAB_PAGE = "Live Camera"


def show_page(page):
    return [gr.update(visible=page == name) for name in PAGE_NAMES]


def build_ui(
    analyse_video, analyse_image, webcam_step, stop_webcam,
    stop_camera_js, start_camera_js, examples_dir,
    acknowledge_event=None, resolve_event=None,
    care_login=None, care_refresh=None, care_room_check=None,
    care_ack=None, care_resolve=None, care_switch_panel=None,
    care_open_ana=None, care_open_pedro=None, care_open_maria=None,
    care_review_alert=None, care_back_to_patients=None, care_run_details=None,
    care_live_preview=None, care_live_status=None,
    care_initial_html: str | None = None,
    care_select_session=None,
):
    # Gradio 6+: theme/css belong on launch(), not Blocks().
    with gr.Blocks(title="CareView | Care Home Monitor") as demo:
        with gr.Row(elem_id="app-shell"):
            with gr.Column(elem_id="sidebar", scale=0, min_width=205):
                gr.HTML("""
                <div class="brand">
                  <div class="brand-mark">CV</div>
                  <div>Care<span>View</span><small>Lab Tools</small></div>
                </div>
                <div class="nav-caption">Workspace</div>
                """)
                nav = gr.Radio(
                    choices=PAGE_CHOICES, value=CARE_PAGE, label=None,
                    show_label=False, elem_id="sidebar-nav",
                )
                gr.HTML("""
                <div class="sidebar-tip">
                  <div class="sidebar-tip-heading"><span class="tip-icon" aria-hidden="true">&#10022;</span><strong>Ana live room</strong></div>
                  <p>Use Live Camera for Ana. Maria / Pedro room checks run from the CareView board.</p>
                </div>
                <div class="sidebar-foot">CareView home is the product.<br>These pages are CV lab tools.</div>
                """)
            with gr.Column(elem_id="main-panel"):
                gr.HTML("""
                <div class="topbar">
                  <div>CareView <span style="color:#55564f">/</span> <b>Lab Tools</b></div>
                  <div class="local-badge"><i></i>Models Loaded | Processing On This Computer</div>
                </div>
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
                    vmon = gr.Markdown("Monitoring status will appear after analysis.")
                    vevents = gr.Dataframe(
                        label="Recent Monitoring Events",
                        headers=["id", "time", "type", "status", "action", "snapshot"],
                        interactive=False,
                    )
                    with gr.Accordion("Technical Details", open=False):
                        vjson = gr.JSON(label="Processing Details")
                    vbtn.click(analyse_video, vin, [vout, vhead, vsum, vtl, vjson, vmon, vevents])
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
                    <div class="section-title"><div class="eyebrow">Camera Station · Room 01</div><h1>Live Camera — Ana</h1>
                    <p>This laptop page owns the webcam. Click <b>Start camera</b>, allow permission, then the orange <b>record</b> button if shown. Caregiver phones view the processed feed on Ana's CareView page — they never open this camera.</p></div>
                    """)
                    gr.Markdown(
                        "Analyzes ~4 frames/sec. Keep the camera stationary, full body, ~2–5 m away. "
                        "**Start camera** begins a persistent ACTIVE session; **Stop camera** finalizes it."
                    )
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
                    wmon = gr.Markdown("Monitoring status (SAFE / ATTENTION / ALERT) will appear while the camera runs.")
                    wsum = gr.Markdown("Session summary will appear here while the camera runs, and again when you stop.")
                    wevents = gr.Dataframe(
                        label="Recent Monitoring Events",
                        headers=["id", "time", "type", "status", "action", "snapshot"],
                        interactive=False,
                    )
                    with gr.Row():
                        weid = gr.Number(label="Event id", precision=0, value=None)
                        wack = gr.Button("Acknowledge event", elem_classes=["secondary"])
                        wres = gr.Button("Resolve event", elem_classes=["secondary"])
                    waction_note = gr.Markdown(
                        "Acknowledge = caregiver saw it (does **not** force SAFE). Resolve = close the event."
                    )
                    wstate = gr.State(None)
                    win.stream(
                        webcam_step, [win, wstate],
                        [wout, whead, wmon, wsum, wevents, wstate],
                        stream_every=0.25, concurrency_limit=1,
                    )
                    wstart.click(None, None, None, js=start_camera_js)
                    wstop.click(
                        stop_webcam, [wstate],
                        [win, wout, whead, wmon, wsum, wevents, wstate],
                        js=stop_camera_js,
                    )
                    if acknowledge_event is not None:
                        wack.click(acknowledge_event, weid, [wevents, waction_note])
                    if resolve_event is not None:
                        wres.click(resolve_event, weid, [wevents, waction_note])
                with gr.Group(visible=True, elem_id="care-page-wrap") as care_page:
                    with gr.Row(elem_id="cv-layout"):
                        with gr.Column(elem_id="cv-gradio-rail", scale=0, min_width=220):
                            gr.HTML("""
                            <div class="cv-brand">
                              <div class="cv-brand-mark">⌂</div>
                              <div><div class="cv-brand-name">CareView</div>
                              <div class="cv-brand-sub">Care Home Monitor</div></div>
                            </div>
                            """)
                            cnav = gr.Radio(
                                choices=["Home", "Alerts", "History", "Settings"],
                                value="Home",
                                label=None,
                                show_label=False,
                                elem_id="cv-side-nav",
                            )
                            gr.HTML('<div class="cv-nav-foot">Demo · caregiver / care123</div>')
                        with gr.Column(elem_id="careview-shell", scale=1):
                            # Board first — CareView mock design stays the hero.
                            _care_fallback = care_initial_html or (
                                '<div id="careview-root"><p style="padding:16px;color:#64748B">'
                                "Loading CareView…</p></div>"
                            )
                            ccards = gr.HTML(_care_fallback)
                            with gr.Group(elem_classes=["cv-tools"]):
                                # Hidden Gradio hooks — board cards/alerts click these via JS.
                                with gr.Row(elem_id="cv-hidden-hooks", elem_classes=["cv-hidden-hooks"]):
                                    cana = gr.Button("Ana", elem_id="cv-btn-ana")
                                    cpedro = gr.Button("Pedro", elem_id="cv-btn-pedro")
                                    cmaria = gr.Button("Maria", elem_id="cv-btn-maria")
                                    creview = gr.Button("Review", elem_id="cv-btn-review")
                                    ceid = gr.Number(
                                        label="Event id", precision=0, value=None, elem_id="cv-hidden-eid",
                                    )
                                cnote = gr.Markdown(
                                    "Click a **patient card** for that patient's details page, "
                                    "or a **Recent Alerts** row to open the related patient."
                                )
                                with gr.Group(visible=False) as cdetails_bar:
                                    with gr.Row(elem_classes=["cv-toolbar", "cv-exit"]):
                                        cback = gr.Button("← Back to patients", scale=1)
                                        crun = gr.Button("Run room check", variant="primary", scale=1)
                                    with gr.Group(elem_classes=["cv-play-block"]):
                                        gr.Markdown("### Issue recording")
                                        csession = gr.Dropdown(
                                            label="Issue recording",
                                            choices=[],
                                            value=None,
                                            interactive=False,
                                            visible=False,
                                            elem_id="cv-session-select",
                                        )
                                        cplayer = gr.HTML(
                                            '<div class="pd-player-panel"><div class="pd-empty">'
                                            "No issue recording to play yet.</div></div>",
                                            elem_id="cv-session-player",
                                        )
                                        cplay_info = gr.HTML(
                                            '<div class="pd-player-info pd-muted">'
                                            "The recording for this alert appears here when available.</div>"
                                        )
                                        csession_sel = gr.State(None)
                                with gr.Row(elem_classes=["media-row"]):
                                    with gr.Column(scale=1, min_width=280):
                                        cframe = gr.Image(
                                            label="Live feed / case preview",
                                            height=320,
                                            elem_classes=["large-viewer"],
                                            elem_id="cv-case-preview",
                                        )
                                    with gr.Column(scale=1, min_width=260):
                                        cmon = gr.Markdown("Status / next action")
                                # Poll shared live frames for Ana while her details page is open.
                                clive_timer = gr.Timer(1.25, active=True)
                                with gr.Row(elem_classes=["cv-toolbar", "cv-exit"]):
                                    cexit = gr.Button("Lab tools", scale=0, min_width=120)
                                    crefresh = gr.Button("Refresh board", scale=0, min_width=120)
                                    cack = gr.Button("Acknowledge", scale=0, min_width=120)
                                    cres = gr.Button("Resolve", scale=0, min_width=100)
                                cuser = gr.Textbox(value="caregiver", visible=False)
                                cpass = gr.Textbox(value="care123", type="password", visible=False)
                                clogin = gr.Button("Sign in", visible=False)
                                cauth = gr.State(None)
                                cmsg = gr.Markdown(visible=False)
                                cpatient = gr.State(None)
                                calert = gr.Dropdown(
                                    label="Patient events", choices=[], interactive=True,
                                )
                                with gr.Accordion("Events log", open=False):
                                    cinbox = gr.Dataframe(
                                        label="Events",
                                        headers=["id", "patient", "time", "type", "status", "action", "snapshot"],
                                        interactive=False,
                                    )
                                    caction_note = gr.Markdown(
                                        "Acknowledge = saw reviewed findings (does **not** force SAFE). "
                                        "Resolve = close those reviewed findings only — not future alerts."
                                    )
                    # open_patient_details (9) + playback panel (4): dropdown, player, info, selected id
                    detail_outs = [
                        ccards, cnote, cframe, cmon, cpatient, calert, ceid, cdetails_bar,
                        csession, cplayer, cplay_info, csession_sel,
                    ]
                    if care_login is not None:
                        clogin.click(
                            care_login, [cuser, cpass],
                            [cauth, cmsg, ccards, cinbox, cnav, calert],
                        )
                    if care_switch_panel is not None:
                        cnav.change(
                            care_switch_panel, [cauth, cnav],
                            [
                                ccards, cnote, cframe, cmon, cpatient, cdetails_bar,
                                csession, cplayer, cplay_info, csession_sel,
                            ],
                        )
                    if care_refresh is not None:
                        crefresh.click(
                            care_refresh, [cauth, cnav, cpatient],
                            [ccards, cinbox, calert],
                        )
                    if care_open_ana is not None:
                        cana.click(care_open_ana, [cauth, cnav], detail_outs)
                    if care_open_pedro is not None:
                        cpedro.click(care_open_pedro, [cauth, cnav], detail_outs)
                    if care_open_maria is not None:
                        cmaria.click(care_open_maria, [cauth, cnav], detail_outs)
                    if care_review_alert is not None:
                        creview.click(care_review_alert, [cauth, ceid, cnav], detail_outs)
                    if care_back_to_patients is not None:
                        cback.click(
                            care_back_to_patients, [cauth, cnav],
                            [
                                ccards, cnote, cframe, cmon, cpatient, calert, ceid,
                                cdetails_bar, cinbox,
                                csession, cplayer, cplay_info, csession_sel,
                            ],
                        )
                    if care_run_details is not None:
                        crun.click(
                            care_run_details, [cauth, cpatient, cnav],
                            [
                                ccards, cnote, cframe, cmon, cpatient, calert, ceid,
                                cdetails_bar, cinbox,
                                csession, cplayer, cplay_info, csession_sel,
                            ],
                        )
                    def _legacy_alert_to_eid(token):
                        from src.alert_center import parse_alert_bundle
                        primary, _ids = parse_alert_bundle(token)
                        return primary

                    calert.change(_legacy_alert_to_eid, calert, ceid)
                    if care_select_session is not None:
                        csession.change(
                            care_select_session, [cauth, cpatient, csession],
                            [cplayer, cplay_info, csession_sel],
                        )
                    if care_live_preview is not None:
                        clive_timer.tick(care_live_preview, [cpatient], [cframe])
                    if care_live_status is not None:
                        clive_timer.tick(care_live_status, [cpatient], [cmon])
                    if care_ack is not None:
                        cack.click(
                            care_ack, [cauth, calert, cnav, cpatient],
                            [cinbox, caction_note, ccards, calert],
                        )
                    if care_resolve is not None:
                        cres.click(
                            care_resolve, [cauth, calert, cnav, cpatient],
                            [cinbox, caction_note, ccards, calert, cnav],
                        )
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
                          <div class="activity-chip"><i></i><b>Fall Alert (extension)</b> | sudden drop / tilt into lying</div>
                          <div class="activity-chip"><i></i><b>Inactivity Alert</b> | little motion while still in view</div>
                          <div class="activity-chip"><i></i><b>Zone Warning</b> | out of view or partial body</div>
                          <div class="activity-chip"><i></i><b>Session Summary</b> | time by activity + alert counts</div>
                        </div>
                      </div>
                      <div class="panel"><h2>Getting A Useful Result</h2><div class="panel-sub">The system works best when the person is easy to see.</div>
                        <div class="pipe-step" style="margin-bottom:8px"><b>Camera Position</b><span>Keep the camera stationary.</span></div>
                        <div class="pipe-step" style="margin-bottom:8px"><b>Framing</b><span>Include the person's full body.</span></div>
                        <div class="pipe-step"><b>Distance And Lighting</b><span>Around 2-5 m away, in a well-lit room.</span></div>
                      </div>
                    </div>
                    <div class="section-note">Core labels are standing/sitting/walking/running/lying. Monitoring extensions add fall / inactive / zone banners and a session summary. Demo heuristics — not clinical devices.</div>
                    """)
        # Order must match PAGE_NAMES.
        pages = [care_page, live_page, video_page, photo_page, model_page]
        nav.change(show_page, inputs=nav, outputs=pages)
        cexit.click(lambda: LAB_PAGE, outputs=nav).then(show_page, inputs=nav, outputs=pages)
        # Bind board card/alert clicks, then auto-sign in.
        demo.load(None, None, None, js=CV_BOARD_CLICK_JS)
        if care_login is not None:
            demo.load(
                care_login, [cuser, cpass],
                [cauth, cmsg, ccards, cinbox, cnav, calert],
            )

    return demo


# Caregiver-only mount: light shell (no dark lab chrome). Reuses CareView CSS rules.
_CAREVIEWS_CSS = APP_CSS.split("/* ---- CareView product", 1)
CAREGIVER_APP_CSS = (
    "html,body,.gradio-container{background:#EEF2F6!important;color:#0F172A!important;"
    "color-scheme:light!important;"
    "--background-fill-primary:#FFFFFF!important;"
    "--background-fill-secondary:#F8FAFC!important;"
    "--block-background-fill:#FFFFFF!important;"
    "--body-background-fill:#EEF2F6!important;"
    "--body-text-color:#0F172A!important;"
    "--input-background-fill:#FFFFFF!important;"
    "--input-border-color:#E2E8F0!important;"
    "--border-color-primary:#E2E8F0!important;"
    "--neutral-950:#0F172A!important;--neutral-900:#1E293B!important;"
    "--neutral-800:#334155!important;--neutral-700:#475569!important;"
    "--neutral-600:#64748B!important;--neutral-500:#94A3B8!important;"
    "--neutral-400:#CBD5E1!important;--neutral-300:#E2E8F0!important;"
    "--neutral-200:#F1F5F9!important;--neutral-100:#F8FAFC!important;"
    "--neutral-50:#FFFFFF!important}"
    ".gradio-container,.gradio-container .contain,.gradio-container .wrap,"
    ".gradio-container .main,.gradio-container .app{max-width:none!important;"
    "padding:0!important;margin:0!important;border:0!important;border-radius:0!important;"
    "box-shadow:none!important;background:#EEF2F6!important}"
    # Global dropdown portals (listboxes often render under body, not inside cards).
    "body [role='listbox'],body ul[role='listbox'],body .options,body ul.options,"
    "body [class*='dropdown'] [role='listbox']{"
    "background:#FFFFFF!important;color:#0F172A!important;border:1px solid #E2E8F0!important;"
    "border-radius:12px!important;box-shadow:0 8px 24px rgba(15,23,42,.12)!important}"
    "body [role='option'],body .options li,body ul.options li{"
    "background:#FFFFFF!important;color:#0F172A!important;opacity:1!important}"
    "body [role='option'][aria-selected='true'],body .options li.selected{"
    "background:#EFF6FF!important;color:#1E40AF!important}"
    "body [role='option']:hover,body .options li:hover{"
    "background:#F1F5F9!important;color:#0F172A!important}"
    "/* ---- CareView product"
    + (_CAREVIEWS_CSS[1] if len(_CAREVIEWS_CSS) > 1 else "")
)


def build_caregiver_ui(
    care_login=None, care_refresh=None,
    care_ack=None, care_resolve=None, care_switch_panel=None,
    care_open_ana=None, care_open_pedro=None, care_open_maria=None,
    care_review_alert=None, care_back_to_patients=None, care_run_details=None,
    care_live_tick=None,
    care_initial_html: str | None = None,
    care_select_session=None,
    care_filter_alerts=None,
    care_open_patient=None,
    care_add_patient=None,
    care_load_demo_pack=None,
    care_alert_watch=None,
    care_response_preview=None,
):
    """Phone/desktop caregiver app — no Camera Station controls, no lab sidebar."""
    with gr.Blocks(title="CareView | Caregiver") as demo:
        with gr.Group(elem_id="care-page-wrap"):
            gr.HTML(
                """
                <div id="cv-mobile-top">
                  <div class="cv-m-brand"><div class="cv-m-mark">⌂</div>CareView</div>
                  <div class="cv-m-meta">
                    <button type="button" class="cv-m-bell" id="cv-m-bell" title="Alerts" aria-label="Open alerts">🔔</button>
                  </div>
                </div>
                """
            )
            with gr.Row(elem_id="cv-layout"):
                with gr.Column(elem_id="cv-gradio-rail", scale=0, min_width=220):
                    gr.HTML("""
                    <div class="cv-brand">
                      <div class="cv-brand-mark">⌂</div>
                      <div><div class="cv-brand-name">CareView</div>
                      <div class="cv-brand-sub">Care Home Monitor</div></div>
                    </div>
                    """)
                    cnav = gr.Radio(
                        choices=["Home", "Alerts", "History", "Settings"],
                        value="Home",
                        label=None,
                        show_label=False,
                        elem_id="cv-side-nav",
                    )
                    gr.HTML('<div class="cv-nav-foot">Demo · caregiver / care123</div>')
                with gr.Column(elem_id="careview-shell", scale=1):
                    # Sticky interrupt when OPEN alerts fire (even while watching live).
                    # Textbox (not State) so Timer reliably round-trips last seen id.
                    calert_watch = gr.HTML("", elem_id="cv-urgent-watch")
                    calert_seen = gr.Textbox(
                        value="0", visible=False, elem_id="cv-alert-seen-id",
                    )
                    # Patient details FIRST so Alert "Open" shows the case, not a wall of offline camera.
                    _care_fallback = care_initial_html or (
                        '<div id="careview-root"><p style="padding:16px;color:#64748B">'
                        "Loading CareView…</p></div>"
                    )
                    ccards = gr.HTML(_care_fallback)
                    with gr.Group(elem_classes=["cv-tools"]):
                        with gr.Row(elem_id="cv-hidden-hooks", elem_classes=["cv-hidden-hooks"]):
                            cana = gr.Button("Ana", elem_id="cv-btn-ana")
                            cpedro = gr.Button("Pedro", elem_id="cv-btn-pedro")
                            cmaria = gr.Button("Maria", elem_id="cv-btn-maria")
                            copen = gr.Button("Open patient", elem_id="cv-btn-open-patient")
                            copen_pid = gr.Textbox(
                                value="", label="open_pid", elem_id="cv-open-pid",
                            )
                            creview = gr.Button("Review", elem_id="cv-btn-review")
                            ceid = gr.Number(
                                label="Event id", precision=0, value=None, elem_id="cv-hidden-eid",
                            )
                        cnote = gr.Markdown(
                            "Tap a patient card or alert to continue. Add patients under Settings.",
                            elem_classes=["cv-status-note"],
                        )
                        # Settings only — never on patient details / Home / Alerts.
                        with gr.Group(visible=False, elem_classes=["cv-settings-tools"]) as csettings_tools:
                            gr.HTML('<div class="cv-section-label">Add patient</div>')
                            with gr.Group(elem_classes=["cv-add-patient"]):
                                padd_name = gr.Textbox(label="Patient name", placeholder="e.g. Ana")
                                padd_room = gr.Textbox(label="Room", placeholder="e.g. Room 01")
                                padd_live = gr.Radio(
                                    label="Laptop Camera Station",
                                    choices=[
                                        ("No — board only", "no"),
                                        ("Yes — use this laptop camera", "yes"),
                                    ],
                                    value="yes",
                                    interactive=True,
                                )
                                padd_btn = gr.Button("Save patient", variant="primary")
                                gr.Markdown(
                                    "Wipe the board and load **Alex** (live) + **Casey** "
                                    "with fall alerts/session already prepared."
                                )
                                pdemo_btn = gr.Button("Reset demo (2 patients)", variant="secondary")
                        # Patient details actions (only while a case is open).
                        with gr.Group(visible=False, elem_classes=["cv-details-tools"]) as cdetails_bar:
                            gr.HTML('<div class="cv-section-label">Patient actions</div>')
                            with gr.Row(elem_classes=["cv-toolbar", "cv-exit"]):
                                cback = gr.Button("← Back to patients", scale=1)
                                crun = gr.Button("Run room check", variant="primary", scale=1)
                        # Watch recording — only after opening from an alert banner.
                        with gr.Group(visible=False, elem_classes=["cv-details-tools"]) as cplay_group:
                            gr.HTML('<div class="cv-section-label">Watch recording</div>')
                            with gr.Group(elem_classes=["cv-play-block"]):
                                csession = gr.Dropdown(
                                    label="Issue recording",
                                    choices=[],
                                    value=None,
                                    interactive=False,
                                    visible=False,
                                    elem_id="cv-session-select",
                                )
                                cplayer = gr.HTML(
                                    '<div class="pd-player-panel"><div class="pd-empty">'
                                    "No issue recording to play yet.</div></div>",
                                    elem_id="cv-session-player",
                                )
                                cplay_info = gr.HTML(
                                    '<div class="pd-player-info pd-muted">'
                                    "The recording for this alert appears here when available.</div>"
                                )
                                csession_sel = gr.State(None)
                        # Care response — only after alert banner / Alert Center.
                        with gr.Group(visible=False, elem_classes=["cv-handle-tools"]) as chandle_group:
                            from src.alert_center import (
                                build_care_response_summary_html,
                                outcome_radio_choices,
                            )
                            cresp_summary = gr.HTML(
                                build_care_response_summary_html(None),
                                elem_classes=["cv-cr-wrap"],
                            )
                            calert = gr.Dropdown(
                                label="Patient alert",
                                choices=[],
                                interactive=True,
                                elem_classes=["cv-cr-event"],
                            )
                            gr.HTML(
                                '<div class="cv-cr-step">'
                                '<div class="cv-cr-step-n">1</div>'
                                '<div><div class="cv-cr-step-h">Caregiver Response</div>'
                                '<div class="cv-cr-step-s">What did you find after checking the patient?</div>'
                                '</div></div>'
                            )
                            coutcome = gr.Radio(
                                choices=outcome_radio_choices(),
                                value="checked_ok",
                                label=None,
                                show_label=False,
                                interactive=True,
                                elem_classes=["cv-outcome-radio"],
                            )
                            gr.HTML(
                                '<div class="cv-cr-step">'
                                '<div class="cv-cr-step-n">2</div>'
                                '<div><div class="cv-cr-step-h">Care Notes (Optional)</div>'
                                '<div class="cv-cr-step-s">Add observations or actions taken during your check.</div>'
                                '</div></div>'
                            )
                            cnote_handle = gr.Textbox(
                                label=None,
                                show_label=False,
                                placeholder="Describe the patient's condition, assistance provided, or other notes...",
                                lines=3,
                                max_length=500,
                                elem_classes=["cv-cr-note"],
                            )
                            gr.HTML(
                                '<div class="cv-cr-info">'
                                "Resolve closes only the reviewed findings for this patient "
                                "(shown above). New alerts that arrive after this review stay open."
                                "</div>"
                            )
                            with gr.Row(elem_classes=["cv-cr-actions"]):
                                cack = gr.Button(
                                    "Keep Incident Active\nAcknowledge reviewed findings",
                                    scale=1,
                                    elem_classes=["cv-cr-keep"],
                                )
                                cres = gr.Button(
                                    "Resolve Incident\nClose reviewed findings",
                                    variant="primary",
                                    scale=1,
                                    elem_classes=["cv-cr-resolve"],
                                )
                            caction_note = gr.Markdown("")
                        # Alert Center filters: only when Alerts nav is active.
                        with gr.Group(visible=False, elem_classes=["cv-ac-filter-wrap"]) as calert_filters:
                            gr.HTML('<div class="cv-section-label">Alert filters</div>')
                            with gr.Row(elem_classes=["cv-toolbar", "cv-exit"]):
                                crefresh = gr.Button("Refresh alerts", scale=1)
                            with gr.Row():
                                f_priority = gr.Dropdown(
                                    label="Priority",
                                    choices=["All", "High", "Medium", "Low"],
                                    value="All",
                                    interactive=True,
                                )
                                f_patient = gr.Dropdown(
                                    label="Patient",
                                    choices=[("All patients", "All")],
                                    value="All",
                                    interactive=True,
                                )
                                f_status = gr.Dropdown(
                                    label="Status",
                                    choices=["All active", "OPEN", "ACKNOWLEDGED"],
                                    value="All active",
                                    interactive=True,
                                )
                                f_type = gr.Dropdown(
                                    label="Event type",
                                    choices=[
                                        "All types",
                                        "possible_fall",
                                        "inactive",
                                        "camera_unavailable",
                                        "out_of_view",
                                        "partial_body",
                                    ],
                                    value="All types",
                                    interactive=True,
                                )
                            with gr.Row():
                                f_from = gr.Textbox(
                                    label="From date (YYYY-MM-DD)", value="",
                                    placeholder="optional",
                                )
                                f_to = gr.Textbox(
                                    label="To date (YYYY-MM-DD)", value="",
                                    placeholder="optional",
                                )
                                f_apply = gr.Button("Apply filters", variant="secondary")
                        cuser = gr.Textbox(value="caregiver", visible=False)
                        cpass = gr.Textbox(value="care123", type="password", visible=False)
                        clogin = gr.Button("Sign in", visible=False)
                        cauth = gr.State(None)
                        cmsg = gr.Markdown(visible=False)
                        # Textbox (not State) so Timer.tick reliably reads the selected patient.
                        cpatient = gr.Textbox(
                            value="", label="patient_id", visible=False, elem_id="cv-patient-id",
                        )
                        _filter_inputs = [
                            f_priority, f_patient, f_status, f_type, f_from, f_to,
                        ]
                        # History only — keep raw event table off the patient page.
                        with gr.Group(visible=False, elem_classes=["cv-events-log"]) as cevents_log:
                            gr.HTML('<div class="cv-section-label">Events log</div>')
                            cinbox = gr.Dataframe(
                                label="Events",
                                headers=["id", "patient", "time", "type", "status", "action", "snapshot"],
                                interactive=False,
                            )
                    # Live feed below details — expands when Camera Station is LIVE.
                    with gr.Group(visible=False, elem_id="cv-live-panel") as clive_panel:
                        clive_banner = gr.Markdown(
                            "### Live Camera\nOpen Ana to view the laptop feed."
                        )
                        cframe = gr.HTML(
                            '<div class="cv-live-empty cv-live-empty-compact">'
                            "Live feed appears here when Camera Station is running.</div>",
                            elem_id="cv-case-preview",
                        )
                        cmon = gr.Markdown("Live status")
                        clive_seq = gr.State(0)
                    clive_timer = gr.Timer(0.22, active=True)
            # open→10 (with seq) + handle/play/filters/settings/events vis + playback 4
            detail_outs = [
                ccards, cnote, cframe, clive_banner, cpatient, calert, ceid,
                cdetails_bar, clive_panel, clive_seq,
                chandle_group, cplay_group, calert_filters, csettings_tools, cevents_log,
                csession, cplayer, cplay_info, csession_sel,
            ]
            if care_login is not None:
                clogin.click(
                    care_login, [cuser, cpass],
                    [cauth, cmsg, ccards, cinbox, cnav, calert],
                )
            if care_switch_panel is not None:
                cnav.change(
                    care_switch_panel,
                    [cauth, cnav, *_filter_inputs],
                    [
                        ccards, cnote, cframe, clive_banner, cpatient,
                        cdetails_bar, clive_panel, clive_seq,
                        chandle_group, cplay_group, calert_filters, csettings_tools, cevents_log,
                        csession, cplayer, cplay_info, csession_sel,
                    ],
                )
            if care_refresh is not None:
                crefresh.click(
                    care_refresh, [cauth, cnav, cpatient, *_filter_inputs],
                    [ccards, cinbox, calert],
                )
            if care_filter_alerts is not None:
                f_apply.click(
                    care_filter_alerts, [cauth, cnav, *_filter_inputs], [ccards],
                )
                for w in _filter_inputs[:4]:
                    w.change(
                        care_filter_alerts, [cauth, cnav, *_filter_inputs], [ccards],
                    )
            def _then_cr(evt):
                if care_response_preview is not None and evt is not None:
                    evt.then(
                        care_response_preview, [cauth, calert], [cresp_summary, ceid],
                    )
                return evt

            if care_open_patient is not None:
                _then_cr(copen.click(care_open_patient, [cauth, copen_pid, cnav], detail_outs))
            if care_open_ana is not None:
                _then_cr(cana.click(care_open_ana, [cauth, cnav], detail_outs))
            if care_open_pedro is not None:
                _then_cr(cpedro.click(care_open_pedro, [cauth, cnav], detail_outs))
            if care_open_maria is not None:
                _then_cr(cmaria.click(care_open_maria, [cauth, cnav], detail_outs))
            if care_add_patient is not None:
                _then_cr(padd_btn.click(
                    care_add_patient,
                    [cauth, padd_name, padd_room, padd_live, cnav],
                    [ccards, cnote, calert, f_patient, cnav],
                ))
            if care_load_demo_pack is not None:
                _then_cr(pdemo_btn.click(
                    care_load_demo_pack,
                    [cauth, cnav],
                    [ccards, cnote, calert, f_patient, cnav],
                ))
            if care_review_alert is not None:
                _then_cr(creview.click(care_review_alert, [cauth, ceid, cnav], detail_outs))
            if care_back_to_patients is not None:
                _then_cr(cback.click(
                    care_back_to_patients, [cauth, cnav],
                    [
                        ccards, cnote, cframe, clive_banner, cpatient, calert, ceid,
                        cdetails_bar, cinbox, clive_panel, clive_seq,
                        chandle_group, cplay_group, calert_filters, csettings_tools, cevents_log,
                        csession, cplayer, cplay_info, csession_sel,
                    ],
                ))
            if care_run_details is not None:
                _then_cr(crun.click(
                    care_run_details, [cauth, cpatient, cnav],
                    [
                        ccards, cnote, cframe, clive_banner, cpatient, calert, ceid,
                        cdetails_bar, cinbox, clive_panel, clive_seq,
                        chandle_group, cplay_group, calert_filters, csettings_tools, cevents_log,
                        csession, cplayer, cplay_info, csession_sel,
                    ],
                ))
            if care_response_preview is not None:
                calert.change(
                    care_response_preview, [cauth, calert],
                    [cresp_summary, ceid],
                )
            else:
                def _bundle_to_primary(token):
                    from src.alert_center import parse_alert_bundle
                    primary, _ids = parse_alert_bundle(token)
                    return primary

                calert.change(_bundle_to_primary, calert, ceid)
            if care_select_session is not None:
                csession.change(
                    care_select_session, [cauth, cpatient, csession],
                    [cplayer, cplay_info, csession_sel],
                )
            if care_live_tick is not None:
                clive_timer.tick(
                    care_live_tick, [cpatient, clive_seq],
                    [cframe, clive_banner, cmon, clive_seq],
                )
            # Interrupt poll: urgent banner + beep even while watching live.
            if care_alert_watch is not None:
                calert_timer = gr.Timer(2.0, active=True)
                calert_timer.tick(
                    care_alert_watch, [cauth, calert_seen],
                    [calert_watch, calert_seen],
                )
            if care_ack is not None:
                _then_cr(cack.click(
                    care_ack,
                    [cauth, calert, cnav, cpatient, coutcome, cnote_handle],
                    [cinbox, caction_note, ccards, calert],
                ))
            if care_resolve is not None:
                _then_cr(cres.click(
                    care_resolve,
                    [cauth, calert, cnav, cpatient, coutcome, cnote_handle],
                    [cinbox, caction_note, ccards, calert, cnav],
                ))
        demo.load(None, None, None, js=CV_BOARD_CLICK_JS)
        if care_login is not None:
            login_evt = demo.load(
                care_login, [cuser, cpass],
                [cauth, cmsg, ccards, cinbox, cnav, calert],
            )
            if care_response_preview is not None:
                login_evt.then(
                    care_response_preview, [cauth, calert], [cresp_summary, ceid],
                )
    return demo
