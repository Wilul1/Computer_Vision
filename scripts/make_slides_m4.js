/**
 * Milestone 4 slide draft (8+ slides).
 *   node scripts/make_slides_m4.js
 */
const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");

const ROOT = path.resolve(__dirname, "..");
const OUT = path.join(ROOT, "docs", "slides_m4.pptx");
const FIG = path.join(ROOT, "docs", "figures");
const FAIL = path.join(ROOT, "docs", "failures_m4");
const CASES = path.join(ROOT, "outputs", "test_cases");

const C = {
  bg: "0F1C1A",
  panel: "17302C",
  card: "1F3D37",
  ink: "F4F7F5",
  muted: "A8BDB6",
  accent: "3DDC97",
  warn: "F0A202",
  bad: "E4572E",
  white: "FFFFFF",
};

async function main() {
  const pres = new pptxgen();
  const S = pres.shapes;
  const card = (slide, x, y, w, h) => {
    slide.addShape(S.ROUNDED_RECTANGLE, {
      x, y, w, h,
      fill: { color: C.card },
      rectRadius: 0.08,
      objectName: "card",
    });
  };
  pres.layout = "LAYOUT_16x9";
  pres.author = "CSST 106 — Alabat · Porcopio · Veridiano";
  pres.title = "Milestone 4 — Activity Monitor (80% check)";
  pres.subject = "Motion Detection and Activity Monitoring";

  // 1. Title / problem
  {
    const s = pres.addSlide();
    s.addShape(S.RECTANGLE, { x: 0, y: 0, w: 10, h: 5.625, fill: { color: C.bg } });
    s.addText("MILESTONE 4 · 80% CHECK", {
      x: 0.55, y: 1.1, w: 9, h: 0.35,
      fontFace: "Arial", fontSize: 14, color: C.accent, bold: true,
      margin: 0, isTextBox: true,
    });
    s.addText("Motion Detection &\nActivity Monitoring", {
      x: 0.55, y: 1.55, w: 8.5, h: 1.5,
      fontFace: "Arial", fontSize: 36, color: C.ink, bold: true,
      margin: 0, isTextBox: true,
    });
    s.addText("Laptop webcam / video → person → standing, sitting, walking, running, or lying.\nNo custom training — YOLO11n + MediaPipe Pose + rules tuned on val.", {
      x: 0.55, y: 3.3, w: 8.5, h: 0.7,
      fontFace: "Arial", fontSize: 15, color: C.muted,
      margin: 0, isTextBox: true,
    });
    s.addText("Alabat Cindy I  ·  Porcopio Zheniel Ivan  ·  Veridiano Wilmark\nCSST 106 · tag cp4-80", {
      x: 0.55, y: 4.5, w: 8.5, h: 0.6,
      fontFace: "Arial", fontSize: 13, color: C.muted,
      margin: 0, isTextBox: true,
    });
  }

  // 2. Data
  {
    const s = pres.addSlide();
    s.addShape(S.RECTANGLE, { x: 0, y: 0, w: 10, h: 5.625, fill: { color: C.bg } });
    s.addText("Dataset", {
      x: 0.5, y: 0.35, w: 9, h: 0.5,
      fontFace: "Arial", fontSize: 32, color: C.ink, bold: true, margin: 0, isTextBox: true,
    });
    s.addText("Split by source (subject / movie) so no person or film leaks across train / val / test.", {
      x: 0.5, y: 0.9, w: 9, h: 0.35,
      fontFace: "Arial", fontSize: 14, color: C.muted, margin: 0, isTextBox: true,
    });

    const boxes = [
      { t: "GMDCSA24", d: "Fixed room camera\nDay + night\nSitting / standing / walking / lying", x: 0.5 },
      { t: "HMDB51", d: "Movie clips\nOften moving camera\nWalk + run", x: 3.5 },
      { t: "Splits", d: "Train 280\nVal 108\nTest 90 segments", x: 6.5 },
    ];
    for (const b of boxes) {
      card(s, b.x, 1.5, 2.8, 2.4);
      s.addText(b.t, {
        x: b.x + 0.2, y: 1.7, w: 2.4, h: 0.4,
        fontFace: "Arial", fontSize: 18, color: C.accent, bold: true, margin: 0, isTextBox: true,
      });
      s.addText(b.d, {
        x: b.x + 0.2, y: 2.25, w: 2.4, h: 1.4,
        fontFace: "Arial", fontSize: 14, color: C.ink, margin: 0, isTextBox: true,
      });
    }
    s.addText("Tuning only on val. Test held out. Own webcam clips still planned (see remaining.md).", {
      x: 0.5, y: 4.3, w: 9, h: 0.5,
      fontFace: "Arial", fontSize: 13, color: C.muted, margin: 0, isTextBox: true,
    });
  }

  // 3. Approach
  {
    const s = pres.addSlide();
    s.addShape(S.RECTANGLE, { x: 0, y: 0, w: 10, h: 5.625, fill: { color: C.bg } });
    s.addText("Approach", {
      x: 0.5, y: 0.35, w: 9, h: 0.5,
      fontFace: "Arial", fontSize: 32, color: C.ink, bold: true, margin: 0, isTextBox: true,
    });
    const steps = [
      ["1  Preprocess", "Resize ≤640 · CLAHE · gray blur"],
      ["2  Infer", "YOLO person · MediaPipe pose · LK camera motion · MOG2"],
      ["3  Postprocess", "Angles, thigh ratio, hip speed − camera, stride → rules"],
      ["4  Render", "Boxes, skeleton, label, confidence, FPS"],
    ];
    steps.forEach((row, i) => {
      const y = 1.1 + i * 0.95;
      card(s, 0.5, y, 9, 0.85);
      s.addText(row[0], {
        x: 0.7, y: y + 0.15, w: 3.2, h: 0.5,
        fontFace: "Arial", fontSize: 18, color: C.accent, bold: true, margin: 0, isTextBox: true,
      });
      s.addText(row[1], {
        x: 4.0, y: y + 0.2, w: 5.2, h: 0.5,
        fontFace: "Arial", fontSize: 15, color: C.ink, margin: 0, isTextBox: true,
      });
    });
  }

  // 4. Pipeline diagram
  {
    const s = pres.addSlide();
    s.addShape(S.RECTANGLE, { x: 0, y: 0, w: 10, h: 5.625, fill: { color: C.bg } });
    s.addText("Pipeline", {
      x: 0.5, y: 0.3, w: 9, h: 0.45,
      fontFace: "Arial", fontSize: 32, color: C.ink, bold: true, margin: 0, isTextBox: true,
    });
    const pipe = path.join(FIG, "pipeline.png");
    if (fs.existsSync(pipe)) {
      s.addImage({ path: pipe, x: 0.4, y: 0.9, w: 9.2, h: 4.3, objectName: "pipeline" });
    } else {
      s.addText("(pipeline.png missing — run scripts/draw_pipeline.py)", {
        x: 0.5, y: 2.5, w: 9, h: 0.4, fontFace: "Arial", fontSize: 16, color: C.warn, isTextBox: true,
      });
    }
  }

  // 5. Results table
  {
    const s = pres.addSlide();
    s.addShape(S.RECTANGLE, { x: 0, y: 0, w: 10, h: 5.625, fill: { color: C.bg } });
    s.addText("Results (held-out test)", {
      x: 0.5, y: 0.3, w: 9, h: 0.45,
      fontFace: "Arial", fontSize: 32, color: C.ink, bold: true, margin: 0, isTextBox: true,
    });
    s.addText("python src/evaluate.py --split test   ·   tag cp4-80   ·   90 segments", {
      x: 0.5, y: 0.8, w: 9, h: 0.3,
      fontFace: "Arial", fontSize: 13, color: C.muted, margin: 0, isTextBox: true,
    });
    s.addTable(
      [
        [
          { text: "Method", options: { bold: true, color: C.ink } },
          { text: "Accuracy", options: { bold: true, color: C.ink } },
          { text: "Macro-F1", options: { bold: true, color: C.ink } },
        ],
        ["MOG2 baseline", "0.178", "0.081"],
        ["Pose + rules (M2)", "0.622", "0.651"],
        ["Pose + rules (M3 params)", "0.633", "0.657"],
        [
          { text: "Pose + rules (M4 main)", options: { bold: true, color: C.accent } },
          { text: "0.756", options: { bold: true, color: C.accent } },
          { text: "0.772", options: { bold: true, color: C.accent } },
        ],
      ],
      {
        x: 0.5, y: 1.25, w: 5.6, h: 2.6,
        colW: [3.2, 1.2, 1.2],
        border: [{ pt: 0 }, { pt: 0 }, { pt: 0 }, { pt: 0 }],
        fontFace: "Arial", fontSize: 13, color: C.ink,
        fill: { color: C.card },
        align: "left", valign: "middle",
      }
    );
    card(s, 6.4, 1.25, 3.1, 2.6);
    s.addText("15.1 FPS", {
      x: 6.55, y: 1.55, w: 2.8, h: 0.7,
      fontFace: "Arial", fontSize: 32, color: C.accent, bold: true, margin: 0, isTextBox: true, align: "center",
    });
    s.addText("66 ms/frame after warm-up\ni7-6700HQ · CPU only\nYOLO 5.6 MB · Pose 5.8 MB", {
      x: 6.55, y: 2.4, w: 2.8, h: 1.2,
      fontFace: "Arial", fontSize: 12, color: C.muted, margin: 0, isTextBox: true, align: "center",
    });
    s.addText("Fixed room camera: 93% (42/45). Movie clips: 58%. Sitting recall 0.39 → 1.00 since M3.", {
      x: 0.5, y: 4.3, w: 9, h: 0.6,
      fontFace: "Arial", fontSize: 14, color: C.ink, margin: 0, isTextBox: true,
    });
  }

  // 6. Sample outputs + confusion
  {
    const s = pres.addSlide();
    s.addShape(S.RECTANGLE, { x: 0, y: 0, w: 10, h: 5.625, fill: { color: C.bg } });
    s.addText("Sample outputs", {
      x: 0.5, y: 0.25, w: 9, h: 0.4,
      fontFace: "Arial", fontSize: 32, color: C.ink, bold: true, margin: 0, isTextBox: true,
    });
    const cm = path.join(FIG, "confusion_test_main.png");
    if (fs.existsSync(cm)) {
      s.addImage({ path: cm, x: 0.35, y: 0.8, w: 4.6, h: 4.4, objectName: "confusion" });
    }
    const samples = [
      [path.join(CASES, "TC12_tc12_sit_photo.jpg"), "TC12 sit photo ✔"],
      [path.join(CASES, "TC13_tc13_empty_room.jpg"), "TC13 empty room ✔"],
      [path.join(FAIL, "T009.jpg"), "Fail: arm exercise → walk"],
    ];
    samples.forEach((item, i) => {
      const y = 0.85 + i * 1.45;
      card(s, 5.2, y, 4.4, 1.3);
      if (fs.existsSync(item[0])) {
        s.addImage({ path: item[0], x: 5.35, y: y + 0.12, w: 1.7, h: 1.05, objectName: `sample${i}` });
      }
      s.addText(item[1], {
        x: 7.2, y: y + 0.45, w: 2.2, h: 0.5,
        fontFace: "Arial", fontSize: 13, color: C.ink, margin: 0, isTextBox: true,
      });
    });
  }

  // 7. Limitations
  {
    const s = pres.addSlide();
    s.addShape(S.RECTANGLE, { x: 0, y: 0, w: 10, h: 5.625, fill: { color: C.bg } });
    s.addText("Limitations", {
      x: 0.5, y: 0.3, w: 5, h: 0.45,
      fontFace: "Arial", fontSize: 32, color: C.ink, bold: true, margin: 0, isTextBox: true,
    });
    const lims = [
      ["Moving / hand-held camera", "8 of 22 test fails — camera motion estimate breaks on pans"],
      ["Tiny / dark / lying missed", "YOLO gate hides false skeletons but also true poses (TC04, TC09)"],
      ["Slow walk → standing", "Speed under tuned threshold on movie clips"],
      ["Arm exercise while standing", "Limb motion looks like a step (TC08)"],
    ];
    lims.forEach((row, i) => {
      const y = 0.95 + i * 0.85;
      card(s, 0.5, y, 5.5, 0.75);
      s.addText(row[0], {
        x: 0.7, y: y + 0.1, w: 5.1, h: 0.28,
        fontFace: "Arial", fontSize: 14, color: C.warn, bold: true, margin: 0, isTextBox: true,
      });
      s.addText(row[1], {
        x: 0.7, y: y + 0.38, w: 5.1, h: 0.28,
        fontFace: "Arial", fontSize: 12, color: C.muted, margin: 0, isTextBox: true,
      });
    });
    const ex = path.join(FAIL, "T048.jpg");
    if (fs.existsSync(ex)) {
      s.addImage({ path: ex, x: 6.3, y: 1.1, w: 3.2, h: 2.8, objectName: "fail-ex" });
      s.addText("Example: moving-camera fail (T048)", {
        x: 6.3, y: 4.05, w: 3.2, h: 0.4,
        fontFace: "Arial", fontSize: 12, color: C.muted, margin: 0, isTextBox: true,
      });
    }
    s.addText("Designed for: fixed camera · full body ~2–5 m · room light · ≥1.5 s activities", {
      x: 0.5, y: 4.7, w: 9, h: 0.4,
      fontFace: "Arial", fontSize: 13, color: C.accent, margin: 0, isTextBox: true,
    });
  }

  // 8. Test cases + demo plan
  {
    const s = pres.addSlide();
    s.addShape(S.RECTANGLE, { x: 0, y: 0, w: 10, h: 5.625, fill: { color: C.bg } });
    s.addText("Test cases & demo plan", {
      x: 0.5, y: 0.3, w: 9, h: 0.45,
      fontFace: "Arial", fontSize: 32, color: C.ink, bold: true, margin: 0, isTextBox: true,
    });
    card(s, 0.5, 1.0, 4.5, 3.6);
    s.addText("13 file cases · 7 hard", {
      x: 0.7, y: 1.2, w: 4.1, h: 0.4,
      fontFace: "Arial", fontSize: 18, color: C.accent, bold: true, margin: 0, isTextBox: true,
    });
    s.addText("9 / 13 pass\n\nHard from M3 errors:\nTC05–07 sitting ✔\nTC08 arm exercise ✘\nTC09 very dark ✘\nTC10 half-res ✘\n\n+ TC14/15 live webcam in class", {
      x: 0.7, y: 1.7, w: 4.1, h: 2.6,
      fontFace: "Arial", fontSize: 14, color: C.ink, margin: 0, isTextBox: true,
    });
    card(s, 5.2, 1.0, 4.3, 3.6);
    s.addText("Live demo order", {
      x: 5.4, y: 1.2, w: 3.9, h: 0.4,
      fontFace: "Arial", fontSize: 18, color: C.accent, bold: true, margin: 0, isTextBox: true,
    });
    s.addText("1. python app.py\n2. TC01 sit · TC03 walk · TC05 night sit\n3. Show a hard fail (TC08 or TC09)\n4. Webcam: full-body walk\n5. Teacher inputs\n6. evaluate.py numbers vs report", {
      x: 5.4, y: 1.75, w: 3.9, h: 2.5,
      fontFace: "Arial", fontSize: 14, color: C.ink, margin: 0, isTextBox: true,
    });
  }

  // 9. Remaining to 100%
  {
    const s = pres.addSlide();
    s.addShape(S.RECTANGLE, { x: 0, y: 0, w: 10, h: 5.625, fill: { color: C.bg } });
    s.addText("To 100%", {
      x: 0.5, y: 0.35, w: 9, h: 0.5,
      fontFace: "Arial", fontSize: 32, color: C.ink, bold: true, margin: 0, isTextBox: true,
    });
    s.addText("From docs/remaining.md — owners/dates to confirm with the teacher", {
      x: 0.5, y: 0.9, w: 9, h: 0.3,
      fontFace: "Arial", fontSize: 13, color: C.muted, margin: 0, isTextBox: true,
    });
    const todos = [
      "Own webcam/phone clips + separate webcam eval",
      "More lying labels from GMDCSA24 fall endings",
      "Fall / sudden abnormal movement objective",
      "Final report, slides, rehearsed demo + backup video",
      "Freeze eval once → tag v1.0 / cp5-final",
    ];
    todos.forEach((t, i) => {
      const y = 1.4 + i * 0.7;
      card(s, 0.5, y, 9, 0.6);
      s.addText(`${i + 1}`, {
        x: 0.7, y: y + 0.12, w: 0.5, h: 0.35,
        fontFace: "Arial", fontSize: 16, color: C.accent, bold: true, margin: 0, isTextBox: true,
      });
      s.addText(t, {
        x: 1.3, y: y + 0.12, w: 7.9, h: 0.35,
        fontFace: "Arial", fontSize: 15, color: C.ink, margin: 0, isTextBox: true,
      });
    });
  }

  await pres.writeFile({ fileName: OUT });
  console.log("Wrote", OUT);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
