// Builds DeepScan_Startup_Pitch.pptx  ->  node build_pitch_deck.js
const pptxgen = require("pptxgenjs");

const INK = "0B1E22", SURF = "13292E", CARD = "F1F6F5", LIGHT = "FAFCFC";
const TEAL = "028090", MINT = "02C39A", SEA = "00A896";
const TXT_D = "1A2E31", MUT_D = "5C7377", TXT_L = "EAF3F2", MUT_L = "9FB6B4";
const H = "Cambria", B = "Calibri";
const W = 13.33, HT = 7.5, M = 0.7;

const p = new pptxgen();
p.layout = "LAYOUT_WIDE";
p.author = "DeepScan";
p.title = "DeepScan pitch";

const shadow = () => ({ type: "outer", color: "0B1E22", blur: 10, offset: 2, angle: 90, opacity: 0.12 });

function dark(s) { s.background = { color: INK }; }
function light(s) { s.background = { color: LIGHT }; }

function title(s, text, sub, onDark) {
  s.addText(text, { x: M, y: 0.55, w: W - 2 * M, h: 0.75, fontSize: 38, bold: true, fontFace: H,
    color: onDark ? TXT_L : TXT_D, isTextBox: true, margin: 0 });
  if (sub) s.addText(sub, { x: M, y: 1.32, w: W - 2 * M, h: 0.42, fontSize: 15, fontFace: B,
    color: onDark ? MUT_L : MUT_D, isTextBox: true, margin: 0 });
}

// numbered card with a mint circle
function card(s, { x, y, w, h, n, head, body, onDark, headSize = 17, bodySize = 13 }) {
  s.addShape(p.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.12,
    fill: { color: onDark ? SURF : CARD }, line: { color: onDark ? "1E3C42" : "DCE8E6", width: 1 }, shadow: shadow() });
  let tx = x + 0.35;
  if (n) {
    s.addShape(p.ShapeType.ellipse, { x: x + 0.32, y: y + 0.3, w: 0.44, h: 0.44, fill: { color: MINT }, line: { color: MINT, width: 1 } });
    s.addText(String(n), { x: x + 0.32, y: y + 0.3, w: 0.44, h: 0.44, fontSize: 13, bold: true, fontFace: B,
      color: INK, align: "center", valign: "middle", isTextBox: true, margin: 0 });
    tx = x + 0.92;
  }
  s.addText(head, { x: tx, y: y + 0.28, w: x + w - tx - 0.3, h: 0.48, fontSize: headSize, bold: true, fontFace: H,
    color: onDark ? TXT_L : TXT_D, isTextBox: true, margin: 0, valign: "top" });
  if (body) s.addText(body, { x: x + 0.35, y: y + 0.92, w: w - 0.7, h: h - 1.18, fontSize: bodySize, fontFace: B,
    color: onDark ? MUT_L : MUT_D, isTextBox: true, margin: 0, valign: "top", lineSpacingMultiple: 1.15 });
}

function stat(s, { x, y, w, value, label, onDark, size = 46, color }) {
  s.addText(value, { x, y, w, h: 0.8, fontSize: size, bold: true, fontFace: H, color: color || MINT,
    isTextBox: true, margin: 0, align: "left" });
  s.addText(label, { x, y: y + 0.82, w, h: 0.75, fontSize: 12.5, fontFace: B, color: onDark ? MUT_L : MUT_D,
    isTextBox: true, margin: 0, lineSpacingMultiple: 1.1 });
}

function foot(s, text, onDark) {
  s.addText(text, { x: M, y: HT - 0.62, w: W - 2 * M, h: 0.3, fontSize: 10, fontFace: B,
    color: onDark ? "6D8A8C" : "8CA3A6", isTextBox: true, margin: 0 });
}

/* 1 — title */
let s = p.addSlide(); dark(s);
s.addShape(p.ShapeType.ellipse, { x: 9.6, y: -1.5, w: 6.2, h: 6.2, fill: { color: TEAL, transparency: 82 }, line: { color: TEAL, width: 0 } });
s.addShape(p.ShapeType.ellipse, { x: 11.0, y: 3.2, w: 3.4, h: 3.4, fill: { color: MINT, transparency: 88 }, line: { color: MINT, width: 0 } });
s.addText("DeepScan", { x: M, y: 2.15, w: 8.6, h: 1.15, fontSize: 60, bold: true, fontFace: H, color: TXT_L, isTextBox: true, margin: 0, charSpacing: -1 });
s.addText("Proof, not a guess, for every image people act on.", { x: M, y: 3.35, w: 8.8, h: 0.6, fontSize: 21, fontFace: B, color: MINT, isTextBox: true, margin: 0 });
s.addText("A multi-signal forensic engine that decides whether a photo is real, AI-generated or a deepfake — and shows the evidence behind the verdict.",
  { x: M, y: 4.05, w: 8.4, h: 1.0, fontSize: 14.5, fontFace: B, color: MUT_L, isTextBox: true, margin: 0, lineSpacingMultiple: 1.2 });
s.addText("Synthetic Data-Augmented Deepfake Detection  ·  Team DeepScan", { x: M, y: 6.25, w: 9, h: 0.35, fontSize: 12, fontFace: B, color: "6D8A8C", isTextBox: true, margin: 0 });
s.addNotes("One line: DeepScan tells you whether an image can be trusted, and shows why. We are a forensic evidence engine, not a single black-box model.");

/* 2 — problem */
s = p.addSlide(); light(s);
title(s, "Seeing is no longer believing", "Anyone can now produce a convincing fake face in minutes, and the people who must judge those images have no tools.", false);
const probs = [
  ["Banks & fintech", "Onboarding selfies and video KYC can be passed with a swapped face."],
  ["Insurance & claims", "Damage photos and documents can be generated to order."],
  ["Courts, HR & press", "Evidence and sources arrive as screenshots no one can verify."],
];
probs.forEach((c, i) => card(s, { x: M + i * 4.13, y: 2.15, w: 3.83, h: 1.95, head: c[0], body: c[1] }));
s.addShape(p.ShapeType.roundRect, { x: M, y: 4.5, w: W - 2 * M, h: 1.75, rectRadius: 0.12, fill: { color: INK }, line: { color: INK, width: 1 } });
stat(s, { x: M + 0.45, y: 4.78, w: 3.3, value: "$40B", label: "projected US generative-AI fraud losses by 2027 (Deloitte Center for Financial Services, 2024)", onDark: true, size: 40 });
stat(s, { x: M + 4.3, y: 4.78, w: 3.3, value: "10x", label: "growth in detected deepfake fraud attempts year over year (Sumsub Identity Fraud Report, 2023)", onDark: true, size: 40 });
stat(s, { x: M + 8.1, y: 4.78, w: 3.4, value: "0", label: "tools most of these teams have today: they forward the image to a colleague and guess", onDark: true, size: 40, color: SEA });
foot(s, "Market figures are third-party estimates, cited above; DeepScan's own measurements appear on the results slide.", false);
s.addNotes("The pain is not curiosity about AI images. It is money and liability: a fake face opens an account, a generated photo settles a claim.");

/* 3 — why existing tools fail */
s = p.addSlide(); light(s);
title(s, "Why today's detectors disappoint", "We tested the public state of the art on our own data before building anything.", false);
const fails = [
  ["One model, one blind spot", "A single detector trained on yesterday's generators scores today's images near zero. In our tests one face-swap image was rated 'real' by four separate detectors at once."],
  ["No explanation", "A bare 'FAKE, 87%' cannot be shown to a customer, a regulator or a judge. Nobody can check the reasoning."],
  ["Overconfident on the edge", "Binary tools must answer even when the evidence is thin, so they fail silently and confidently."],
  ["Stale within months", "Every new image model breaks a fixed detector; retraining the whole model each time is not viable."],
];
fails.forEach((c, i) => card(s, { x: M + (i % 2) * 6.13, y: 2.15 + Math.floor(i / 2) * 2.25, w: 5.83, h: 2.05, n: i + 1, head: c[0], body: c[1] }));
s.addNotes("This slide is our credibility: we measured these failures, we did not read them in a blog post.");

/* 4 — solution / pipeline */
s = p.addSlide(); dark(s);
title(s, "DeepScan: a jury, not a single judge", "Independent detectors and measured image forensics are weighed together into one calibrated decision.", true);
const steps = [
  ["Image or video", "Face found?\nFull-frame and face paths"],
  ["5 ML detectors", "Face-swap, AI-image and\nmodern-generator models"],
  ["41 forensic signals", "Eyes, mouth, skin, blending,\nlighting, noise, frequency"],
  ["Weighted fusion", "Weights learned from\nlabelled images"],
  ["Verdict + evidence", "REAL · AI-GENERATED\nDEEPFAKE · UNCERTAIN"],
];
const bw = 2.22, gap = 0.29;
steps.forEach((st, i) => {
  const x = M + i * (bw + gap);
  s.addShape(p.ShapeType.roundRect, { x, y: 2.35, w: bw, h: 2.25, rectRadius: 0.12,
    fill: { color: i === 4 ? TEAL : SURF }, line: { color: i === 4 ? MINT : "1E3C42", width: 1 }, shadow: shadow() });
  s.addText(st[0], { x: x + 0.18, y: 2.6, w: bw - 0.36, h: 0.75, fontSize: 15, bold: true, fontFace: H, color: TXT_L,
    isTextBox: true, margin: 0, align: "center", valign: "middle" });
  s.addText(st[1], { x: x + 0.18, y: 3.35, w: bw - 0.36, h: 1.05, fontSize: 11.5, fontFace: B,
    color: i === 4 ? "D8F0EC" : MUT_L, isTextBox: true, margin: 0, align: "center", lineSpacingMultiple: 1.15 });
  if (i < 4) s.addText("›", { x: x + bw, y: 3.2, w: gap, h: 0.5, fontSize: 22, bold: true, fontFace: B, color: MINT,
    isTextBox: true, margin: 0, align: "center" });
});
s.addText("Every signal is measured from the actual pixels. A signal that cannot be measured is reported as unavailable and the remaining evidence is re-weighted — the engine never invents a number.",
  { x: M, y: 5.1, w: W - 2 * M, h: 0.8, fontSize: 14, fontFace: B, color: MUT_L, isTextBox: true, margin: 0, lineSpacingMultiple: 1.2 });
s.addNotes("The jury metaphor: five models plus measurable physical evidence, weighted by how reliable each one proved to be on labelled data.");

/* 5 — product */
s = p.addSlide(); light(s);
title(s, "What the user actually gets", "An answer they can defend, in about three seconds per image.", false);
card(s, { x: M, y: 2.15, w: 5.9, h: 3.3, head: "Verdict with a confidence you can trust", body:
  "REAL · AI-GENERATED · DEEPFAKE · UNCERTAIN\n\nUNCERTAIN is a feature, not a cop-out: it fires only when the evidence genuinely conflicts, so a confident answer means something.\n\nEvery verdict carries a REAL score and a FAKE score on the same 0-1 scale." });
card(s, { x: M + 6.13, y: 2.15, w: 5.9, h: 3.3, head: "Evidence breakdown per image", body:
  "Nine signal groups scored separately: ML detectors, face geometry and blending, eyes, nose and mouth, skin texture, background, lighting, frequency and compression, metadata.\n\nPlus a plain-language reason, the raw detector scores, and any camera or C2PA provenance found." });
s.addShape(p.ShapeType.roundRect, { x: M, y: 5.65, w: W - 2 * M, h: 0.95, rectRadius: 0.1, fill: { color: "E4F2EE" }, line: { color: "C6E3DB", width: 1 } });
s.addText("Deployed as a web app today; the same engine is exposed as a REST API and runs entirely on the customer's own machine — no image ever leaves their network.",
  { x: M + 0.35, y: 5.82, w: W - 2 * M - 0.7, h: 0.62, fontSize: 13.5, fontFace: B, color: TXT_D, isTextBox: true, margin: 0, valign: "middle" });
s.addNotes("Demo beat: upload an image, show the verdict, then scroll to the evidence panel and the raw detector trace.");

/* 6 — moat */
s = p.addSlide(); light(s);
title(s, "What competitors cannot copy quickly", "The fusion layer is the product; detectors are interchangeable parts.", false);
const moats = [
  ["Model-agnostic fusion", "A better detector is added as one more input and gets its weight from data, without rebuilding anything."],
  ["Calibrated on labelled data", "Weights and decision boundaries are fitted on 1,135 labelled images and checked on held-out images, not hand-tuned."],
  ["Explainability by construction", "The evidence panel is a by-product of how the decision is made, not a story written afterwards."],
  ["Recalibration flywheel", "New generator appears → add images → refit in minutes → measured before and after. No full retraining."],
];
moats.forEach((c, i) => card(s, { x: M + (i % 2) * 6.13, y: 2.15 + Math.floor(i / 2) * 2.25, w: 5.83, h: 2.05, n: i + 1, head: c[0], body: c[1] }));
s.addNotes("Anyone can download a detector. The defensible part is the calibrated fusion and the pipeline that keeps it current.");

/* 7 — measured results */
s = p.addSlide(); light(s);
title(s, "Measured, not claimed", "Results on held-out images the system never saw during training or calibration.", false);
s.addChart(p.ChartType.bar, [{
  name: "Held-out validation",
  labels: ["Real photos\ncorrect", "AI images\ncaught", "Deepfakes\ncaught", "Real wrongly\ncalled fake"],
  values: [78, 76, 50, 4],
}], {
  x: M, y: 2.1, w: 7.2, h: 4.0, barDir: "col", chartColors: [TEAL, SEA, MINT, "E0A458"],
  varyColors: true, showTitle: true, title: "Verdict accuracy on 379 held-out images (%)", titleFontSize: 13,
  titleColor: TXT_D, titleFontFace: B, showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 12,
  dataLabelColor: TXT_D, dataLabelFontFace: B, showLegend: false, valAxisMaxVal: 100, valAxisMajorUnit: 25,
  catAxisLabelColor: MUT_D, valAxisLabelColor: MUT_D, catAxisLabelFontSize: 11, valAxisLabelFontSize: 11,
  catAxisLabelFontFace: B, valAxisLabelFontFace: B, valGridLine: { color: "E2EBE9", size: 1 }, catGridLine: { style: "none" },
});
stat(s, { x: 8.3, y: 2.35, w: 4.3, value: "0.93", label: "ranking quality (ROC-AUC) on the 72-image acceptance set: fakes score above reals 93% of the time" });
stat(s, { x: 8.3, y: 3.95, w: 4.3, value: "97%", label: "precision on that set: when DeepScan says fake, it is almost always right" });
stat(s, { x: 8.3, y: 5.5, w: 4.3, value: "2.0%", label: "false alarms on 523 held-out real photos for the modern-generator detector", color: TEAL });
foot(s, "Sources: FaceForensics++, FaceShifter, OpenFake (70+ generators), DeepFakeFace, DF40/Celeb-DF, bitmind face-swap. Numbers from our own test runs, September 2026.", false);
s.addNotes("Be honest in the room: deepfake recall at 50% is the weak spot and the reason for the current data expansion. Precision is the number that matters for trust.");

/* 8 — market */
s = p.addSlide(); light(s);
title(s, "Who pays, and why now", "Verification is becoming a compliance requirement, not a nice-to-have.", false);
const segs = [
  ["Identity & KYC providers", "Per-check verification inside onboarding flows."],
  ["Insurance & claims teams", "Photo evidence screening before payout."],
  ["Newsrooms & fact-checkers", "Source images verified before publication."],
  ["Legal, HR & universities", "Disputed images and submitted work."],
];
segs.forEach((c, i) => card(s, { x: M + i * 3.1, y: 2.15, w: 2.85, h: 2.0, head: c[0], body: c[1], headSize: 15, bodySize: 12 }));
s.addShape(p.ShapeType.roundRect, { x: M, y: 4.5, w: W - 2 * M, h: 1.75, rectRadius: 0.12, fill: { color: INK }, line: { color: INK, width: 1 } });
s.addText("Regulation is arriving on a timetable", { x: M + 0.45, y: 4.75, w: 11, h: 0.4, fontSize: 17, bold: true, fontFace: H, color: TXT_L, isTextBox: true, margin: 0 });
s.addText("The EU AI Act requires providers and deployers to label AI-generated and manipulated content; India's IT ministry has moved to mandate labelling of synthetic media; US financial regulators (FinCEN, 2024) have alerted institutions to deepfake-enabled fraud. Labelling duties create demand for independent verification.",
  { x: M + 0.45, y: 5.2, w: 11.6, h: 0.9, fontSize: 13, fontFace: B, color: MUT_L, isTextBox: true, margin: 0, lineSpacingMultiple: 1.15 });
foot(s, "Regulatory references are public policy documents; confirm current status before quoting dates in a live pitch.", false);
s.addNotes("Frame timing around obligation, not novelty: once labelling is required, someone has to check the labels.");

/* 9 — business model */
s = p.addSlide(); light(s);
title(s, "Business model", "Land with a self-serve API, expand into on-premise licences where privacy and volume matter.", false);
const tiers = [
  ["Verify API", "Pay per image or video checked, volume tiers. Fits an existing onboarding or claims flow in a day."],
  ["On-premise licence", "Annual licence per deployment. The engine runs inside the customer's network; no image leaves it."],
  ["Evidence reports", "Per-case PDF evidence pack with the full signal breakdown, for disputes, audits and court use."],
];
tiers.forEach((t, i) => {
  const x = M + i * 4.13;
  card(s, { x, y: 2.15, w: 3.83, h: 2.6, head: t[0], body: t[1] });
});
s.addShape(p.ShapeType.roundRect, { x: M, y: 5.05, w: W - 2 * M, h: 1.35, rectRadius: 0.12, fill: { color: "E4F2EE" }, line: { color: "C6E3DB", width: 1 } });
s.addText("Costs stay low because detection runs on ordinary hardware: the whole engine was built and is served from a single 8 GB laptop, with no per-image cloud inference bill.",
  { x: M + 0.4, y: 5.3, w: W - 2 * M - 0.8, h: 0.85, fontSize: 14, fontFace: B, color: TXT_D, isTextBox: true, margin: 0, lineSpacingMultiple: 1.15 });
s.addNotes("Pricing levels are deliberately not on the slide; quote them only if asked, and say they are indicative.");

/* 10 — go to market */
s = p.addSlide(); light(s);
title(s, "How we reach the first customers", "Narrow, evidence-led, and cheap to run.", false);
const gtm = [
  ["Pilot with one workflow", "Free evaluation on a customer's own images; we report measured accuracy on their data, not ours."],
  ["Publish the method", "Open benchmark results and the evidence format build trust faster than marketing claims."],
  ["Academic and campus pilots", "Universities face AI-submitted work and have image-verification needs today."],
  ["Partner, don't replace", "Integrate as a verification step inside existing KYC and claims platforms."],
];
gtm.forEach((c, i) => card(s, { x: M + (i % 2) * 6.13, y: 2.15 + Math.floor(i / 2) * 2.25, w: 5.83, h: 2.05, n: i + 1, head: c[0], body: c[1] }));
s.addNotes("The pilot offer is the whole sales motion: we measure on their data and show the evidence panel.");

/* 11 — roadmap and limits */
s = p.addSlide(); light(s);
title(s, "What works today, and what comes next", "We are explicit about the limits — that is why customers can trust the numbers we do publish.", false);
card(s, { x: M, y: 2.15, w: 5.9, h: 3.55, head: "Working today", body:
  "Images: real, AI-generated, deepfake or uncertain, with full evidence\nVideo: frames sampled and scored the same way\nRuns fully offline on a laptop\nWeb app plus REST API\nPrecision 97% on held-out images" });
card(s, { x: M + 6.13, y: 2.15, w: 5.9, h: 3.55, head: "Next, in order", body:
  "1. Deepfake recall (50% today): training on newly added public face-swap datasets is in progress\n2. Recent generators: continuous ingestion of new AI models\n3. Audio and full video pipelines\n4. Evidence report export and audit log\n5. Independent third-party benchmark" });
foot(s, "Honesty policy: no result in this deck comes from an image the system was trained or calibrated on.", false);
s.addNotes("Saying the weak number out loud before they find it is what makes the strong numbers believable.");

/* 12 — closing */
s = p.addSlide(); dark(s);
s.addShape(p.ShapeType.ellipse, { x: -1.6, y: 3.6, w: 5.6, h: 5.6, fill: { color: TEAL, transparency: 85 }, line: { color: TEAL, width: 0 } });
title(s, "The web is losing its default trust.", "DeepScan gives that trust back one image at a time — with the evidence attached.", true);
const asks = [
  ["Pilot partners", "One KYC, insurance or newsroom team willing to test on their own images."],
  ["Compute and data", "Support for continuous recalibration against new generators."],
  ["Advisors", "Forensics, fraud and compliance expertise for product direction."],
];
asks.forEach((c, i) => card(s, { x: M + i * 4.13, y: 2.6, w: 3.83, h: 2.1, n: i + 1, head: c[0], body: c[1], onDark: true, headSize: 16 }));
s.addText("Try it on any image you bring to this room.", { x: M, y: 5.3, w: 9, h: 0.5, fontSize: 20, bold: true, fontFace: H, color: MINT, isTextBox: true, margin: 0 });
s.addText("DeepScan — Synthetic Data-Augmented Deepfake Detection", { x: M, y: 5.9, w: 9, h: 0.4, fontSize: 13, fontFace: B, color: MUT_L, isTextBox: true, margin: 0 });
s.addNotes("Close by inviting a live test with an image the audience chooses. The evidence panel is the memorable part.");

p.writeFile({ fileName: "/Users/akshitraj/Downloads/Ai_deepfake-main/presentation/DeepScan_Startup_Pitch.pptx" })
  .then(f => console.log("wrote", f));
