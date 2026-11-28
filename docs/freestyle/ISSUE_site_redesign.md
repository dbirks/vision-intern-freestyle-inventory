TITLE: Redesign the operator site around one live photo: Now / Retraining / Trends (+ Gelato design system)

## Summary

The site should answer one question first: **what does the cabinet look like right now?** It should lead with the best current photo of the shelves, with the machine's per-cartridge counts and a **"fix / correct me"** action beside it. Everything else — dashboards, exports, admin faff — earns its place only if it serves *photo → counts → corrections → reorder*.

We just ran a clean-room vision project on the cabinet photos (separate repo, branch `dbirks/vision-intern-freestyle-inventory@freestyle-cartridges`, docs under `docs/freestyle/`). It produced a working detector + VLM counter, a phone quiz that produced human ground truth, and one very clear lesson: **photo quality is the ceiling on everything else**, and the human-in-the-loop correction step is where the actual value is. This issue asks for a redesign in the style of the "Gelato" study-guide design system, a three-page information architecture (Now / Retraining / Trends), and a ruthless simplification pass.

> **Assumption, please correct me:** the implementing agent has the current site in front of us; we do not (clean-room rule). Anything below about "current pages/panels/widgets" is a question, not a statement. Where a decision depends on what already exists, ask before deleting.

---

## What we learned from the clean-room vision project

Read these as **directional, not settled** — the numbers are honest but the sample is tiny.

1. **The photo dominates everything.** Of 1,383 frames the deployment declared, **54 survived retention** (~96% loss); after dropping non-cabinet/occluded frames, **40 usable cabinet frames** remain. Of those 40, only **21 reach our "sharp" bar** (Laplacian variance ≥ 20) and the single sharpest frame in the whole corpus scores **26.1** — no frame is genuinely crisp. Motion blur was severe in 15, **3 frames were upside-down and 1 rotated 90°**, and **people appear in 38 of 69 frames**. When a human tried to answer "how many cartridges are in this photo?" for the 12 held-out test photos, **10 of them were uncountable** (door swing, blur, angle) — 8 explicitly marked *"not a usable cabinet view"*, 2 never answered.
   - **Implication for this site:** the biggest quality win available is not a better model. It is (a) a **fixed, well-positioned camera** instead of a door-mounted one, and (b) the software **picking the sharpest usable frame per door opening** and showing the operator *why* it was picked. Frame selection and camera mounting are product decisions, not ML decisions.
2. **Detector + reader architecture works, provisionally.** Best combination: an **RF-DETR Small detector trained on cartridge front faces** (trained on 9 auto-labelled frames / 597 faces) + **Qwen 3.8 Flash Next** reading each crop (two-stage: brand family → zoomed numeral, with a shelf hint). On the **2** test photos a human could count (22 and 28 cartridges) it was off by **0** and **1** cartridges — mean **0.5/photo** — and named the right product on **21 of the 28** crops the human also named (**0.750**; the alternative reader got 18/28).
   - **This is n = 2 photographs of one cabinet with one stocking.** The bootstrap intervals printed beside those numbers are arithmetic over two observations, not uncertainty. Call it *provisional* in any UI copy; do not market it as accuracy.
3. **Naming is the weak link, not counting.** Brand/family is roughly **~70% right** (family-correct 0.69 on 39 human-labelled crops). The printed **Part 1 / Part 2 numeral is effectively unreadable at this camera's resolution and blur**: the best reader got the digit right on **1 of 11** crops where a human knew the answer; across whole label passes a digit came back on **3 of 47** and **2 of 123** two-part-brand crops. Consequence: **a Part-1/Part-2 inventory split is not available from this camera today.** Two-part brands ship as **glued double cartridges**; public placement maps always put `CART 1` left of / above `CART 2`, so a "pair rule" is our workaround — currently a **hypothesis**, reported never applied.
4. **`unknown` is a real output and must look like one.** Even in the best run, **7–10 of ~25 cartridges per photo** come back `unknown`, and the system is deliberately *not* allowed to guess them into a class. The UI must show "counted, not named" as a first-class state (with a jump to Retraining), never silently as zero or as a guess.
5. **Ground-truth semantics matter more than accuracy.** The single most consequential bug in the whole project was scoring an uncountable photo as "zero cartridges". Fixing that one rule flipped the ranking of 13 model variants and retracted most of the earlier headline numbers. **Rule for this site: "not usable" excludes a record; it is never a zero.** A zero is a claim about the cabinet; "not usable" is a claim about the photo.
6. **The correction loop is cheap and it works.** A phone-first quiz (one zoomed crop + the model's guess + big tap targets) got **128 human answers** in ~20 minutes of tapping, and those answers are the only trustworthy ground truth we have. That interaction is the model for the **Retraining** page below.
7. **Latency is a design constraint.** The reader is one API call per cartridge: measured **3–9 s per cartridge**, so a stocked shelf (~25 cartridges) takes **~75–225 s** end to end. The site must show counts **as they arrive** (or show the previous complete set until a new one lands) — never a spinner that makes the operator wait two minutes for a page.
8. **Cost of the missing metadata:** the raw package has **no capture timestamps, no firmware attribution, no door-angle offsets** (the data owners' own `GAPS.md`), so openings can't be ordered in time and blur can't be tied to door angle. Whatever this site already records, **per-opening timestamps and per-SKU counts are the inputs that make the Trends page possible later** — see that section.

Docs if you want the detail: `docs/freestyle/REPORT.md` (exec summary + limitations §11), `RESULTS_GT.md` (human-scored tables), `READER_COMPARE.md` (reader bake-off), `QUIZ_APP.md` (the phone quiz), `HOW_TO_RUN.md` (what the pipeline actually outputs today).

---

## Proposed information architecture

Three pages. **Now** is the product; the other two support it. Suggested routes: `/` (Now), `/retraining`, `/trends`.

**Q for the implementer:** what are today's routes, and which are dead weight? Is there already an operator login (the ordering person) vs admin, and does the current app have a per-opening concept to hang "the current photo" on?

### 1. Now — what the cabinet looks like right now

The current best photo, large. Machine counts to the side. One obvious **Fix / correct me** button.

**Desktop (≥1024 px)**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ ● Freestyle Cabinet                                   ☼/☾   [Search]  Now ▾  │  floating pill
├───────────┬──────────────────────────────────────────────┬───────────────────┤
│ Now       │  ◉ Best frame from opening #412 · 08:12       │  DETECTED         │
│ Retraining│  ┌──────────────────────────────────────────┐ │  24 cartridges    │
│ Trends    │  │                                          │ │  17 named · 7 ?   │
│ ───────── │  │        PHOTO (largest element on the     │ │  ──────────────   │
│ sharpness │  │        page; tap/click to zoom)          │ │  NNS Sweetener  6 │
│ ▓▓▓░ soft │  │        (boxes overlaid, toggleable)      │ │  Fanta          2 │
│ 21/40     │  │                                          │ │  Sprite         1 │
│ frame was │  │                                          │ │  Hi-C           3 │
│ picked    │  └──────────────────────────────────────────┘ │  …                │
│ because…  │  [ ◐ Show detections ]  [ ⤢ Full size ]       │  [ Fix / correct  │
│           │  ⚠ 7 cartridges counted but not named →      │    me ]  (primary)│
│           │    3 older frames from this opening · camera  │                   │
└───────────┴──────────────────────────────────────────────┴───────────────────┘
```

**Mobile (<640 px)** — single column, photo first, counts directly under it, fix action pinned.

```
┌────────────────────────────┐
│ ☰  Now        ⚠ soft  ⬤   │
├────────────────────────────┤
│ ⬤ Opening #412 · 08:12     │
│ ┌────────────────────────┐ │
│ │                        │ │
│ │     PHOTO (full width, │ │
│ │     ~55–65vh, zoomable)│ │
│ │                        │ │
│ └────────────────────────┘ │
│ 24 cartridges · 17 named   │
│ ──────────────────────────  │
│ NNS Sweetener      6        │
│ Fanta              2        │
│ … (top 5, then "all 18")   │
│ ⚠ 7 counted, not named →   │
│ ┌────────────────────────┐ │
│ │    Fix / correct me    │ │  pinned, 44px+
│ └────────────────────────┘ │
└────────────────────────────┘
```

**Rules for Now**
- The photo is the hero. The AI result is a **sidebar/annotation of the photo**, not a separate dashboard competing with it.
- "Best frame" must say which frame it is and **why** (sharpness score, frame index in the opening), and let you flip to sibling frames from the same opening.
- Counts state their provenance: `machine count` vs `human-corrected` vs `stale`. If the newest photo is 6 hours old, say so.
- "Fix / correct me" goes to the correction flow (see Retraining §interaction) but **seeded with this opening's photo and counts**, pre-filled, not a blank form.

### 2. Retraining — the model's shaky items, one tap at a time

A quiz fed by the model's own uncertainty: highest-impact questions first, one zoomed crop per screen.

**Mobile (primary device — this is a standing-at-the-cabinet task)**

```
┌────────────────────────────┐
│ Retraining      12/47   ⬤  │   progress: answered / queued
├────────────────────────────┤
│ What is this cartridge?    │   (the model's guess below, in serif)
│ ┌────────────────────────┐ │
│ │  ZOOMED CROP           │ │   crop from ORIGINAL pixels, padded, upscaled
│ │  (tap to zoom further) │ │
│ └────────────────────────┘ │
│  ▸ thumbnail of the frame   │   with this box highlighted
│  Model: Fanta (62%)        │
│ ──────────────────────────  │
│ ┌────────────────────────┐ │
│ │  ✓ That's right        │ │   big target, 46px
│ └────────────────────────┘ │
│ [NNS][AA][CC][Fanta][Hi-C] │   18 catalog chips, colour swatch, 2 rows
│ [Sprite][Cola P1][Cola P2] │
│ ── or ──                   │
│ [Can't tell][Not usable]   │
│                            │
│        (tap picks →        │
│        [ Next ▸ ] appears) │
└────────────────────────────┘
```

**Desktop** — same content, one card centred at the reading measure (~720 px); queue position + "why this question" in the left sidebar; the last 10 answers as a trailing strip. No tables, no multi-select grid on this page — a grid is what made our desktop review app something nobody used twice.

### 3. Trends — trajectory and reorder rates (not built yet)

**Mobile**

```
┌────────────────────────────┐
│ Trends                     │
├────────────────────────────┤
│ ⚠ Still gathering data to  │   butter callout, always visible
│   make recommendations —   │   until the gate in §Trends is met
│   here's what current      │
│   reorder rates look like. │
│   31 openings · 18 days    │
│ ──────────────────────────  │
│ Total cartridges over time │
│ 28┤      ╭─╮   ╭╮          │
│ 20┤ ╭────╯ ╰───╯ ╰╭─╮      │   past runs = solid; current run
│ 12┤─╯             ╰ ╰─ ─   │   = solid to "now", dashed after
│    └──────────────────────  │
│    Mar 3        Mar 21 ⬤    │
│ ──────────────────────────  │
│ Reorder rate per SKU       │
│ NNS   ▓▓▓▓▓▓▓▓ 1.8/wk       │
│ Fanta ▓▓▓░░░░░ 0.4/wk  ?    │   ? = too few samples to say
│ ──────────────────────────  │
│ Add a photo of each opening │   if manual capture is still needed
└────────────────────────────┘
```

**Desktop** — the graph fills the reading column; a compact SKU rate table beside it; no more than one chart above the fold.

---

## Design language (the "Gelato" system, from the study guide)

Reference implementation: `claude-certified-developer-professional-study-guide` — `src/styles/tokens.css`, `base.css`, `layout.css`, `home.css`, `mdx.css`, `src/layouts/BaseLayout.astro`. Our devlog (`vision-intern/devlog/build.py`) is a working copy of the same system with a Coke-red accent; crib from whichever is easier to read.

**Type**
- `--font-serif: 'Young Serif'` — display only (page titles, the model's answer, card titles). **Young Serif ships weight 400 only; never set `font-weight > 400` on serif text.**
- `--font-sans: 'Manrope Variable'` — everything else. `--font-mono: 'DM Mono'` — counts, IDs, timestamps, table numerals (`font-variant-numeric: tabular-nums`).
- Scale actually used: body 16 px (17 px ≥1024), line-height 1.65; page hero 34→42 px; `h2` 18 px/700 sans; `h3` 16 px/700; eyebrows 12 px/700 uppercase `letter-spacing .04em`; secondary text 13–14 px; rail 13.5 px. Reading measure `72ch`, clamped to a **720 px column** on desktop.

**Colour (light)** — `--bg #fff8f1`, `--surface #fff`, `--bg-subtle #fbefe4`, `--ink #2b2330`, `--ink-muted #6e6470`, `--border #efe3d8`. Four pastel *fills* used as semantic cards: `--sky #dcebfa` (hero/info), `--pistachio #ddefd8` (ok / correct / next), `--strawberry #fbdde3` (mistake / low stock / alert), `--butter #fff0c2` (warning / "still gathering data"). Each fill carries its own `*-ink`/`*-title` tokens — never black-on-pastel. Accent for links/actions: `--sky-ink #2a4a6e`, or **Coke red `--cola #c8102e`** (our devlog's focus colour) if you want the brand to own the primary action.

**Colour (dark)** — warm plum-gray, **never black**: `--bg #34313a`, `--surface #423e48`, `--ink #f3eef0`, and the pastels *darken to muted fills with pastel text* (`--sky #3c4a5a` / `--sky-ink #cfe2f7`, `--strawberry #5a3f47` / `#f8cbd5`, …). The dark theme must be **designed in the same pass**, not inverted: photos on a pure-black page look wrong and the alert colours go muddy.

**Shape, space, depth** — radius scale 8/16/20/22/26/28 px + `--r-pill 999px`; spacing 4/8/12/16/24/32; `--shadow: 0 8px 28px rgba(80,50,40,.14)`, `--shadow-pill: 0 2px 10px rgba(80,50,40,.06)`. Cards are **filled, borderless, generously rounded** (`--r-lg`/`--r-2xl`, 16–22 px padding). Focus ring `2px solid var(--focus)` + 2 px offset; pills keep their round focus shape.

**Layout** — mobile-first, three breakpoints (640 / 1024 / 1280):
- `<1024`: single column; the nav is a **left drawer** (`transform: translateX(-100%)` → `none`), with a backdrop, a header pill, and a theme switch inside it.
- `≥1024`: `grid-template-columns: 300px minmax(0,1fr)` (study guide) — for this site the left rail is *now* the detected-counts panel, so consider `1fr 320px` instead; either is fine, one rail only.
- `≥1280`: adds a 220 px right "on this page" rail (skip it if there is nothing to put there).
- Sticky **floating pill topbar**, 52 px mobile / 60 px desktop, no border, translucent `backdrop-filter: blur(14px) saturate(1.2)` band fading under it. `env(safe-area-inset-*)` padding everywhere.

**Dark-mode switching** — `<html data-theme="light">` plus an **inline pre-paint script** reading `localStorage` → falling back to `prefers-color-scheme`, so there is no flash; `color-scheme: light dark` in CSS and `<meta name="color-scheme">`; theme toggle in the topbar and in the drawer.

**Components worth copying verbatim in spirit** — stat pills (fixed 32 px height, `flex: 0 0 auto`, tabular numerals), callout cards (filled pastel, `--r-lg`, uppercase kind label + serif title), the "signal → verdict" striped table (rows butt together, odd rows `--surface-alt`, verdict as a pill — a good fit for *cartridge → count*), the two-card pager, and hover effects gated behind `@media (hover: hover)` with `prefers-reduced-motion` respected. Tap targets ≥44 px (our quiz uses 44/46).

---

## Simplification checklist

**Principles**
1. Every page must justify itself against: *does it help with the current photo, the counts, a correction, or the order?* If not, it goes.
2. One primary action per screen. On Now that's **Fix / correct me**; on Retraining it's **✓ / pick a class**.
3. Show fewer numbers, labelled better. A count with no provenance ("machine" vs "human-corrected", "as of when") is worse than no count.
4. No metric on any page without a sentence a non-technical operator could read aloud.
5. Prefer a photo + a caption to a chart + a legend.
6. Screens a person uses weekly are not worth building; screens used daily get the biggest type.

**Audit questions for whoever implements this (we have not seen the current site)**
- Which pages/routes exist, and which had any traffic in the last 90 days? List them with a keep / merge / delete recommendation.
- Are there duplicated representations of the same fact (a photo, a table, a chart and a status card all saying "24 cartridges")? Keep **one** (the photo + count) and link out.
- Is there an existing "history", "reports", "settings", "admin", "devices" or "export/CSV" area? For each: is the ordering person the user? If not, move it under a secondary route or remove it — **don't** leave it on the main navigation.
- Does the current app already have any correction/annotation UI we should reuse instead of the new one? (Better to upgrade a live path than to add a fourth place to fix things.)
- Does the current site already show the photo, and if so, is it the deployment's *selected* keyframe? Who selects it today — the edge device or the server? (This determines whether "pick the sharpest frame" is our work or theirs.)
- Is there an existing login/role model, and does the ordering person have credentials today?
- What is already timestamped and stored per opening (counts, boxes, corrections)? Anything we'd have to start capturing to make Trends real.
- Anything here is a candidate for deletion, but **nothing gets deleted in the same PR that adds the new pages** — hide first, delete after a release of traffic data.

---

## Retraining page spec

Model it on the phone quiz that produced all our ground truth (works today, ~20 min of tapping, 128 answers).

**Interaction**
- **One question per screen**, highest-impact first. No grids, no tables to scan.
- Two question types: **count** ("model counts 14 top / 10 bottom = 24. Right?" → ✓ / fix with two steppers / *not a usable photo*) and **identity** (one crop).
- **Identity screen**: a crop rendered from **original pixels** with ~15 % padding, upscaled to fill the phone width (cap ~6× so tiny boxes don't become soup), plus a small frame thumbnail with that box highlighted, plus the model's guess and its confidence in big serif type.
- **Answer affordances**: a big `✓ That's right`, then every active catalog SKU as a chip with its colour swatch; `Can't tell`; `Not a cartridge`; `This box covers 2+ cartridges`; `This box cuts a cartridge`. For a two-part brand family (Coca-Cola / Diet Coke / Dr Pepper / Coke Zero), put `Part 1` / `Part 2` / `Can't tell` **first and largest** — that's the question the model can't answer, so don't bury it under 18 other chips.
- **Select-then-Next**: a tap *picks*, `[ Next ▸ ]` appears and submits. Never fire on the first tap — a mis-tap on an 18-chip grid is otherwise unrecoverable.
- **Brief ✓ confirmation**: hold a big ✓ overlay ~500 ms before submitting. It's the whole reason the quiz felt good on a phone with cold hands.
- `Skip` always available; `undo` re-opens the last question; **resume** returns to the first unanswered question across sessions/devices; prefetch the next 2–3 crops so paging is instant.
- Answer log is **append-only, atomic** (one JSON line per answer, `fsync`; last answer wins; undo rewrites atomically). Every answer carries an ISO timestamp, the model's guess and its confidence at the time.
- Keyboard on desktop (`y`, `1`–`9`, `x`, `s`, `u`) — cheap, and it makes the page usable on a laptop.

**Queue order (this is the product)**
- **Highest expected information gain first**: crops where the reader is least confident, SKUs with the fewest confirmed examples, and the un-readable Part-1/Part-2 numerals. Our active-learning pass did exactly this and it is why 128 answers were enough to score anything at all.
- **A question the human has already answered is never dropped** from the queue when it is rebuilt (dropping it throws the answer away); only *unanswered* questions may be re-prioritised.
- **Skipping/not-usable on a photo skips that photo's crops.** In the quiz, marking the count question *"not a usable cabinet view"* auto-answered all of that image's crop questions as implied-not-usable. Do not make someone tap through 25 crops of a blurred frame.
- Show `n answered / total queued` and, if the queue is empty, say *"nothing uncertain right now — the model is confident, or there are new photos to process first."*

**Ground-truth semantics (copy these exactly; they cost us a retraction)**
- `Not usable` / `not a cabinet view` = **exclude the record**. It is *never* a zero, and never a "0 cartridges" inventory row. A zero is a claim about stock; unusable is a claim about a photo.
- `Can't tell` is a legitimate answer, not a failure: it keeps the item as `unknown` and does not enter an accuracy denominator. A reader should be *rewarded* for abstaining on these; our best reader abstained on 4–5 of the 5 crops a human also couldn't call.
- `This box covers 2+ cartridges` / `cuts a cartridge` are **localisation** corrections, not naming answers — they should route to a box-fix path (or at least be recorded as such), because they mean the count is wrong, not the label.
- **Model labels are never ground truth.** A corrected record is human-verified; a pre-filled suggestion is not, and the UI should never render the two with the same weight.
- **Test vs train separation must be explicit in the schema.** Each photo/opening belongs to a split: photos held out for **evaluation** ("measure the system") must have their corrections kept **out of the training/identification gallery**. Our quiz tags every question `purpose: eval | tune | train` and refuses to put an eval/val image in front of the gallery builder. If the site's "training data" is a single undifferentiated pile of corrections, then the accuracy the site quotes is fiction — the model will have studied the exam. Ship a UI-visible marker for it (a small "held for evaluation" pill on those questions) so it stays honest as people join.
- Corrections from the Now page's **Fix / correct me** are the same answer log and the same semantics, just seeded at that opening — one pipeline, two entry points.

---

## Trends / estimation spec (not built yet — the honesty rules first)

**Non-negotiable copy rule:** until the data gate below is met, the page leads with a `--butter` callout in the *voice of the product*:

> **Still gathering data to make recommendations.** Here's what current reorder rates look like — and what we're not ready to predict yet.

and states its own basis in the same breath: *N openings · M days · K human-corrected counts*.

**Data gate (propose, then tune)**
- No forecast line, no "order by" date, no reorder recommendation until **≥ 60 days and ≥ 40 openings** of continuous, timestamped per-SKU counts, and **no per-SKU rate** until that SKU has appeared in ≥ 8 openings with confirmed counts.
- Below the gate: show *observations only* (counts over time, observed drop per week) with the sample size on the chart itself. Above the gate: still label the projection as an estimate with a range, and show when it last agreed with reality.
- While the numeral is unreadable, **two-part SKUs share a combined series** (Part 1+Part 2 as one "Coca-Cola" line) and the chart *says so*. Plotting a Part-1 line that is really a coin flip is the fastest way to lose the operator's trust.

**What to draw once we can**
1. **Cartridges-over-time trajectory**: past openings as a solid line, the current opening as the live point, and the projection dashed **only** after the gate. "Past runs vs current run" on one axis is the graph the developer asked for.
2. **Reorder rate per SKU**: observed consumption per week (bars, mono numerals), sorted by rate, with `?` on anything below the per-SKU sample gate.
3. A `low / watch` pill using `--strawberry` when a SKU's observed rate implies empty before the next planned visit — the one place the site may make a forward claim, phrased as "at the current rate", never as a promise.

**What must be captured now so this page is possible later** (ask which of these already exist):
- **One record per door opening** with a stable ID: ISO timestamp (the raw corpus has none — this is the top gap), camera/device ID, the chosen frame + its sharpness score and frame index, and *how* it was chosen.
- Per-SKU counts **and** the `unknown` count, per shelf, with the model + bundle version and thresholds that produced them (a count without a model version can't be re-scored when the model changes).
- Every human correction with timestamp and user, appended, never overwriting the machine count.
- Per-frame boxes in **original pixel coordinates** — never in the coordinates of a resized copy — and **keep the chosen photo forever**. Losing 96% of frames to retention is the reason the vision project scored on two photos; storage of one sharp frame per opening is cheap compared to that.
- Refill events (what came out of the box / what went on the shelf), if any operator-capturable form exists; without them, "reorder rate" stays an inference from counts alone.

---

## Acceptance criteria

- [ ] Now page loads with **the current best photo occupying the largest visual element** on both a 390 px-wide phone and a 1440 px desktop; a 3-way screenshot check is in the PR.
- [ ] Now shows the per-SKU counts, the `unknown`/not-named count, the photo's age and which frame it is — each with a visible provenance label (`machine` / `human-corrected`, `as of <time>`).
- [ ] "Fix / correct me" is reachable in one tap from the photo and opens a correction flow pre-filled with that opening's counts and boxes.
- [ ] Counts arrive progressively (photo first, then counts as the reader finishes) with no blocking spinner longer than ~5 s; the previous complete set stays visible meanwhile.
- [ ] A photo marked **not usable** never produces a zero-cartridge inventory record anywhere in the app; a test asserts the two states are distinct end to end.
- [ ] Retraining presents one crop at a time with the crop rendered from original pixels, the model's guess + confidence, ≥44 px targets, select-then-Next, ~500 ms ✓ confirmation, `Can't tell`, `Not usable`, undo, and resume from the first unanswered question.
- [ ] Marking a photo not usable skips that photo's remaining crops.
- [ ] Queue order is by model uncertainty (and shows a one-line "why am I being asked this"), and previously answered questions survive a queue rebuild.
- [ ] Corrections on held-out/evaluation photos are recorded with a split/purpose tag and demonstrably excluded from whatever feeds identification (a test that a held-out answer cannot reach the training/gallery path).
- [ ] Trends leads with the "still gathering data" state, quotes its own basis (openings/days/corrected counts), and cannot display a projection or a reorder recommendation before the data gate; part-numbered SKUs are plotted combined while the numeral is unreadable.
- [ ] All three pages render correctly in light **and** dark, chosen before first paint with no flash, and respect `prefers-reduced-motion`; hover effects only under `@media (hover: hover)`.
- [ ] Type/tokens come from a shared token file (Gelato: Young Serif 400-only display, Manrope body, DM Mono numerals, pastel filled borderless cards, 640/1024/1280 breakpoints); no per-page colour literals.
- [ ] Keyboard shortcuts and focus rings work on the correction flow; the site is usable one-handed on a phone at arm's length.
- [ ] A short `DESIGN.md`/PR note lists what was **removed or hidden** and the traffic evidence for it; nothing is deleted in the same change that adds the new pages.
- [ ] No cabinet photos, no PII, no credentials, no internal hostnames in the repo, in fixtures, or in screenshots — use synthetic/placeholder imagery for tests and CI.

## Out of scope / open questions

- **Out of scope:** retraining or re-serving the model itself (this site's job is to *collect good corrections*; the training loop lives in the vision project), on-device camera control, hardware procurement, any accuracy claim shown to operators, and the ordering/purchasing integration.
- Should Trends be a third top-level page or a section at the bottom of Now? (Recommend separate: it is read weekly, Now is read daily.)
- What exactly is the ordering person's cadence — a morning check, or on demand when a drink fails? It decides whether "live" means "photo from the last opening" or "photo from the last 10 minutes", and whether a notification exists at all.
- Camera: can we mount a fixed one, and does the edge device expose frame selection to us? **This is the highest-leverage change the project can make and it is not a web change.**
- Who is allowed to correct? One trusted role (recommended: the ordering person *is* the ground truth), or many? Corrections need an author to be usable as GT.
- Does the current site already keep chosen frames? For how long? Retention policy decides whether Trends starts at zero or at a month of history.
- Do we show operators *any* accuracy/confidence numbers, or keep those internal? Recommendation: keep them internal — at n=2 they are not quotable, and a wrong number on an operator screen outlives the correction.
- Does the "held for evaluation" split get decided by the vision project (a frozen split file pushed to the site) or by the site? It must not be decided by whoever is tapping at the time.
