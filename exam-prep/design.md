# Design - NLP Final Mock Room (ห้องสอบจำลอง)

Locked design system v2. Every screen in `mock_exam.html` defers to this file. Amend it on purpose; do not override locally. The page is generated from `tools/mock_template.html`, so edit the template and run `python tools/build_html.py`.

## Direction change (read first)

v1 was an austere exam-paper look (white, blue-black, red pen). The user asked for something modern, colourful and dimensional, with real components. v2 keeps the usability rules from v1 but replaces the look: a study cockpit with soft depth, one colour per exam week, and a tutor panel that reacts to the answer. Do not drift back to monochrome.

## Brief

- Audience: Thai CS undergraduate, 4 days before the final, alone, often at night, laptop first and phone second.
- Job: sit a realistic 80-question mock under time, then learn from the mistakes.
- Tone: confident, energetic, calm under pressure. Colour carries meaning (which week, right or wrong), never decoration. No confetti, streaks, or points.
- References (Pinterest, mood only, nothing copied): real computer-based-test screens (timer on top, answer-status palette, fixed action bar) for the usability skeleton; a quiz page with a thin progress bar and a status list; pastel SaaS exam portals were the look to avoid (illustrated students, donut dashboards). Dribbble was blocked in the tool.

## System

- Genre: modern-minimal with colour (product UI, not a landing page).
- Macrostructure: Cockpit. Three columns on desktop: question map (left), question card (centre), tutor panel (right).
- Theme: custom (vibe: "night-study, electric blue, week-coded hues"). Light and dark both first class; dark is deep navy, never black.
- Framework decision: no CSS framework and no component CDN. Components are plain JS render functions plus the CSS below, so the page works offline except for fonts and KaTeX (both degrade gracefully).

## Tokens

```css
:root{
  /* surfaces (tinted, never pure white or black) */
  --bg:oklch(0.965 0.012 250); --surface:oklch(0.995 0.004 250); --surface-2:oklch(0.955 0.012 250);
  --line:oklch(0.9 0.015 255); --ink:oklch(0.24 0.04 265); --ink-2:oklch(0.48 0.03 265);
  /* the one brand accent: electric blue */
  --brand:oklch(0.56 0.21 259); --brand-2:oklch(0.48 0.21 263); --brand-ink:oklch(0.99 0.005 259); --brand-soft:oklch(0.94 0.04 259);
  /* semantic */
  --ok:oklch(0.62 0.17 152); --ok-soft:oklch(0.95 0.05 152);
  --bad:oklch(0.6 0.21 22);  --bad-soft:oklch(0.95 0.04 22);
  --warn:oklch(0.78 0.15 80); --warn-soft:oklch(0.96 0.06 85);
  /* week hue wheel; each week sets --h, components derive their tints from it */
  /* W8 25 coral, W9 75 amber, W10 148 green, W11 185 teal, W12 225 cyan-blue, W13 265 blue, W14 345 pink (spaced at least 40 degrees apart) */
  /* --wk, --wk-soft, --wk-ink are derived from --h ON the element that sets --h, via the selector [style*="--h"] (a :root definition would freeze them at the fallback hue) */
  --font:"IBM Plex Sans Thai","Noto Sans Thai","Leelawadee UI",Tahoma,system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;
  --r-card:16px; --r-ctl:12px; --r-pill:999px;
  --sh-1:0 1px 2px oklch(0.3 0.05 260 / .08),0 8px 24px -10px oklch(0.4 0.1 260 / .25);
  --sh-2:0 2px 4px oklch(0.3 0.05 260 / .1),0 18px 40px -14px oklch(0.4 0.14 260 / .35);
  --ease:cubic-bezier(0.16,1,0.3,1);
}
```

Dark theme keeps the same hues with lower lightness and higher chroma on accents: bg `oklch(0.2 0.035 265)`, surface `oklch(0.25 0.04 265)`, ink `oklch(0.95 0.01 255)`, brand `oklch(0.72 0.16 255)`. No raw colours outside the token blocks.

## Depth rules (the "dimension" the user asked for)

- Three elevation levels only: flat (hairline), raised (`--sh-1`, cards), floating (`--sh-2`, hovered or active cards, sticky tutor panel). Shadows are tinted blue, never grey-black.
- Cards have a 1px translucent top highlight (`inset 0 1px 0 oklch(1 0 0 / .6)` in light, `/ .06` in dark).
- Primary button: vertical gradient `--brand` to `--brand-2`, inner top highlight, coloured shadow; on press it drops 1px and the shadow shrinks.
- Hover on interactive cards lifts 2px. Active presses down 1px. Transform and shadow only.
- Page background is a quiet two-bloom gradient in the corners, tinted from the brand hue and the teal week hue. Nothing animated.

## Shape lock

Cards 16px. Controls (buttons, option rows, inputs) 12px. Chips, badges, map pills, bubbles: pill or 10px squares for map cells. One scale, used everywhere.

## Typography

IBM Plex Sans Thai for everything, IBM Plex Mono for numbers (timer, question numbers, scores, map cells). Two families total. Headings 700, tight tracking, always roman (no italic). Question stem 18px, line-height 1.75, max 68ch. No em-dashes in visible strings.

## Layout

Desktop >= 1180px: `grid-template-columns: 248px minmax(0,1fr) 336px`.

- Left, question map (sticky, scrolls inside): grouped by week; each week group has a coloured header chip and 10 to 12 square cells (36px). Cell states: empty (outline in week colour), answered (filled week colour), current (2px ink ring plus lift), flagged (small amber corner), after check or after submit correct (green) and wrong (rose).
- Centre, question card: week chip (week colour), topic, difficulty badge, "ข้อ n จาก N" in mono; the stem; four option cards with a letter badge; a fixed action row at the bottom of the card (previous, flag, hint, check, next).
- Right, tutor panel (sticky): a card with a coloured header. Exam mode: shows remaining time, answered count, shortcut keys, a calm "ไม่มีเฉลยจนกว่าจะส่ง". Tutor mode: idle tips; hint; after check a verdict card (green on correct, rose on wrong) with the explanation and a source chip; fill-in answers get the answer key plus three self-grade buttons.
- 760px to 1179px: map collapses to a horizontal scroll strip under the header; tutor panel drops below the question card.
- < 760px: single column, action row fixed to the bottom in two rows (secondary actions above, primary "ข้อถัดไป" full width), map becomes a collapsible drawer. Minimum target 44px.

Start screen: hero text, three stat chips (days to the real exam, 80 questions, 180 minutes), then a week picker made of 7 coloured cards plus an "ทั้งหมด" card (this is the scope selector), beside a setup card (question set as a 2x2 choice grid, mode as a two-way segmented control, time toggle, gradient start button, resume button, history list below).

Submit: review gate card (unanswered and flagged numbers as clickable pills) before the final send. Time-out auto-submits and says so.

Result: score gauge (conic ring in brand colour), three stat chips, week bars in week colours, tutor summary card, then the filterable review list with a coloured left edge per week and green or rose verdict chips.

## Components and states

Every interactive component has all eight states: default, hover, focus-visible, active, disabled, loading, error, success. Focus ring is 3px `--brand` at 45% alpha plus 2px solid, visible instantly.

- Option card: letter badge (mono) plus text. Selected: brand-soft fill, brand border, brand badge. After check: correct is green with a check word "ถูก", chosen-wrong is rose with "ที่คุณเลือก". Never colour alone: the chip text also changes.
- Chip: pill, week-soft background, week-ink text, 1px week border.
- Segmented control, choice card, week card: radio inputs under the hood, whole card clickable.
- Timer pill: mono; under 10 minutes it turns warn with underline text-decoration.
- Toast-free: no popups. Status lives in the panel.

## Usability rules (hard, carried from v1)

1. Question number, week, answered count, time left always visible without scrolling at 1280x720.
2. Keyboard: A to D or 1 to 4 select, arrows move, F flags, Enter checks (tutor) then goes next.
3. Selecting an answer never moves the layout. Never auto-advance.
4. Exam mode is silent: no right or wrong colours until submit.
5. Autosave and resume restore the exact question, answers, flags and remaining time.
6. Contrast at least 4.5:1 for text in both themes.
7. Math via KaTeX; if it fails to load the raw TeX stays readable.
8. Do not invent numbers in the UI; everything comes from the data or the student's attempt.

## Motion

Short and functional: card lift 180ms, option select 120ms, tutor verdict slides in 220ms (opacity and translateY 6px). Reduced-motion: all collapse to opacity-only or instant. No page transitions, no stagger, no loops.

## Voice

Plain Thai, short, concrete, no exclamation marks, no emoji. Tutor example: "ยังไม่ใช่ ที่ถูกคือ B" then the reason then the source path. Result example: "สัปดาห์ 12 ได้ 5 จาก 12 ข้อ กลับไปอ่านก่อน".

## Anti-patterns

Purple gradients, glassmorphism on everything, neon glow, three identical feature cards, decorative icons, hand-drawn SVG icons, stock illustrations of students, gradient text, donut-chart dashboards with fake numbers, a floating mascot, emoji, confetti, a modal for submit, colour as the only signal.

## Audit rules added after the 2026-11-05 UX audit

- Solid fills that carry text use dedicated tokens: `--ok-solid`, `--bad-solid` (with `--on-solid`) and `--on-wk` for week-colour fills. The light tints (`--ok`, `--bad`, `--wk`) are for borders and backgrounds behind dark text only. Contrast target: 4.5:1, checked with the script in the audit notes.
- Smallest text is 12.8px (0.8rem). Primary reading text is 16px or more.
- Targets: 44px minimum on touch (`pointer:coarse`), 40px minimum with a mouse. The question map cells are 40px on desktop and 44px on touch.
- Header must fit at 320px: below 480px the logo text hides, the timer and two buttons stay visible.
- The progress segment row must not inherit the header's `min-height`.

## v3: difficulty levels and subjects (2026-11-05)

- Four levels, names and rubric in `subjects/<id>/LEVELS.md` and `subject.json`: 1 ง่าย, 2 ปานกลาง, 3 ยาก, 4 ยากมาก. Level is shown as a chip with a four-step meter (neutral ink, never a traffic-light colour, because colour is reserved for week identity and right or wrong).
- Start screen has a level picker above the set picker: "ผสมตามสัดส่วนข้อสอบจริง" (default, quota per 10 MCQ from `mixLevels`, short levels borrow from neighbours) plus one card per level showing how many questions exist. A live line under the picker says how many questions and minutes the current choice produces, or warns when none match.
- Subjects: everything course-specific lives in `subjects/<id>/` (unit names and hues, MCQ and fill counts per unit, sets, levels, exam date, notes path). The header shows a subject drop-down only when more than one subject exists. The UI word for a unit comes from `units.label` (สัปดาห์, บท, ...). Storage keys are namespaced per subject.
- Local testing: the page is 570 KB, too large for the in-app file preview, so serve it (`.claude/launch.json` has `mock-exam` on port 8791).

## v4 additions (2026-11-05)

- Options are shuffled per run (stored in the saved run, so resume keeps the order). Items whose stem, options or explanation refer to option letters (about 77 of the bank) are marked `ns` by `tools/build_html.py` and keep their original order. Letters shown to the student are display positions; scoring uses the original letter.
- Fill-in answers get a suggested grade ("ระบบเสนอ: ถูก / ถูกบางส่วน / ผิด" with how many blanks matched, numbers compared with 0.5% tolerance, terms compared after normalising case, spaces and commas). The student always confirms; nothing is graded silently.
- Stats screen (`#stats`): attempts, accuracy, accuracy by level and by unit, weakest topics, review queue (items whose last result was wrong, up to 40 per run, tutor mode). Stored per subject in `localStorage` key `mock-<id>-qstat`; clearing needs two clicks.
- Offline: `python tools/vendor_fetch.py` downloads KaTeX 0.16.9 and the IBM Plex subsets once into `tools/vendor/`; `tools/build_html.py` then embeds them (page about 1.5 MB, no network needed). `--online` builds the small version that loads them from CDNs.
- Element ids must stay unique across all sections (a duplicate `stats` id once broke the review queue).
