# The web deck (`web/`)

A 10-slide presentation that runs in any browser. Same 10-slide order the problem statement asks
for, told as a story: **the problem → why it's hard → our answer → the proof → what's next**.

## What is in the folder

| File | What it is |
|---|---|
| `web/index.html` | the 10 slides (text lives here, easy to edit) |
| `web/styles.css` | colours, fonts, layout |
| `web/deck.js` | draws the charts, handles keys and the timeline at the bottom |
| `web/data.js` | the real numbers and signals behind the charts |
| `web/assets/demo.webp` | screenshot of our Streamlit demo |

No build step, no npm. Open `web/index.html` in Chrome and it works.

## How to present

| Key | Does |
|---|---|
| → / Space | next slide |
| ← | previous slide |
| 1 … 9, 0 | jump to slide 1 … 9, 10 |
| F | full screen |
| T | switch to a dark look and back (the default is the original white deck) |
| N | show speaker notes |

Clicking a dot or a chapter name on the bottom timeline also jumps there. On a phone, swipe.

## How to put it online (Vercel)

1. Push the repo to GitHub.
2. On vercel.com: **Add New → Project**, pick the repo.
3. Set **Root Directory** to `web`. Framework preset: **Other**. No build command.
4. Deploy. The link opens on slide 1; add `#5` to the end to open slide 5.
5. The live demo is in the same deploy at `/demo/` (see `05_web_demo.md`).

(Netlify or GitHub Pages work the same way: serve the `web` folder as it is.)

## How to refresh the charts

The charts read `web/data.js`, which comes from saved outputs only (no experiment run spent):

```bash
python scripts/web_deck_data.py
```

## Handy links

- `index.html?still`: no animations (good for screenshots).
- `index.html?theme=dark`: open in the dark look.
- Chrome → Print → Save as PDF gives one slide per page.

## Exporting the web deck

The deck we submit is the LaTeX version (`deck/sentinel.tex` -> `presentation/Sentinel.pdf`). The web deck can
still be exported as pictures:

```bash
python scripts/export_web_pptx.py        # -> presentation/Sentinel_web.pptx and presentation/Sentinel_web.pdf
```

Each slide is a sharp 4K picture of the web page, so the files look exactly like the site.
The PPTX also carries the speaker notes. To change a word, edit `web/index.html` and run it again.
