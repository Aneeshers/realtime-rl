# realtime-rl

Static project page for **Learning Planning Budgets in Real-Time RL**.

## Contents

- `index.html`: single-page project site
- `styles.css`: layout, typography, color system, motion
- `script.js`: scroll reveals and figure lightbox
- `assets/figures/`: copied figure assets from the paper
- `.github/workflows/deploy.yml`: GitHub Pages deployment via Actions

## Local preview

From the repo root:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Publishing

The repo is set up to deploy automatically from the `main` branch using GitHub
Pages Actions. After the first push:

1. Open the GitHub repository settings.
2. Go to `Pages`.
3. Confirm the source is `GitHub Actions`.
4. Wait for the `Deploy GitHub Pages` workflow to complete.

The site will publish at:

`https://aneeshers.github.io/realtime-rl/`

## Final wiring

- Replace the paper placeholder in `index.html` with your final PDF or arXiv URL.
- Update any author metadata once the paper front matter is finalized.
