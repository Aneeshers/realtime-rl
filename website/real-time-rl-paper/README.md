# real-time-rl-paper

Static paper page for *Finding the Time to Think: Adaptive MCTS in Real-Time RL*.

## Design goals

- Berkeley Mono everywhere, including body copy and code
- White background, minimal borders, no gradients
- Centered paper-style hero with compact link buttons
- Opening gifs for AlphaZero-style planning and real-time delay
- Figure-first layout with math support
- KaTeX support for inline and display math
- No framework or build step
- Works on GitHub Pages

## Files

- `index.html`: page shell
- `content.js`: all paper-specific content lives here
- `script.js`: renderer for the hero, sections, links, and metadata
- `styles.css`: minimal research-oriented styling
- `assets/figures/`: paper figures and gifs
- `.github/workflows/deploy.yml`: GitHub Pages deployment workflow

## Reusing for a new project

1. Edit `content.js`.
2. Replace the figure assets in `assets/figures/`.
3. Update any paper/code links.
4. Push to a public GitHub repo and enable Pages with `GitHub Actions`.

## Local preview

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Notes

- Inline math works with `$...$`.
- Display math works with `$$...$$`.
- Content strings are trusted HTML, so you can use tags like `<code>` and `<em>` inside `content.js`.
- If you want a different body font later, change the `@font-face` and `body` font stack in `styles.css`.
