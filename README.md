# Gradient Boosting Practice – Noisy Polynomial App

Front‑end only demo (no frameworks) to generate a synthetic "noisy polynomial" dataset (sometimes quadratic, sometimes cubic), visualize it, and download as CSV. Use this as a starting point for teaching gradient boosting concepts.

## Try it locally

Open `index.html` in a browser. For best results when developing, serve it with a simple static server to avoid any cross‑origin quirks.

Examples:

- Python 3: `python3 -m http.server 8080`
- Node (if installed): `npx serve .`

Then open http://localhost:8080/ in your browser.

## What’s included

- `index.html` – structure and controls
- `styles.css` – minimal styling and responsive canvas
- `script.js` – data generation, canvas plotting, CSV export,
	and an optional regression tree (depth 2) overlay

## Dataset details

- x is sampled uniformly in [−3, 3]
- y is computed from a randomly chosen quadratic or cubic polynomial, with Gaussian noise added
- The underlying function is not drawn and the exact coefficients are not shown to the learner

Use the Regenerate button to create a fresh random sample. Use Download CSV to save the current sample.

You can tweak noise level, sample size, and degree options in `script.js` under the `CONFIG` object.

## Regression tree overlay (depth 2)

Click "Reveal Tree (depth 2) Solution" to overlay a small 1D regression tree (up to 4 step segments) trained with least squares on the current dataset. This mirrors using scikit‑learn’s DecisionTreeRegressor with `max_depth=2` on a single feature. The solution is automatically hidden when you regenerate data.
