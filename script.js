// Simple front-end app: generate a noisy polynomial dataset, render scatter, allow CSV download.
(function () {
  'use strict';

  // Config with sensible defaults
  const CONFIG = {
    n: 200, // number of samples
    xMin: -3,
    xMax: 3,
    noiseStd: 2.0, // Gaussian noise standard deviation
    // We'll randomly choose degree 2 (quadratic) or 3 (cubic) per dataset
    possibleDegrees: [2, 3],
    padding: 40, // px margins for axes around plot
    pointRadius: 2.0,
  };

  // State
  let data = [];
  let xRange = [CONFIG.xMin, CONFIG.xMax];
  let yRange = [-10, 10]; // will tighten to data after generation
  let currentDegree = 3;
  let currentCoeffs = [3, 2, -1.2, 0.5];

  // Elements
  const canvas = document.getElementById('plot');
  const regenBtn = document.getElementById('regenerateBtn');
  const downloadBtn = document.getElementById('downloadCsvBtn');
  const toggleSolutionBtn = document.getElementById('toggleSolutionBtn');
  const solutionPanel = document.getElementById('solutionPanel');
  const predictionsTable = document.getElementById('predictionsTable').querySelector('tbody');
  const downloadPredCsvBtn = document.getElementById('downloadPredCsvBtn');

  // Helpers
  function poly(x, coeffs) {
    // Horner's method for stability
    let y = 0;
    for (let i = coeffs.length - 1; i >= 0; i--) {
      y = y * x + coeffs[i];
    }
    return y;
  }

  // Box–Muller transform for Gaussian noise
  function randn() {
    let u = 0, v = 0;
    // Avoid 0 to prevent log(0)
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  function randBetween(a, b) { return a + Math.random() * (b - a); }
  function randSign() { return Math.random() < 0.5 ? -1 : 1; }

  function randomCoeffs(deg) {
    // Generate "nice" coefficients that keep values reasonable on [-3, 3]
    if (deg === 2) {
      const c2 = randBetween(0.5, 1.2) * randSign();
      const c1 = randBetween(-2.0, 2.0);
      const c0 = randBetween(-2.0, 4.0);
      return [c0, c1, c2];
    } else {
      const c3 = randBetween(0.2, 0.7) * randSign();
      const c2 = randBetween(-1.5, 1.5);
      const c1 = randBetween(-2.0, 2.0);
      const c0 = randBetween(-2.0, 4.0);
      return [c0, c1, c2, c3];
    }
  }

  function pickDegree() {
    return CONFIG.possibleDegrees[Math.floor(Math.random() * CONFIG.possibleDegrees.length)];
  }

  function generateData() {
    currentDegree = pickDegree();
    currentCoeffs = randomCoeffs(currentDegree);

    const pts = [];
    for (let i = 0; i < CONFIG.n; i++) {
      const t = i / (CONFIG.n - 1);
      const x = CONFIG.xMin + t * (CONFIG.xMax - CONFIG.xMin);
      const y = poly(x, currentCoeffs) + CONFIG.noiseStd * randn();
      pts.push({ x, y });
    }
    data = pts;
    // derive yRange with a small margin
    const ys = data.map(p => p.y);
    const yMin = Math.min(...ys);
    const yMax = Math.max(...ys);
    const pad = 0.08 * (yMax - yMin || 1);
    yRange = [yMin - pad, yMax + pad];
  }

  function setCanvasSize() {
    // Handle device pixel ratio for crisp drawing
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(300, Math.floor(rect.width * dpr));
    canvas.height = Math.max(200, Math.floor(rect.height * dpr));
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  // 1D regression tree trainer (max depth = 2)
  // Returns a small tree with up to 3 thresholds, forming up to 4 constant segments.
  function trainDepth2Tree(points) {
    if (points.length === 0) return null;
    // Sort by x
    const pts = points.slice().sort((a, b) => a.x - b.x);
    const n = pts.length;

    const prefixSum = new Array(n + 1).fill(0);
    const prefixSumSq = new Array(n + 1).fill(0);
    for (let i = 0; i < n; i++) {
      prefixSum[i + 1] = prefixSum[i] + pts[i].y;
      prefixSumSq[i + 1] = prefixSumSq[i] + pts[i].y * pts[i].y;
    }

    function segmentError(l, r) {
      // inclusive [l, r]
      const len = r - l + 1;
      if (len <= 0) return 0;
      const s = prefixSum[r + 1] - prefixSum[l];
      const s2 = prefixSumSq[r + 1] - prefixSumSq[l];
      const mean = s / len;
      // SSE = sum(y^2) - n * mean^2
      return s2 - len * mean * mean;
    }
    function segmentMean(l, r) {
      const len = r - l + 1;
      if (len <= 0) return 0;
      const s = prefixSum[r + 1] - prefixSum[l];
      return s / len;
    }

    // Depth-1 split candidates on the whole range
    function bestSplit(l, r) {
      // find best i in [l, r-1]
      let best = { idx: -1, sse: Infinity };
      for (let i = l; i < r; i++) {
        if (pts[i].x === pts[i + 1].x) continue;
        const sse = segmentError(l, i) + segmentError(i + 1, r);
        if (sse < best.sse) best = { idx: i, sse };
      }
      return best;
    }

    // Root split
    const root = bestSplit(0, n - 1);
    if (root.idx === -1) {
      const mean = segmentMean(0, n - 1);
      return { leaves: [{ range: [0, n - 1], mean }], thresholds: [] };
    }
    const thr0 = (pts[root.idx].x + pts[root.idx + 1].x) * 0.5;

    // Left child split
    const left = { l: 0, r: root.idx };
    const leftBest = bestSplit(left.l, left.r);
    let thrL = null;
    if (leftBest.idx !== -1) thrL = (pts[leftBest.idx].x + pts[leftBest.idx + 1].x) * 0.5;

    // Right child split
    const right = { l: root.idx + 1, r: n - 1 };
    const rightBest = bestSplit(right.l, right.r);
    let thrR = null;
    if (rightBest.idx !== -1) thrR = (pts[rightBest.idx].x + pts[rightBest.idx + 1].x) * 0.5;

    // Build leaves (up to 4)
    const leaves = [];
    if (thrL !== null) {
      // left-left
      leaves.push({ range: [left.l, leftBest.idx], mean: segmentMean(left.l, leftBest.idx) });
      // left-right
      leaves.push({ range: [leftBest.idx + 1, left.r], mean: segmentMean(leftBest.idx + 1, left.r) });
    } else {
      // single left leaf
      leaves.push({ range: [left.l, left.r], mean: segmentMean(left.l, left.r) });
    }
    if (thrR !== null) {
      // right-left
      leaves.push({ range: [right.l, rightBest.idx], mean: segmentMean(right.l, rightBest.idx) });
      // right-right
      leaves.push({ range: [rightBest.idx + 1, right.r], mean: segmentMean(rightBest.idx + 1, right.r) });
    } else {
      // single right leaf
      leaves.push({ range: [right.l, right.r], mean: segmentMean(right.l, right.r) });
    }

    // Collect thresholds and dedupe
    const eps = 1e-9;
    const thresholds = [thr0];
    if (thrL !== null) thresholds.push(thrL);
    if (thrR !== null) thresholds.push(thrR);
    const uniqThresh = thresholds
      .slice()
      .sort((a, b) => a - b)
      .filter((v, i, arr) => i === 0 || Math.abs(v - arr[i - 1]) > eps);

    return { leaves, thresholds: uniqThresh, pts };
  }

  let showSolution = false;
  let tree = null;

  function mapX(x, w) {
    const [xmin, xmax] = xRange;
    const px = CONFIG.padding + (x - xmin) * (w - 2 * CONFIG.padding) / (xmax - xmin);
    return px;
  }

  function mapY(y, h) {
    const [ymin, ymax] = yRange;
    // y grows downward on canvas
    const py = h - CONFIG.padding - (y - ymin) * (h - 2 * CONFIG.padding) / (ymax - ymin);
    return py;
  }

  function drawAxes(ctx, w, h) {
    ctx.save();
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;

    // Frame
    ctx.strokeRect(CONFIG.padding, CONFIG.padding, w - 2 * CONFIG.padding, h - 2 * CONFIG.padding);

    // Zero axes if in range
    // y=0 horizontal line
    if (yRange[0] < 0 && yRange[1] > 0) {
      ctx.beginPath();
      ctx.moveTo(CONFIG.padding, mapY(0, h));
      ctx.lineTo(w - CONFIG.padding, mapY(0, h));
      ctx.stroke();
    }
    // x=0 vertical line
    if (xRange[0] < 0 && xRange[1] > 0) {
      ctx.beginPath();
      ctx.moveTo(mapX(0, w), CONFIG.padding);
      ctx.lineTo(mapX(0, w), h - CONFIG.padding);
      ctx.stroke();
    }

    // Ticks and labels (simple 5 ticks each)
    ctx.fillStyle = '#9ca3af';
    ctx.font = '12px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let i = 0; i <= 5; i++) {
      const t = i / 5;
      const xv = xRange[0] + t * (xRange[1] - xRange[0]);
      const xpx = mapX(xv, w);
      ctx.beginPath();
      ctx.moveTo(xpx, h - CONFIG.padding);
      ctx.lineTo(xpx, h - CONFIG.padding + 4);
      ctx.stroke();
      ctx.fillText(xv.toFixed(1), xpx, h - CONFIG.padding + 6);
    }
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 5; i++) {
      const t = i / 5;
      const yv = yRange[0] + t * (yRange[1] - yRange[0]);
      const ypx = mapY(yv, h);
      ctx.beginPath();
      ctx.moveTo(CONFIG.padding - 4, ypx);
      ctx.lineTo(CONFIG.padding, ypx);
      ctx.stroke();
      ctx.fillText(yv.toFixed(1), CONFIG.padding - 6, ypx);
    }

    ctx.restore();
  }

  function drawData(ctx, w, h) {
    // Draw points only (no underlying function curve)
    ctx.save();
    ctx.fillStyle = '#22d3ee';
    const r = CONFIG.pointRadius;
    for (const p of data) {
      const x = mapX(p.x, w);
      const y = mapY(p.y, h);
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  function drawTree(ctx, w, h) {
    if (!showSolution || !tree) return;
    ctx.save();
    // Draw thresholds
    ctx.strokeStyle = '#f59e0b';
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = 1.5;
    // Sort and dedupe thresholds to ensure proper interval ordering
    const eps = 1e-9;
    const sortedThresh = (tree.thresholds || [])
      .slice()
      .sort((a, b) => a - b)
      .filter((v, i, arr) => i === 0 || Math.abs(v - arr[i - 1]) > eps)
      .filter(v => v > xRange[0] + eps && v < xRange[1] - eps);

    for (const thr of sortedThresh) {
      const xThr = mapX(thr, w);
      ctx.beginPath();
      ctx.moveTo(xThr, CONFIG.padding);
      ctx.lineTo(xThr, h - CONFIG.padding);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Draw piecewise constant segments across the full x-range
    ctx.strokeStyle = '#f43f5e';
    ctx.lineWidth = 2.5;
    // Build sorted boundaries
    const bounds = [xRange[0], ...sortedThresh, xRange[1]];
    // Determine mean for each interval by finding a representative x
    for (let i = 0; i < bounds.length - 1; i++) {
      const a = bounds[i];
      const b = bounds[i + 1];
      if (b - a <= eps) continue; // skip zero/negative width
      // pick midpoint and find which leaf covers it
      const mid = (a + b) * 0.5;
      let mean = null;
      // Convert mid to nearest index range using tree.pts ordering
      const pts = tree.pts || data.slice().sort((u, v) => u.x - v.x);
      // binary search index of last x <= mid
      let lo = 0, hi = pts.length - 1, idx = 0;
      while (lo <= hi) {
        const m = (lo + hi) >> 1;
        if (pts[m].x <= mid) { idx = m; lo = m + 1; } else { hi = m - 1; }
      }
      for (const leaf of tree.leaves) {
        if (idx >= leaf.range[0] && idx <= leaf.range[1]) { mean = leaf.mean; break; }
      }
      if (mean === null) continue;
      ctx.beginPath();
      ctx.moveTo(mapX(a, w), mapY(mean, h));
      ctx.lineTo(mapX(b, w), mapY(mean, h));
      ctx.stroke();
    }
    ctx.restore();
  }

  function render() {
    const ctx = setCanvasSize();
    const rect = canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;

    // clear
    ctx.clearRect(0, 0, w, h);

    drawAxes(ctx, w, h);
    drawData(ctx, w, h);
    drawTree(ctx, w, h);
  }

  function predictOne(xVal) {
    if (!tree || !tree.pts || !tree.leaves) return null;
    // Find index of last x <= xVal in sorted pts
    const pts = tree.pts;
    let lo = 0, hi = pts.length - 1, idx = 0;
    while (lo <= hi) {
      const m = (lo + hi) >> 1;
      if (pts[m].x <= xVal) { idx = m; lo = m + 1; } else { hi = m - 1; }
    }
    for (const leaf of tree.leaves) {
      if (idx >= leaf.range[0] && idx <= leaf.range[1]) return leaf.mean;
    }
    // Fallback to nearest leaf mean by x
    return tree.leaves.length ? tree.leaves[0].mean : null;
  }

  function buildPredictions() {
    // returns array of {x, y, y_pred}
    return data.map(p => ({ x: p.x, y: p.y, y_pred: predictOne(p.x) }));
  }

  function updatePredictionsTable() {
    predictionsTable.innerHTML = '';
    if (!showSolution) return;
    const rows = buildPredictions();
    const frag = document.createDocumentFragment();
    for (const r of rows) {
      const tr = document.createElement('tr');
      const tdX = document.createElement('td');
      const tdY = document.createElement('td');
      const tdYp = document.createElement('td');
      tdX.textContent = r.x.toFixed(6);
      tdY.textContent = r.y.toFixed(6);
      tdYp.textContent = (r.y_pred ?? '').toFixed ? r.y_pred.toFixed(6) : '';
      tr.append(tdX, tdY, tdYp);
      frag.appendChild(tr);
    }
    predictionsTable.appendChild(frag);
  }

  function toPredCSV(rows) {
    const header = 'x,y,y_pred\n';
    const body = rows.map(r => `${r.x},${r.y},${r.y_pred}`).join('\n');
    return header + body + '\n';
  }

  function downloadPredCSV() {
    const rows = buildPredictions();
    const csv = toPredCSV(rows);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    a.href = url;
    a.download = `predictions_depth2_${ts}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function toCSV(rows) {
    const header = 'x,y\n';
    const body = rows.map(p => `${p.x},${p.y}`).join('\n');
    return header + body + '\n';
  }

  function downloadCSV() {
    const csv = toCSV(data);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    a.href = url;
    a.download = `noisy_polynomial_${ts}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function handleResize() {
    render();
  }

  // Wire up events
  regenBtn.addEventListener('click', () => {
    // Regenerate data and reset solution visibility
    generateData();
    tree = trainDepth2Tree(data);
    showSolution = false;
    toggleSolutionBtn.textContent = 'Reveal Tree (depth 2) Solution';
    solutionPanel.classList.add('hidden');
    solutionPanel.setAttribute('aria-hidden', 'true');
    predictionsTable.innerHTML = '';
    render();
  });
  downloadBtn.addEventListener('click', downloadCSV);
  downloadPredCsvBtn.addEventListener('click', downloadPredCSV);
  toggleSolutionBtn.addEventListener('click', () => {
    showSolution = !showSolution;
    toggleSolutionBtn.textContent = showSolution ? 'Hide Tree (depth 2) Solution' : 'Reveal Tree (depth 2) Solution';
    if (showSolution && !tree) tree = trainDepth2Tree(data);
    if (showSolution) {
      solutionPanel.classList.remove('hidden');
      solutionPanel.setAttribute('aria-hidden', 'false');
      updatePredictionsTable();
    } else {
      solutionPanel.classList.add('hidden');
      solutionPanel.setAttribute('aria-hidden', 'true');
    }
    render();
  });
  window.addEventListener('resize', handleResize);

  // Initial
  generateData();
  tree = trainDepth2Tree(data);
  render();
  // Keep solution panel hidden initially
  solutionPanel.classList.add('hidden');
  solutionPanel.setAttribute('aria-hidden', 'true');
})();
