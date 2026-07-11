// Batch-render LaTeX expressions to standalone SVG files using MathJax v3.
//
// Usage: node render_math.mjs <input.json> <output_dir>
//
// input.json: [{"id": "m0001", "tex": "x^2", "display": true}, ...]
// Writes <output_dir>/<id>.svg for each entry plus <output_dir>/metrics.json
// with {id: {width_ex, height_ex, valign_ex}} used for sizing the <img> tags.

import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);

const [inputPath, outDir] = process.argv.slice(2);
if (!inputPath || !outDir) {
  console.error('usage: node render_math.mjs <input.json> <output_dir>');
  process.exit(1);
}

const entries = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
fs.mkdirSync(outDir, { recursive: true });

const MathJax = await require('mathjax').init({
  loader: { load: ['input/tex', 'output/svg'] },
  tex: {
    packages: ['base', 'ams', 'newcommand', 'noundefined', 'boldsymbol', 'color', 'cancel'],
  },
  svg: { fontCache: 'local' },
});

const adaptor = MathJax.startup.adaptor;
const metrics = {};
let failures = 0;

for (const { id, tex, display } of entries) {
  let svgNode;
  try {
    const container = await MathJax.tex2svgPromise(tex, {
      display: !!display,
      em: 16,
      ex: 8,
      containerWidth: 10000, // avoid premature line-breaking; reader scales instead
    });
    svgNode = adaptor.firstChild(container);
  } catch (err) {
    failures++;
    console.error(`FAIL ${id}: ${err.message}\n  tex: ${tex}`);
    continue;
  }

  // MathJax reports an error inline via <merror>/data-mjx-error rather than throwing.
  const html = adaptor.outerHTML(svgNode);
  if (html.includes('data-mjx-error')) {
    failures++;
    console.error(`ERRNODE ${id}: ${tex}`);
    continue;
  }

  const getEx = (attr) => {
    const v = adaptor.getAttribute(svgNode, attr) || '';
    const m = v.match(/^(-?[\d.]+)ex$/);
    return m ? parseFloat(m[1]) : null;
  };
  const style = adaptor.getAttribute(svgNode, 'style') || '';
  const va = style.match(/vertical-align:\s*(-?[\d.]+)ex/);

  metrics[id] = {
    width_ex: getEx('width'),
    height_ex: getEx('height'),
    valign_ex: va ? parseFloat(va[1]) : 0,
  };

  // E-readers may not resolve currentColor; pin to black.
  let out = html.replace(/currentColor/g, '#000000');
  // Standalone SVG files need the xmlns (tex2svg emits it already, but be safe).
  if (!out.includes('xmlns=')) {
    out = out.replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ');
  }
  fs.writeFileSync(path.join(outDir, `${id}.svg`), out);
}

fs.writeFileSync(path.join(outDir, 'metrics.json'), JSON.stringify(metrics, null, 1));
console.log(`rendered ${Object.keys(metrics).length}/${entries.length} expressions (${failures} failures)`);
if (failures > 0) process.exit(2);
