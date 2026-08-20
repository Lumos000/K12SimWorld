const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

function chromiumArgs(gpuMode) {
  const args = ['--no-sandbox', '--disable-setuid-sandbox', '--allow-file-access-from-files'];
  if (gpuMode === 'off') {
    args.push('--disable-gpu', '--enable-unsafe-swiftshader');
  } else {
    args.push('--enable-gpu', '--ignore-gpu-blocklist', '--enable-zero-copy');
    if (gpuMode === 'auto') args.push('--enable-unsafe-swiftshader');
  }
  return args;
}

async function launch(gpuMode) {
  try {
    return await puppeteer.launch({headless: true, defaultViewport: {width: 512, height: 512, deviceScaleFactor: 1}, args: chromiumArgs(gpuMode)});
  } catch (error) {
    if (gpuMode === 'off') throw error;
    console.log(`[GPU] accelerated Chromium launch failed; falling back to software: ${error.message}`);
    return puppeteer.launch({headless: true, defaultViewport: {width: 512, height: 512, deviceScaleFactor: 1}, args: chromiumArgs('off')});
  }
}

async function gpuInfo(page) {
  return page.evaluate(() => {
    const probe = document.createElement('canvas');
    const gl = probe.getContext('webgl') || probe.getContext('experimental-webgl');
    if (!gl) return 'WebGL unavailable (2D Canvas capture remains supported)';
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  });
}

async function capture(folderPath, fps, duration, gpuMode) {
  const framesPath = path.join(folderPath, 'frames');
  fs.mkdirSync(framesPath, {recursive: true});
  const browser = await launch(gpuMode);
  try {
    const page = await browser.newPage();
    let pageError = null;
    page.on('pageerror', error => { pageError = error; console.log(`[PAGE ERROR] ${error.message}`); });
    page.on('error', error => { pageError = error; console.log(`[ERROR] ${error.message}`); });
    page.on('console', message => console.log(`[${message.type().toUpperCase()}] ${message.text()}`));
    await page.evaluateOnNewDocument(() => { window.__k12simFastCaptureRequested = true; });
    const htmlPath = path.resolve(path.join(folderPath, 'index.html'));
    console.log(`[NAVIGATE] Loading: file://${htmlPath}`);
    await page.goto(`file://${htmlPath}`, {waitUntil: 'domcontentloaded', timeout: 30000});
    await new Promise(resolve => setTimeout(resolve, 100));
    if (pageError) throw new Error(`Page JavaScript failed: ${pageError.message}`);
    const hasHook = await page.evaluate(() => typeof window.__k12simRenderFrame === 'function');
    if (!hasHook) throw new Error('document does not expose window.__k12simRenderFrame');
    const canvas = await page.$('[data-k12-recording="true"], canvas');
    if (!canvas) throw new Error('document has no canvas');
    console.log(`[GPU] requested=${gpuMode}; renderer=${await gpuInfo(page)}`);
    const count = Math.max(2, Math.round(fps * duration));
    const width = String(count - 1).length;
    for (let index = 0; index < count; index += 1) {
      const progress = index / (count - 1);
      await page.evaluate(value => window.__k12simRenderFrame(value), progress);
      const filename = `frame_${String(index).padStart(width, '0')}.png`;
      await canvas.screenshot({path: path.join(framesPath, filename), type: 'png'});
    }
    console.log(`[FAST CAPTURE] wrote ${count} deterministic frames at ${fps} fps for ${duration}s`);
  } finally {
    await browser.close();
  }
}

const folderPath = process.argv[2];
const fps = Number(process.argv[3] || 5);
const duration = Number(process.argv[4] || 8);
const gpuMode = String(process.env.K12SIMWORLD_BROWSER_GPU || 'auto').toLowerCase();
if (!folderPath || !(fps > 0) || !(duration > 0) || !['auto', 'on', 'off'].includes(gpuMode)) {
  console.error('usage: node fast_main.js FOLDER FPS DURATION; K12SIMWORLD_BROWSER_GPU=auto|on|off');
  process.exit(2);
}
capture(folderPath, fps, duration, gpuMode).catch(error => { console.error(error); process.exitCode = 1; });
