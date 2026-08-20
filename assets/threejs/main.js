const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

const gpuMode = String(process.env.K12SIMWORLD_BROWSER_GPU || 'auto').toLowerCase();
function chromiumArgs() {
    const args = ["--no-sandbox", "--disable-setuid-sandbox", "--allow-file-access-from-files"];
    if (gpuMode === 'off') {
        args.push("--disable-gpu", "--enable-unsafe-swiftshader");
    } else {
        args.push("--enable-gpu", "--ignore-gpu-blocklist", "--enable-zero-copy");
        if (gpuMode === 'auto') args.push("--enable-unsafe-swiftshader");
    }
    return args;
}

async function reportGpu(page) {
    const renderer = await page.evaluate(() => {
        const probe = document.createElement('canvas');
        const gl = probe.getContext('webgl') || probe.getContext('experimental-webgl');
        if (!gl) return 'WebGL unavailable';
        const ext = gl.getExtension('WEBGL_debug_renderer_info');
        return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    });
    console.log(`[GPU] requested=${gpuMode}; renderer=${renderer}`);
}

async function recordAnimation(folderPath) {
    const downloadPath = path.join(folderPath, 'downloads');

    const browser = await puppeteer.launch({
        headless: true,
        defaultViewport: null,
        args: chromiumArgs()
    });

    try {
        const page = await browser.newPage();
        let recordingComplete = false;
        let pageError = null;

        // 立即设置事件监听器
        page.on("pageerror", err => {
            pageError = err;
            console.log(`[PAGE ERROR] ${err.message}`);
            console.log(`[PAGE ERROR] ${err.stack}`);
        });

        page.on('console', msg => {
            console.log(`[${msg.type().toUpperCase()}] ${msg.text()}`);
            const text = msg.text() || '';
            if (text.includes('Video download initiated') || text.includes('Recording stopped')) {
                recordingComplete = true;
            }
        });

        page.on('error', err => {
            pageError = err;
            console.log(`[ERROR] ${err.message}`);
        });

        page.on('load', () => {
            console.log(`[LOAD] Page loaded successfully`);
        });

        const client = await page.createCDPSession();
        await client.send('Page.setDownloadBehavior', {
            behavior: 'allow',
            downloadPath: downloadPath
        });

        const htmlPath = path.join(folderPath, 'index.html');
        console.log(`[NAVIGATE] Loading: file://${path.resolve(htmlPath)}`);

        await page.goto(`file://${path.resolve(htmlPath)}`, {
            waitUntil: 'domcontentloaded',
            timeout: 30000
        });

        console.log(`[NAVIGATE] Page navigation completed`);
        await reportGpu(page);

        // 等待一点时间让 JavaScript 执行
        await new Promise(resolve => setTimeout(resolve, 200));

        if (pageError) {
            throw new Error(`Page JavaScript failed: ${pageError.message}`);
        }

        // 智能等待录制完成 - 检查控制台输出 + 轮询文件落盘
        let waitTime = 0;
        const maxWaitTime = 120000; // 最多等待120秒（支持更长录制）
        const fsPollInterval = 250;

        // 轮询等待录制完成
        while (!recordingComplete && waitTime < maxWaitTime) {
            if (pageError) {
                throw new Error(`Page JavaScript failed: ${pageError.message}`);
            }
            await new Promise(resolve => setTimeout(resolve, fsPollInterval));
            waitTime += fsPollInterval;
            try {
                const files = fs.readdirSync(downloadPath);
                if (files.some(f => f.endsWith('.webm'))) {
                    recordingComplete = true;
                }
            } catch (e) {
                // ignore
            }
        }

        if (!recordingComplete) {
            console.log('⚠️ 录制状态检测超时，使用固定等待');
        }

        // 额外缓冲时间确保文件写入完成
        await new Promise(resolve => setTimeout(resolve, 2000));

        const files = fs.readdirSync(downloadPath);
        const videoFile = files.find(file => file.endsWith('.webm'));

        if (videoFile) {
            const oldPath = path.join(downloadPath, videoFile);
            const newPath = path.join(downloadPath, "output.webm");

            // 只有在文件名不是 output.webm 时才重命名
            if (videoFile !== 'output.webm') {
                fs.renameSync(oldPath, newPath);
                console.log(`Video recorded and renamed from ${videoFile} to: output.webm`);
            } else {
                console.log(`Video recorded as: output.webm`);
            }
            console.log(`Saved to: ${newPath}`);
        } else {
            console.error('Video could not be recorded! Available files:', files);
            process.exit(1);
        }

    } catch (error) {
        console.error('An error occurred:', error);
        process.exitCode = 1;
    } finally {
        await browser.close();
    }
}

const folderPath = process.argv[2];

if (!folderPath) {
    console.error('Please provide the folder path as a command line argument');
    process.exit(2);
}

recordAnimation(folderPath).catch(error => {
    console.error(error);
    process.exitCode = 1;
});
