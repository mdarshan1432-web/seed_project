﻿/* =========================================================================
   AgroSeed.AI  -  Real-Time Agriculture Seeds Detection & Recognition
   script.js   :   Frontend logic for index.html
   -------------------------------------------------------------------------
   Sections:
     1.  Tiny helpers (dom, fetch, toast, throttle, debounce)
     2.  Light mode setup
     3.  Hero stat-card counter animation
     4.  Camera controller (enumerate -> open -> auto-detect loop)
     5.  Upload controller (drag/drop + browse + preview)
     6.  Predict pipeline (POST /predict, /predict_frame)
     7.  Result rendering (class, confidence bar, top-5, KA info, web)
     8.  Mandi search & sort, chip jump-to-card
     9.  Footer health-status fetch
    10.  Smooth scroll & nav-link active highlight
    11.  Boot
   ========================================================================= */

(function () {
    "use strict";

    /* ===================================================================
     * 1.  TINY HELPERS
     * =================================================================== */
    const $  = (sel, parent = document) => parent.querySelector(sel);
    const $$ = (sel, parent = document) => Array.from(parent.querySelectorAll(sel));

    const log = (...args) => console.log("%c[AgroSeed]", "color:#0e6b3a;font-weight:bold", ...args);
    const warn = (...args) => console.warn("[AgroSeed]", ...args);

    function throttle(fn, ms = 250) {
        let last = 0, timer = null;
        return function (...a) {
            const now = Date.now();
            const wait = ms - (now - last);
            if (wait <= 0) {
                last = now;
                fn.apply(this, a);
            } else {
                clearTimeout(timer);
                timer = setTimeout(() => { last = Date.now(); fn.apply(this, a); }, wait);
            }
        };
    }
    function debounce(fn, ms = 220) {
        let t;
        return function (...a) {
            clearTimeout(t);
            t = setTimeout(() => fn.apply(this, a), ms);
        };
    }

    async function jsonPost(url, body) {
        const headers = { "Accept": "application/json" };
        let opts = { method: "POST", headers };
        const isFormData = body instanceof FormData || (body && body.constructor && body.constructor.name === "FormData");
        if (isFormData) {
            opts.body = body;
        } else {
            headers["Content-Type"] = "application/json";
            opts.body = JSON.stringify(body || {});
        }
        const r = await fetch(url, opts);
        const data = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(data.error || ("HTTP " + r.status));
        return data;
    }
    async function jsonGet(url) {
        const r = await fetch(url, { headers: { "Accept": "application/json" } });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(data.error || ("HTTP " + r.status));
        return data;
    }

    /* ----------------------------------------------------- toast ------- */
    const toastEl = $("#toast");
    function toast(msg, type = "ok") {
        if (!toastEl) return;
        toastEl.textContent = msg;
        toastEl.dataset.type = type;
        toastEl.classList.add("show");
        clearTimeout(toastEl._t);
        toastEl._t = setTimeout(() => toastEl.classList.remove("show"), 3200);
    }

    const SEED_NAME_KN = {
        almond: "ಬಾದಾಮಿ",
        amla: "ನೆಲ್ಲಿಕಾಯಿ",
        apple: "ಸೇಬು",
        avocado: "ಅವಕಾಡೊ",
        betel_nut: "ಅಡಿಕೆ",
        bitter_melon: "ಹಾಗಲಕಾಯಿ",
        black_plum: "ನೇರಳೆ",
        bottle_gourd_long: "ಸೋರೆಕಾಯಿ",
        carrot: "ಕ್ಯಾರೆಟ್",
        cashew: "ಗೋಡಂಬಿ",
        cauliflower: "ಹೂಕೋಸು",
        chilli: "ಮೆಣಸಿನಕಾಯಿ",
        chinese_date: "ಬೋರೆಹಣ್ಣು",
        coriander_leaves: "ಕೊತ್ತಂಬರಿ ಸೊಪ್ಪು",
        date: "ಖರ್ಜೂರ",
        guava: "ಪೇರಳೆ",
        hog_plum: "ಅಂಬಟೆ",
        hyacinth_bean: "ಅವರೆಕಾಯಿ",
        jackfruit: "ಹಲಸು",
        litchi: "ಲೀಚಿ",
        malabar_spinach: "ಬಸಳೆ ಸೊಪ್ಪು",
        mango: "ಮಾವಿನಹಣ್ಣು",
        okra: "ಬೆಂಡೆಕಾಯಿ",
        olive: "ಆಲಿವ್",
        onion: "ಈರುಳ್ಳಿ",
        orange: "ಕಿತ್ತಳೆ",
        palm: "ತಾಳೆ",
        papaya: "ಪಪ್ಪಾಯಿ",
        pea: "ಬಟಾಣಿ",
        pumpkin: "ಕುಂಬಳಕಾಯಿ",
        radish: "ಮೂಲಂಗಿ",
        ribbed_gourd: "ಹೀರೆಕಾಯಿ",
        sapodilla: "ಚಿಕ್ಕು",
        spinach: "ಪಾಲಕ್ ಸೊಪ್ಪು",
        tamarind: "ಹುಣಸೆ",
        tomato: "ಟೊಮ್ಯಾಟೊ",
        water_spinach: "ನೀರು ಬಸಳೆ",
        ash_gourd: "ಬೂದು ಕುಂಬಳಕಾಯಿ",
        bitter_gourd: "ಹಾಗಲಕಾಯಿ",
        chilli_green_long: "ಉದ್ದ ಹಸಿರು ಮೆಣಸಿನಕಾಯಿ",
        cluster_beans: "ಗೋರಿಕಾಯಿ",
        corn: "ಜೋಳ",
        cucumber: "ಸೌತೆಕಾಯಿ",
        cucumber_green_long: "ಉದ್ದ ಹಸಿರು ಸೌತೆಕಾಯಿ",
        dill_leaves: "ಸಬ್ಬಸಿಗೆ ಸೊಪ್ಪು",
        ladys_finger: "ಬೆಂಡೆಕಾಯಿ",
        palak: "ಪಾಲಕ್",
        pole_beans: "ಅವರೆಕಾಯಿ",
        ridge_gourd_long: "ಉದ್ದ ಹೀರೆಕಾಯಿ",
        snake_gourd: "ಪಡವಲಕಾಯಿ",
        snake_gourd_green_long: "ಉದ್ದ ಹಸಿರು ಪಡವಲಕಾಯಿ",
        sponge_gourd: "ತುರಾಯಿ",
        string_bean: "ಹುರಳಿಕಾಯಿ",
        wax_gourd: "ಬೂದು ಕುಂಬಳಕಾಯಿ",
        unknown: "ಗೊತ್ತಿಲ್ಲದ ಬೀಜ",
    };

    function seedKey(name) {
        return (name || "")
            .toString()
            .trim()
            .toLowerCase()
            .replace(/\s+/g, "_")
            .replace(/_+/g, "_");
    }

    function seedLabel(name, lang = currentLang) {
        const raw = (name || "").toString().trim();
        if (lang === "kn") {
            return SEED_NAME_KN[seedKey(raw)] || humanise(raw);
        }
        return humanise(raw);
    }

    function translateSeedLabels() {
        $$("#classes .chip").forEach(chip => {
            const original = chip.dataset.seedName || chip.textContent;
            chip.dataset.seedName = original.trim();
            chip.textContent = `🌱 ${seedLabel(original)}`;
            chip.title = humanise(original);
        });
        if (lastResult?.predicted_class && resultPanel?.style.display !== "none") {
            const wi = lastResult.web_info || {};
            resultClass.textContent = currentLang === "kn" && wi.seed_name_kn
                ? wi.seed_name_kn
                : seedLabel(lastResult.predicted_class);
            top5List.innerHTML = "";
            (lastResult.top5 || []).forEach(t => {
                const li = document.createElement("li");
                li.innerHTML = `
                    <span title="${humanise(t.class)}">${seedLabel(t.class)}</span>
                    <span class="pct">${(t.confidence * 100).toFixed(1)}%</span>`;
                top5List.appendChild(li);
            });
            drawResultBoxes(lastResult.yolo || []);
        }
    }


    /* ===================================================================
     * 2.  THEME SETUP (Light / Dark Mode)
     * =================================================================== */
    function applyTheme(theme) {
        const saved = localStorage.getItem("theme");
        const t = theme || saved || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
        
        document.documentElement.dataset.theme = t;
        try { localStorage.setItem("theme", t); } catch (e) {}
        
        const btn = $("#themeToggle");
        if (btn) {
            btn.innerHTML = t === "dark" ? "☀️" : "🌙";
            btn.title = t === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode";
        }
    }
    applyTheme();


    /* ===================================================================
     * 3.  HERO COUNTER ANIMATION (count-up on view)
     * =================================================================== */
    function animateCounter(el) {
        const target = parseInt(el.dataset.count || "0", 10);
        if (!target) { el.textContent = "0"; return; }
        const dur = 1100;
        const start = performance.now();

        function tick(ts) {
            const p = Math.min(1, (ts - start) / dur);
            const eased = 1 - Math.pow(1 - p, 3);
            el.textContent = Math.round(target * eased).toLocaleString();
            if (p < 1) requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
    }

    const counters = $$(".stat-num[data-count]");
    if ("IntersectionObserver" in window && counters.length) {
        const io = new IntersectionObserver((entries) => {
            entries.forEach(en => {
                if (en.isIntersecting) {
                    animateCounter(en.target);
                    io.unobserve(en.target);
                }
            });
        }, { threshold: 0.4 });
        counters.forEach(c => io.observe(c));
    } else {
        counters.forEach(animateCounter);
    }


    /* ===================================================================
     * 4.  CAMERA CONTROLLER
     * =================================================================== */
    const video        = $("#cameraVideo");
    const overlay      = $("#cameraOverlay");
    const cameraDebug  = $("#cameraDebug");
    const cameraEmpty  = $("#cameraEmpty");
    const cameraStatus = $("#cameraStatus");
    const liveBadge    = $("#liveBadge");
    const btnOn        = $("#btnCameraOn");
    const btnOff       = $("#btnCameraOff");
    const btnCapture   = $("#btnCapture");
    const autoDetect   = $("#autoDetect");
    const cameraSelect = $("#cameraSelect");

    let stream      = null;
    let detectTimer = null;
    let inFlight    = false;
    let dynInterval = 700;
    let lastCameraFrameSize = null;

    function onVideoReady() {
        if (!video || !video.videoWidth || !video.videoHeight) return;
        cameraEmpty.style.display = "none";
        video.style.visibility = "visible";
        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;
    }

    if (video) {
        video.addEventListener("loadeddata", onVideoReady);
        video.addEventListener("playing", onVideoReady);
        video.addEventListener("error", (event) => {
            warn("Video playback error", event);
            toast(cameraText("accessError") + " video playback failed", "err");
        });
    }

    function cameraText(key) {
        const kn = currentLang === "kn";
        const text = {
            unsupported: kn ? "ಈ ಬ್ರೌಸರ್‌ನಲ್ಲಿ ಕ್ಯಾಮೆರಾ ಬೆಂಬಲ ಇಲ್ಲ" : "Camera not supported in this browser",
            live: kn ? "ಕ್ಯಾಮೆರಾ ಲೈವ್" : "Camera Live",
            off: kn ? "ಕ್ಯಾಮೆರಾ ಆಫ್" : "Camera Off",
            error: kn ? "ಕ್ಯಾಮೆರಾ ದೋಷ" : "Camera Error",
            started: kn ? "ಕ್ಯಾಮೆರಾ ಆರಂಭವಾಗಿದೆ" : "Camera started",
            accessError: kn ? "ಕ್ಯಾಮೆರಾ ಪ್ರವೇಶಿಸಲು ಆಗಲಿಲ್ಲ: " : "Could not access camera: ",
            waiting: kn ? "ಕ್ಯಾಮೆರಾ ಫ್ರೇಮ್ ಪರಿಶೀಲಿಸಲಾಗುತ್ತಿದೆ..." : "Waiting for frame analysis...",
        };
        return text[key] || key;
    }

    async function listCameras() {
        if (!navigator.mediaDevices?.enumerateDevices) return;
        try {
            const devs = await navigator.mediaDevices.enumerateDevices();
            console.log("listCameras: devices=", devs);
            const cams = devs.filter(d => d.kind === "videoinput");
            if (cameraDebug) cameraDebug.textContent = `Devices: ${devs.length} (cams=${cams.length})`;
            const selected = cameraSelect?.value || "facing:environment";
            cameraSelect.innerHTML = "";
            [
                ["facing:environment", currentLang === "kn" ? "ಹಿಂಬದಿ ಕ್ಯಾಮೆರಾ" : "Back camera"],
                ["facing:user", currentLang === "kn" ? "ಮುಂಭಾಗದ ಕ್ಯಾಮೆರಾ" : "Front camera"],
            ].forEach(([value, label]) => {
                const o = document.createElement("option");
                o.value = value;
                o.textContent = label;
                cameraSelect.appendChild(o);
            });
            cams.forEach((c, i) => {
                const o = document.createElement("option");
                o.value = c.deviceId;
                o.textContent = c.label || `Camera ${i + 1}`;
                cameraSelect.appendChild(o);
            });
            if ([...cameraSelect.options].some(o => o.value === selected)) {
                cameraSelect.value = selected;
            }
            if (!cams.length) {
                const o = document.createElement("option");
                o.textContent = "No camera detected";
                o.disabled = true;
                cameraSelect.appendChild(o);
                if (cameraDebug) cameraDebug.textContent = "No camera devices detected by enumerateDevices.";
            }
        } catch (e) {
            warn("enumerateDevices error", e);
            if (cameraDebug) cameraDebug.textContent = `enumerateDevices error: ${e && e.message ? e.message : e}`;
        }
    }

    function buildCameraConstraints(deviceId) {
        const quality = { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, max: 60 } };
        if (!deviceId) {
            return { video: { facingMode: "environment", ...quality }, audio: false };
        }
        if (deviceId.startsWith("facing:")) {
            const facing = deviceId.replace("facing:", "");
            return { video: { facingMode: { ideal: facing }, ...quality }, audio: false };
        }
        return { video: { deviceId: { exact: deviceId }, ...quality }, audio: false };
    }

    async function startCamera() {
        if (!navigator.mediaDevices?.getUserMedia) {
            toast(cameraText("unsupported"), "err");
            return;
        }

        // Reset any existing camera stream before starting a new one.
        stopCamera();
        const deviceId = cameraSelect?.value;
        let constraints = buildCameraConstraints(deviceId);

        async function attempt(constraintsToTry) {
            try {
                const s = await navigator.mediaDevices.getUserMedia(constraintsToTry);
                return s;
            } catch (err) {
                warn("getUserMedia attempt failed", err, constraintsToTry);
                return null;
            }
        }

        try {
            stream = await attempt(constraints);
            if (!stream && deviceId && !deviceId.startsWith("facing:")) {
                constraints = buildCameraConstraints("facing:environment");
                stream = await attempt(constraints);
            }
            if (!stream) {
                constraints = { video: true, audio: false };
                stream = await attempt(constraints);
            }
            if (!stream) {
                throw new Error("Unable to open camera with current constraints.");
            }

            video.srcObject = stream;
            video.muted = true;
            video.playsInline = true;
            video.style.visibility = "visible";
            await video.play();

            cameraEmpty.style.display = "none";
            liveBadge.style.display   = "flex";
            cameraStatus.textContent  = cameraText("live");
            cameraStatus.classList.add("live");
            cameraStatus.classList.remove("err");

            btnOn.disabled       = true;
            btnOff.disabled      = false;
            btnCapture.disabled  = false;

            listCameras();

            startAutoDetect();
            setCameraDebugMessage(cameraText("waiting"), "info");
            toast(cameraText("started"), "ok");
            log("Camera started");
        } catch (e) {
            warn(e);
            stopCamera();
            const errName = e && e.name ? e.name + ': ' : '';
            const errMsg = e && e.message ? e.message : String(e);
            toast(`${cameraText("accessError")}${errName}${errMsg}`, "err");
            cameraStatus.textContent = cameraText("error");
            cameraStatus.classList.add("err");
        }
    }

    function stopCamera() {
        stopAutoDetect();
        if (stream) {
            stream.getTracks().forEach(t => t.stop());
            stream = null;
        }
        if (video) {
            video.pause();
            video.srcObject = null;
            video.removeAttribute("src");
            video.load();
            video.style.visibility = "hidden";
        }
        if (cameraEmpty) cameraEmpty.style.display = "";
        if (liveBadge) liveBadge.style.display = "none";
        if (cameraStatus) {
            cameraStatus.textContent = cameraText("off");
            cameraStatus.classList.remove("live", "err");
        }
        setCameraDebugMessage("");

        if (btnOn) btnOn.disabled = false;
        if (btnOff) btnOff.disabled = true;
        if (btnCapture) btnCapture.disabled = true;

        const ctx = overlay?.getContext?.("2d");
        if (ctx) ctx.clearRect(0, 0, overlay.width, overlay.height);

        log("Camera stopped");
    }

    function grabFrameBase64() {
        if (!video.videoWidth || !video.videoHeight) return null;

        const MAX_DIM = 640;
        const scale = Math.min(MAX_DIM / video.videoWidth, MAX_DIM / video.videoHeight, 1);
        const w = Math.max(320, Math.round(video.videoWidth * scale));
        const h = Math.max(320, Math.round(video.videoHeight * scale));

        const c = document.createElement("canvas");
        c.width = w;
        c.height = h;
        const ctx = c.getContext("2d");
        ctx.drawImage(video, 0, 0, w, h);
        lastCameraFrameSize = { w, h };

        // Use JPEG to preserve enough detail for recognition while keeping payload small
        return c.toDataURL("image/jpeg", 0.82);
    }

    function drawYoloBoxes(dets) {
        if (!overlay || !video || !video.videoWidth || !video.videoHeight) return;

        // The overlay canvas is scaled to match the rendered video size, so box
        // coordinates coming from the backend align correctly with the live feed.
        const ctx = overlay.getContext("2d");
        const dpr = window.devicePixelRatio || 1;
        const rect = video.getBoundingClientRect();
        const viewW = Math.max(1, rect.width || video.clientWidth || video.videoWidth);
        const viewH = Math.max(1, rect.height || video.clientHeight || video.videoHeight);
        overlay.style.width = `${viewW}px`;
        overlay.style.height = `${viewH}px`;
        overlay.width = Math.round(viewW * dpr);
        overlay.height = Math.round(viewH * dpr);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, viewW, viewH);
        if (!dets?.length) return;

        const frameW = lastCameraFrameSize?.w || video.videoWidth;
        const frameH = lastCameraFrameSize?.h || video.videoHeight;
        const scale = Math.min(viewW / frameW, viewH / frameH);
        const drawnW = frameW * scale;
        const drawnH = frameH * scale;
        const offX = (viewW - drawnW) / 2;
        const offY = (viewH - drawnH) / 2;

        ctx.lineWidth = Math.max(2, viewW / 320);
        ctx.font = `${Math.max(12, viewW / 60)}px Inter, sans-serif`;
        ctx.textBaseline = "top";

        dets.forEach(d => {
            const [x1, y1, x2, y2] = d.bbox || [0, 0, 0, 0];
            // Coordinates are in standardized frame space, scale to display
            const rx = offX + x1 * scale;
            const ry = offY + y1 * scale;
            const rw = Math.max(1, (x2 - x1) * scale);
            const rh = Math.max(1, (y2 - y1) * scale);
            ctx.strokeStyle = "#2bc26d";
            ctx.fillStyle   = "rgba(43,194,109,.85)";
            ctx.strokeRect(rx, ry, rw, rh);
            const label = `${seedLabel(d.class)} ${(d.conf * 100).toFixed(0)}%`;
            const tw = ctx.measureText(label).width + 10;
            const th = parseInt(ctx.font, 10) + 6;
            ctx.fillRect(rx, Math.max(0, ry - th), tw, th);
            ctx.fillStyle = "#fff";
            ctx.fillText(label, rx + 5, Math.max(0, ry - th) + 3);
        });
    }

    async function detectFrame(saveIt = false) {
        if (inFlight || !stream) return;

        // Capture the current video frame, send it to the backend, and render
        // the returned prediction result along with any YOLO bounding boxes.
        const b64 = grabFrameBase64();
        if (!b64) return;
        inFlight = true;
        try {
            const res = await jsonPost("/predict_frame", { image: b64, save: saveIt });
            renderResult(res, b64);
            drawYoloBoxes(res.yolo || []);
            renderCameraDebug(res);
        } catch (e) {
            warn("detectFrame error", e);
        } finally {
            inFlight = false;
        }
    }

    function startAutoDetect() {
        stopAutoDetect();
        if (!autoDetect?.checked) return;
        detectTimer = setInterval(detectFrame, dynInterval || 900);
    }

    function setCameraDebugMessage(message, type = "info") {
        if (!cameraDebug) return;
        cameraDebug.textContent = message || "";
        cameraDebug.dataset.type = type;
        cameraDebug.style.display = message ? "block" : "none";
    }

    function renderCameraDebug(res) {
        if (!cameraDebug) return;
        const debug = res?.debug_detection?.camera_debug;
        if (!debug) {
            setCameraDebugMessage("");
            return;
        }

        const lines = [];
        if (res.no_seed) {
            lines.push("No seed detected in current frame.");
        } else {
            const label = res.predicted_class ? seedLabel(res.predicted_class) : "Unknown";
            lines.push(`Prediction: ${label} (${Math.round((res.confidence || 0) * 100)}%)`);
            if (debug.used_source) {
                lines.push(`Source: ${debug.used_source}`);
            }
            if (typeof res.debug_detection?.raw_yolo_count !== "undefined") {
                lines.push(`YOLO: ${res.debug_detection.raw_yolo_count}, proposals: ${res.debug_detection.proposal_count}, boxes: ${res.yolo?.length || 0}`);
            }
            if (debug.frame_prediction?.class) {
                lines.push(`Frame model: ${seedLabel(debug.frame_prediction.class)} (${Math.round((debug.frame_prediction.confidence || 0) * 100)}%)`);
            }
        }

        setCameraDebugMessage(lines.join("\n"));
    }
    function stopAutoDetect() {
        if (detectTimer) clearInterval(detectTimer);
        detectTimer = null;
    }

    btnOn?.addEventListener("click", startCamera);
    btnOff?.addEventListener("click", stopCamera);
    btnOff?.addEventListener("pointerdown", (e) => {
        if (stream) {
            e.preventDefault();
            stopCamera();
        }
    });
    btnCapture?.addEventListener("click", () => detectFrame(true));
    autoDetect?.addEventListener("change", () => {
        if (autoDetect.checked && stream) startAutoDetect();
        else stopAutoDetect();
    });
    cameraSelect?.addEventListener("change", () => {
        if (stream) {
            stopCamera();
            setTimeout(startCamera, 250);
        }
    });

    $("#themeToggle")?.addEventListener("click", () => {
        const current = document.documentElement.dataset.theme;
        applyTheme(current === "dark" ? "light" : "dark");
        toast(`Theme switched to ${current === "dark" ? "Light" : "Dark"} mode`);
    });

    // populate camera list at boot (will re-list once permission granted)
    listCameras();


    /* ===================================================================
     * 5.  UPLOAD CONTROLLER
     * =================================================================== */
    const dropZone   = $("#dropZone");
    const fileInput  = $("#fileInput");
    const previewImg = $("#previewImg");
    const btnBrowse  = $("#btnBrowse");
    const btnPredict = $("#btnPredict");
    const btnClear   = $("#btnClear");
    const btnOnlineRefresh = $("#btnOnlineRefresh");

    let selectedFile = null;

    function setSelected(file) {
        if (!file) { clearSelected(); return; }
        const isImageType = file.type && file.type.startsWith("image/");
        const ext = file.name.split('.').pop().toLowerCase();
        const isImageExt = ["png", "jpg", "jpeg", "bmp", "webp"].includes(ext);
        if (!isImageType && !isImageExt) {
            toast("Only image files supported", "err");
            return;
        }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = e => {
            previewImg.src = e.target.result;
            dropZone.classList.add("has-img");
        };
        reader.readAsDataURL(file);
        btnPredict.disabled = false;
        btnClear.disabled   = false;
        toast(`Selected file: ${file.name}`, "ok");
    }

    function clearSelected() {
        selectedFile = null;
        previewImg.src = "";
        dropZone.classList.remove("has-img");
        if (fileInput) fileInput.value = "";
        btnPredict.disabled = true;
        btnClear.disabled   = true;
    }

    function handleFileInputEvent(e) {
        const files = e.target?.files || fileInput?.files || e.currentTarget?.files;
        const file = files?.[0];
        if (!file) {
            clearSelected();
            return;
        }
        console.log("handleFileInputEvent: selected", file && file.name, file && file.type);
        setSelected(file);
    }

    fileInput?.addEventListener("change", handleFileInputEvent);
    fileInput?.addEventListener("input", handleFileInputEvent);
    fileInput?.addEventListener("click", () => {
        if (fileInput) fileInput.value = "";
    });
    btnBrowse?.addEventListener("click", (e) => {
        e.preventDefault();
        fileInput?.click();
    });

    ["dragenter", "dragover"].forEach(ev =>
        dropZone?.addEventListener(ev, e => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("drag");
        })
    );
    ["dragleave", "drop"].forEach(ev =>
        dropZone?.addEventListener(ev, e => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("drag");
        })
    );
    dropZone?.addEventListener("drop", e => {
        if (e.dataTransfer?.files?.length) setSelected(e.dataTransfer.files[0]);
    });
    dropZone?.addEventListener("click", e => {
        if (e.target === dropZone || e.target.classList.contains("drop-inner")
            || e.target.tagName === "P" || e.target.classList.contains("big-icon")) {
            fileInput?.click();
        }
    });

    btnPredict?.addEventListener("click", async () => {
        if (!selectedFile && fileInput?.files?.length) {
            setSelected(fileInput.files[0]);
        }
        if (!selectedFile) {
            toast("Please select an image first.", "warn");
            return;
        }
        console.log("btnPredict: sending", selectedFile.name);
        showLoader("Running ensemble inference...");
        try {
            const fd = new FormData();
            fd.append("file", selectedFile, selectedFile.name || "upload.jpg");
            const res = await jsonPost("/predict", fd);
            renderResult(res, previewImg.src);
            toast("Detection complete", "ok");
            // scroll to result
            $("#resultPanel")?.scrollIntoView({ behavior: "smooth", block: "start" });
        } catch (e) {
            console.error("btnPredict error", e);
            toast("Detection failed: " + (e.message || e), "err");
        } finally {
            hideLoader();
        }
    });
    btnClear?.addEventListener("click", clearSelected);

    async function refreshOnlineDetection() {
        if (!navigator.onLine) {
            if (webSummary) webSummary.textContent = "Data not available. Please connect to the internet.";
            toast("Please connect to the internet.", "warn");
            return;
        }
        showLoader("Searching online seed sources...");
        try {
            let res;
            if (selectedFile) {
                const fd = new FormData();
                fd.append("file", selectedFile, selectedFile.name || "upload.jpg");
                fd.append("force_online", "1");
                res = await jsonPost("/predict", fd);
                renderResult(res, previewImg.src);
            } else if (lastImageSrc) {
                res = await jsonPost("/predict", { image: lastImageSrc, force_online: true });
                renderResult(res, lastImageSrc);
            } else {
                toast("Please upload the seed image again.", "warn");
                return;
            }
            toast(res.seed_data_available ? "High-confidence online seed details loaded" : "Low confidence result. Accurate seed data not available.", res.seed_data_available ? "ok" : "warn");
        } catch (e) {
            if (webSummary) webSummary.textContent = "Low confidence result. Accurate seed data not available.";
            toast("Online research failed: " + (e.message || e), "err");
        } finally {
            hideLoader();
        }
    }

    btnOnlineRefresh?.addEventListener("click", refreshOnlineDetection);


    /* ===================================================================
     * 6.  RESULT RENDERING
     * =================================================================== */
    const resultPanel    = $("#resultPanel");
    const resultImg      = $("#resultImg");
    const resultOverlay  = $("#resultOverlay");
    const resultMeta     = $("#resultMeta");
    const resultDebug    = $("#resultDebug");
    const resultClass    = $("#resultClass");
    const confFill       = $("#confidenceFill");
    const confText       = $("#confidenceText");
    const resultSource   = $("#resultSource");
    const top5List       = $("#top5List");
    const karnatakaBlock = $("#karnatakaBlock");
    const kPrice  = $("#kPrice");
    const kSoil   = $("#kSoil");
    const kSeason = $("#kSeason");
    const kMandis = $("#kMandis");
    const kNotes  = $("#kNotes");
    const webBlock   = $("#webBlock");
    const webSummary = $("#webSummary");
    const webLinks   = $("#webLinks");
    const closeResultBtn = $("#closeResult");

    closeResultBtn?.addEventListener("click", () => {
        resultPanel.style.display = "none";
    });

    function drawResultBoxes(dets) {
        if (!resultOverlay || !resultImg) return;
        const ctx = resultOverlay.getContext("2d");
        const dpr = window.devicePixelRatio || 1;
        const rect = resultImg.getBoundingClientRect();
        resultOverlay.width = Math.max(1, Math.round(rect.width * dpr));
        resultOverlay.height = Math.max(1, Math.round(rect.height * dpr));
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, rect.width, rect.height);
        if (!dets?.length || !resultImg.naturalWidth || !resultImg.naturalHeight) return;

        const scale = Math.min(rect.width / resultImg.naturalWidth, rect.height / resultImg.naturalHeight);
        const drawnW = resultImg.naturalWidth * scale;
        const drawnH = resultImg.naturalHeight * scale;
        const offX = (rect.width - drawnW) / 2;
        const offY = (rect.height - drawnH) / 2;

        ctx.lineWidth = 2;
        ctx.font = "13px Inter, sans-serif";
        ctx.textBaseline = "top";
        dets.forEach(d => {
            const [x1, y1, x2, y2] = d.bbox || [0, 0, 0, 0];
            const rx = offX + x1 * scale;
            const ry = offY + y1 * scale;
            const rw = Math.max(1, (x2 - x1) * scale);
            const rh = Math.max(1, (y2 - y1) * scale);
            const label = `${seedLabel(d.class)} ${(d.conf * 100).toFixed(0)}%`;
            ctx.strokeStyle = "#2bc26d";
            ctx.fillStyle = "rgba(43,194,109,.88)";
            ctx.strokeRect(rx, ry, rw, rh);
            const tw = ctx.measureText(label).width + 10;
            const th = 20;
            ctx.fillRect(rx, Math.max(0, ry - th), tw, th);
            ctx.fillStyle = "#fff";
            ctx.fillText(label, rx + 5, Math.max(0, ry - th) + 3);
        });
    }

    function renderResult(res, imageSrc) {
        if (!res || !res.ok) {
            toast("Detection failed", "err");
            return;
        }
        resultPanel.style.display = "block";

        // image
        if (imageSrc) {
            resultImg.src = imageSrc;
        } else if (res.image_url) {
            resultImg.src = res.image_url;
        }
        resultImg.onload = () => drawResultBoxes(res.yolo || []);
        if (resultImg.complete) drawResultBoxes(res.yolo || []);

        // meta
        resultMeta.textContent =
            `${res.timestamp || ""} - ${res.elapsed_ms || 0} ms - ${res.filename || "live frame"}`;

        // class + confidence
        const cls = res.predicted_class || "Unknown";
        const wi = res.web_info || {};
        const displayCls = res.online_pending
            ? (currentLang === "kn" ? "ಆಫ್‌ಲೈನ್ ಡೇಟಾಬೇಸ್‌ನಲ್ಲಿ ಇಲ್ಲ" : "Not in offline database")
            : res.online_researched && !res.seed_data_available
            ? (res.low_confidence_message || "Low confidence result. Accurate seed data not available.")
            : currentLang === "kn" && wi.seed_name_kn ? wi.seed_name_kn : seedLabel(cls);
        resultClass.textContent = displayCls;
        resultClass.title = humanise(cls);
        const c = Math.round((res.confidence || 0) * 100);
        confFill.style.width = c + "%";
        confText.textContent = c + "%";

        if (c >= 80) {
            confFill.style.background = "linear-gradient(90deg,#2bc26d,#0e6b3a)";
        } else if (c >= 55) {
            confFill.style.background = "linear-gradient(90deg,#f7d76b,#d4a017)";
        } else {
            confFill.style.background = "linear-gradient(90deg,#ff8a76,#c0392b)";
        }

        resultSource.textContent = res.source || "-";

        // top-5
        top5List.innerHTML = "";
        if (res.out_of_database) {
            const li = document.createElement("li");
            li.className = "muted small";
            li.textContent = res.online_pending
                ? (navigator.onLine ? "Data not available in offline database." : "Data not available. Please connect to the internet.")
                : (res.seed_data_available ? "High-confidence online result" : "Low confidence result. Accurate seed data not available.");
            top5List.appendChild(li);
        } else {
            (res.top5 || []).forEach(t => {
                const li = document.createElement("li");
                li.innerHTML = `
                    <span title="${humanise(t.class)}">${seedLabel(t.class)}</span>
                    <span class="pct">${(t.confidence * 100).toFixed(1)}%</span>`;
                top5List.appendChild(li);
            });
        }

        // karnataka info
        const kb = res.karnataka_info || {};
        if (!res.out_of_database && Object.keys(kb).length) {
            karnatakaBlock.style.display = "";
            // Show English or Kannada version based on current language
            const isKannada = currentLang === "kn";
            kPrice.textContent  = isKannada ? (kb.price_inr_per_kg_kn || kb.price_inr_per_kg || "-") : (kb.price_inr_per_kg || "-");
            kSoil.textContent   = isKannada ? (kb.soil_kn || kb.soil || "-") : (kb.soil || "-");
            kSeason.textContent = isKannada ? (kb.season_kn || kb.season || "-") : (kb.season || "-");
            kMandis.textContent = (kb.mandis || []).join(", ") || "-";
            kNotes.textContent  = isKannada ? (kb.notes_kn || kb.notes || "-") : (kb.notes || "-");
            renderDetectedMandiCard(cls, kb);
        } else {
            karnatakaBlock.style.display = "none";
            renderDetectedMandiCard("", null);
        }

        if (resultDebug) {
            const debug = res.debug_detection?.camera_debug || null;
            if (debug) {
                const lines = [];
                if (typeof res.debug_detection?.raw_yolo_count !== "undefined") {
                    lines.push(`YOLO boxes: ${res.debug_detection.raw_yolo_count}, proposals: ${res.debug_detection.proposal_count}`);
                }
                if (debug.frame_prediction?.class) {
                    lines.push(`Frame predicted: ${seedLabel(debug.frame_prediction.class)} (${(debug.frame_prediction.confidence * 100).toFixed(1)}%)`);
                }
                if (debug.used_source) {
                    lines.push(`Detection source: ${debug.used_source}`);
                }
                if (debug.no_seed) {
                    lines.push("No-seed gate applied");
                }
                if (debug.thresholds) {
                    lines.push(`Thresholds: YOLO=${debug.thresholds.yolo_conf_threshold}, crop=${debug.thresholds.camera_class_threshold}, fallback=${debug.thresholds.camera_fallback_threshold}`);
                }
                resultDebug.textContent = lines.join("\n");
                resultDebug.style.display = "block";
            } else {
                resultDebug.textContent = "";
                resultDebug.style.display = "none";
            }
        }

        // web fallback
        if (wi.summary || res.out_of_database) {
            webBlock.style.display = "";
            if (btnOnlineRefresh) {
                btnOnlineRefresh.style.display = res.online_pending ? "" : "none";
                btnOnlineRefresh.disabled = !res.online_pending;
            }
            const nameKn = wi.seed_name_kn || (currentLang === "kn" ? seedLabel(cls) : "");
            const onlineLabels = currentLang === "kn"
                ? {
                    seed: "ಬೀಜ",
                    kannada: "ಕನ್ನಡ ಹೆಸರು",
                    price: "ಪ್ರಸ್ತುತ ಮಾರುಕಟ್ಟೆ ಬೆಲೆ",
                    soil: "ಸೂಕ್ತ ಮಣ್ಣು",
                }
                : {
                    seed: "Seed",
                    kannada: "Kannada",
                    price: "Current market price",
                    soil: "Suitable soil",
                };
            webSummary.textContent = [
                res.online_pending
                    ? (navigator.onLine
                        ? (res.out_of_database_message || "Data not available in offline database.")
                        : (res.connect_message || "Data not available. Please connect to the internet."))
                    : "",
                res.online_researched && !res.seed_data_available
                    ? (res.low_confidence_message || "Low confidence result. Accurate seed data not available.")
                    : "",
                res.online_researched
                    ? `Confidence: ${((res.online_confidence || res.confidence || 0) * 100).toFixed(1)}%`
                    : "",
                wi.seed_name ? `${onlineLabels.seed}: ${wi.seed_name}` : "",
                nameKn ? `${onlineLabels.kannada}: ${nameKn}` : "",
                wi.market_price ? `${onlineLabels.price}: ${wi.market_price}` : "",
                wi.soil ? `${onlineLabels.soil}: ${wi.soil}` : "",
                wi.basic_details || wi.summary || "",
            ].filter(Boolean).join("\n\n");
            webLinks.innerHTML = "";
            (wi.links || []).forEach(l => {
                if (!l.url) return;
                const li = document.createElement("li");
                li.innerHTML = `<a href="${l.url}" target="_blank" rel="noopener">${l.title || l.url}</a>`;
                webLinks.appendChild(li);
            });
        } else {
            webBlock.style.display = "none";
            if (btnOnlineRefresh) btnOnlineRefresh.style.display = "none";
        }

        // soft pulse highlight on any matching mandi card
        highlightMandiCard(cls);
    }

    function highlightMandiCard(name) {
        const key = (name || "").toLowerCase();
        $$(".mandi-card").forEach(el => el.classList.remove("highlight"));
        const target = $(`.mandi-card[data-name="${key}"]`);
        if (target) {
            target.classList.add("highlight");
            setTimeout(() => target.classList.remove("highlight"), 4000);
        }
    }

    function renderDetectedMandiCard(name, info) {
        if (!mandiGrid) return;
        mandiGrid.innerHTML = "";

        const card = document.createElement("div");
        card.className = "mandi-card";

        if (!name || !info) {
            card.classList.add("mandi-empty");
            card.innerHTML = `
                <div class="mandi-card-head">
                    <h4>No seed detected yet</h4>
                    <span class="price-pill">Waiting</span>
                </div>
                <div class="mandi-card-body">
                    <p>Upload a seed photo or start the live camera to see the detected seed's Karnataka mandi price here.</p>
                </div>
            `;
            mandiGrid.appendChild(card);
            return;
        }

        card.dataset.name = name.toLowerCase();
        card.dataset.price = info.price_inr_per_kg || "";

        const head = document.createElement("div");
        head.className = "mandi-card-head";

        const title = document.createElement("h4");
        title.textContent = seedLabel(name);
        title.title = humanise(name);

        const isKannada = currentLang === "kn";
        const price = document.createElement("span");
        price.className = "price-pill";
        price.textContent = isKannada ? (info.price_inr_per_kg_kn || info.price_inr_per_kg || "-") : (info.price_inr_per_kg || "-");

        head.append(title, price);

        const body = document.createElement("div");
        body.className = "mandi-card-body";
        [
            ["Soil", isKannada ? (info.soil_kn || info.soil) : info.soil],
            ["Season", isKannada ? (info.season_kn || info.season) : info.season],
            ["Mandis", (info.mandis || []).join(", ")],
        ].forEach(([label, value]) => {
            const p = document.createElement("p");
            const strong = document.createElement("strong");
            strong.textContent = `${label}: `;
            p.append(strong, value || "-");
            body.appendChild(p);
        });

        if (info.notes) {
            const notes = document.createElement("p");
            notes.className = "muted small";
            notes.textContent = isKannada ? (info.notes_kn || info.notes) : info.notes;
            body.appendChild(notes);
        }

        card.append(head, body);
        mandiGrid.appendChild(card);
        requestAnimationFrame(() => card.classList.add("highlight"));
        setTimeout(() => card.classList.remove("highlight"), 4000);
    }


    /* ===================================================================
     * 7.  LOADER OVERLAY
     * =================================================================== */
    const loader     = $("#loader");
    const loaderText = $("#loaderText");
    function showLoader(msg = "Working...") {
        if (!loader) return;
        loaderText.textContent = msg;
        loader.style.display = "grid";
    }
    function hideLoader() {
        if (!loader) return;
        loader.style.display = "none";
    }


    /* ===================================================================
     * 8.  MANDI SEARCH / SORT  +  CHIP JUMP-TO
     * =================================================================== */
    const mandiSearch = $("#mandiSearch");
    const mandiSort   = $("#mandiSort");
    const mandiGrid   = $("#mandiGrid");

    function priceMidpoint(text) {
        if (!text) return 0;
        const nums = (text.match(/\d+(?:\.\d+)?/g) || []).map(Number);
        if (!nums.length) return 0;
        return nums.reduce((a, b) => a + b, 0) / nums.length;
    }

    function applyMandiFilter() {
        if (!mandiGrid) return;
        const q = (mandiSearch?.value || "").toLowerCase().trim();
        $$(".mandi-card", mandiGrid).forEach(c => {
            const name = (c.dataset.name || "").toLowerCase();
            const txt = c.textContent.toLowerCase();
            c.style.display = (q && !name.includes(q) && !txt.includes(q))
                ? "none" : "";
        });
    }

    function applyMandiSort() {
        if (!mandiGrid) return;
        const key = mandiSort?.value || "name";
        const cards = $$(".mandi-card", mandiGrid);
        cards.sort((a, b) => {
            if (key === "name") {
                return a.dataset.name.localeCompare(b.dataset.name);
            }
            return priceMidpoint(a.dataset.price) - priceMidpoint(b.dataset.price);
        });
        cards.forEach(c => mandiGrid.appendChild(c));
    }

    mandiSearch?.addEventListener("input", debounce(applyMandiFilter, 150));
    mandiSort?.addEventListener("change", applyMandiSort);

    // chip jump to mandi card
    $$(".chip[data-jump]").forEach(chip => {
        chip.addEventListener("click", e => {
            const key = chip.dataset.jump;
            const target = $(`.mandi-card[data-name="${key}"]`);
            if (target) {
                e.preventDefault();
                target.scrollIntoView({ behavior: "smooth", block: "center" });
                target.classList.add("highlight");
                setTimeout(() => target.classList.remove("highlight"), 2500);
            }
        });
    });


    /* ===================================================================
     * 9.  FOOTER HEALTH
     * =================================================================== */
    async function refreshHealth() {
        try {
            const h = await jsonGet("/api/health");
            const setIf = (id, val) => {
                const el = document.getElementById(id);
                if (el) el.textContent = val;
            };
            setIf("statModels",  `${h.cnn_models_loaded || 0} loaded`);
            setIf("statYolo",    h.yolo_loaded ? "loaded" : "not loaded");
            setIf("statClasses", h.classes || "-");
            setIf("statWeb",     h.web_search ? "online" : "offline");
        } catch (e) {
            warn("health fetch failed", e);
        }
    }
    refreshHealth();


    /* ===================================================================
     * 10. SMOOTH SCROLL + ACTIVE NAV LINK
     * =================================================================== */
    const navLinks = $$(".nav-link[href^='#']");
    function setActiveLink() {
        const y = window.scrollY + 120;
        let chosen = null;
        navLinks.forEach(a => {
            const id = a.getAttribute("href").slice(1);
            const sec = document.getElementById(id);
            if (sec && sec.offsetTop <= y) chosen = a;
        });
        navLinks.forEach(a => a.classList.remove("active"));
        chosen?.classList.add("active");
    }
    window.addEventListener("scroll", throttle(setActiveLink, 150));
    setActiveLink();


    /* ===================================================================
     * 11. BOOT
     * =================================================================== */
    log("Frontend ready");
    if ("serviceWorker" in navigator) {
        // PWA hook reserved - skipping for now.
    }

    // graceful shutdown of camera if user navigates away
    window.addEventListener("beforeunload", () => {
        try { stopCamera(); } catch (e) {}
    });

    // expose a tiny debug surface (handy during defence)
    window.__agro = {
        startCamera, stopCamera, detectFrame,
        renderResult, refreshHealth,
        applyTheme,
    };

    /* ===================================================================
     * 12. KEYBOARD SHORTCUTS
     *      C       toggle camera on/off
     *      U       open file picker (upload)
     *      Enter   run prediction on selected file
     *      Space   capture & save current frame
     *      A       toggle auto-detect
     *      H       go to history page
     *      Esc     close result panel / loader / modal
     *      ?       show shortcut help toast
     * =================================================================== */
    document.addEventListener("keydown", (e) => {
        // Ignore if user is typing inside an input / select / textarea
        const tag = (e.target?.tagName || "").toLowerCase();
        if (["input", "select", "textarea"].includes(tag)) return;
        if (e.target?.isContentEditable) return;
        switch (e.key.toLowerCase()) {
            case "t":
                $("#themeToggle")?.click();
                e.preventDefault();
                break;
            case "c":
                if (stream) stopCamera(); else startCamera();
                e.preventDefault();
                break;
            case "u":
                fileInput?.click();
                e.preventDefault();
                break;
            case "enter":
                if (selectedFile) btnPredict?.click();
                break;
            case " ":
                if (stream) {
                    detectFrame(true);
                    e.preventDefault();
                }
                break;
            case "a":
                if (autoDetect) {
                    autoDetect.checked = !autoDetect.checked;
                    autoDetect.dispatchEvent(new Event("change"));
                    toast(`Auto-detect ${autoDetect.checked ? "ON" : "OFF"}`);
                }
                break;
            case "h":
                location.href = "/history";
                break;
            case "escape":
                if (loader && loader.style.display !== "none") hideLoader();
                else if (resultPanel && resultPanel.style.display !== "none") {
                    resultPanel.style.display = "none";
                }
                break;
            case "?":
                toast(
                    "C=camera, U=upload, Enter=detect, Space=capture, A=auto, H=history",
                    "ok"
                );
                break;
        }
    });
    /* ===================================================================
     * 13. RESULT EXPORT (JSON / Image / Print)
     * -------------------------------------------------------------------
     *  Adds a tiny floating action bar inside the result panel that lets
     *  the user download the prediction as JSON, save the analysed image
     *  with the YOLO overlay, or print the result for a research log.
     * =================================================================== */
    let lastResult = null;
    let lastImageSrc = null;
    // Wrap renderResult to remember the last result.  We do this without
    // touching the original function so behaviour stays identical.
    const _origRender = renderResult;
    // eslint-disable-next-line no-func-assign
    renderResult = function (res, imageSrc) {
        lastResult = res;
        lastImageSrc = imageSrc;
        _origRender(res, imageSrc);
        ensureResultActionBar();
    };
    function ensureResultActionBar() {
        if (!resultPanel) return;
        if ($("#resultActions", resultPanel)) return;
        const bar = document.createElement("div");
        bar.id = "resultActions";
        bar.className = "result-actions";
        bar.innerHTML = `
            <button class="btn btn-ghost small" id="btnExportJson">JSON</button>
            <button class="btn btn-ghost small" id="btnExportImg">Image</button>
            <button class="btn btn-ghost small" id="btnCopyClass">Copy class</button>
            <button class="btn btn-ghost small" id="btnPrintResult">Print</button>
            <button class="btn btn-ghost small" id="btnShareResult">Share</button>
        `;
        $(".result-head", resultPanel)?.appendChild(bar);
        $("#btnExportJson").addEventListener("click", exportResultJSON);
        $("#btnExportImg").addEventListener("click", exportResultImage);
        $("#btnCopyClass").addEventListener("click", copyClassToClipboard);
        $("#btnPrintResult").addEventListener("click", () => window.print());
        $("#btnShareResult").addEventListener("click", shareResult);
    }
    function exportResultJSON() {
        if (!lastResult) { toast("Nothing to export", "warn"); return; }
        const blob = new Blob([JSON.stringify(lastResult, null, 2)],
                              { type: "application/json" });
        downloadBlob(blob, `agroseed_${Date.now()}.json`);
        toast("Saved JSON", "ok");
    }
    function exportResultImage() {
        const src = lastImageSrc || resultImg?.src;
        if (!src) { toast("No image to save", "warn"); return; }
        // download via temp anchor
        const a = document.createElement("a");
        a.href = src;
        a.download = `agroseed_${Date.now()}.jpg`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        toast("Saved image", "ok");
    }
    async function copyClassToClipboard() {
        if (!lastResult?.predicted_class) {
            toast("Nothing to copy", "warn"); return;
        }
        try {
            await navigator.clipboard.writeText(seedLabel(lastResult.predicted_class));
            toast("Copied to clipboard", "ok");
        } catch {
            toast("Clipboard unavailable", "err");
        }
    }
    async function shareResult() {
        if (!lastResult) { toast("Nothing to share", "warn"); return; }
        const text = `AgroSeed.AI prediction: ${seedLabel(lastResult.predicted_class)}` +
                     ` (${(lastResult.confidence * 100).toFixed(1)}%)`;
        if (navigator.share) {
            try {
                await navigator.share({
                    title: "AgroSeed.AI",
                    text,
                    url: location.href,
                });
            } catch (e) { /* user cancelled */ }
        } else {
            try {
                await navigator.clipboard.writeText(text + " - " + location.href);
                toast("Share text copied to clipboard", "ok");
            } catch {
                toast("Sharing unavailable", "err");
            }
        }
    }
    function downloadBlob(blob, name) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = name;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    }
    /* ===================================================================
     * 14. LIVE FRAME OVERLAY RESIZE
     *      Keep canvas overlay in sync with video element size when the
     *      window is resized (otherwise the YOLO boxes drift off).
     * =================================================================== */
    function resizeOverlay() {
        if (!video || !overlay) return;
        const rect = video.getBoundingClientRect();
        const viewW = Math.max(1, rect.width || video.clientWidth || video.videoWidth);
        const viewH = Math.max(1, rect.height || video.clientHeight || video.videoHeight);
        overlay.style.width = `${viewW}px`;
        overlay.style.height = `${viewH}px`;
        overlay.width = Math.round(viewW * (window.devicePixelRatio || 1));
        overlay.height = Math.round(viewH * (window.devicePixelRatio || 1));
        if (stream && lastResult?.yolo) {
            drawYoloBoxes(lastResult.yolo);
        }
    }
    window.addEventListener("resize", throttle(resizeOverlay, 200));
    window.addEventListener("resize", throttle(() => {
        if (lastResult?.yolo) drawResultBoxes(lastResult.yolo);
    }, 200));
    /* ===================================================================
     * 15. PASTE IMAGE FROM CLIPBOARD
     *      Lets the user press Ctrl/Cmd-V anywhere on the page and the
     *      pasted image is treated like a normal upload.
     * =================================================================== */
    document.addEventListener("paste", async (e) => {
        const items = e.clipboardData?.items;
        if (!items) return;
        for (const it of items) {
            if (it.type?.startsWith("image/")) {
                const file = it.getAsFile();
                if (file) {
                    setSelected(file);
                    toast("Image pasted from clipboard", "ok");
                    btnPredict?.scrollIntoView({ behavior: "smooth", block: "center" });
                }
                return;
            }
        }
    });
    /* ===================================================================
     * 16. PROGRESSIVE ENHANCEMENT - CONNECTION INDICATOR
     *      Shows a toast when the user goes offline / online (useful
     *      because the web-fallback search relies on connectivity).
     * =================================================================== */
    window.addEventListener("offline", () => toast("You are offline - web fallback disabled", "warn"));
    window.addEventListener("online",  () => toast("Back online - web fallback active", "ok"));
    /* ===================================================================
     * 17. SOFT REFRESH OF KARNATAKA HIGHLIGHT WHEN PAGE LOADS
     *      If a query-string ?seed=tomato is present we scroll to that
     *      mandi card and pulse-highlight it (handy for sharing links).
     * =================================================================== */
    const params = new URLSearchParams(location.search);
    const seedQ = (params.get("seed") || "").toLowerCase();
    if (seedQ) {
        const target = $(`.mandi-card[data-name="${seedQ}"]`);
        if (target) {
            setTimeout(() => {
                target.scrollIntoView({ behavior: "smooth", block: "center" });
                target.classList.add("highlight");
                setTimeout(() => target.classList.remove("highlight"), 4000);
            }, 600);
        }
    }
    /* ===================================================================
     * 18. PERFORMANCE LOGGING (page load to first interactive)
     *      Just a console table for the researcher's reference; never
     *      sent off-machine.
     * =================================================================== */
    window.addEventListener("load", () => {
        if (!performance?.timing) return;
        const t = performance.timing;
        const data = {
            "DNS lookup":   t.domainLookupEnd - t.domainLookupStart,
            "TCP connect":  t.connectEnd     - t.connectStart,
            "Request":      t.responseStart  - t.requestStart,
            "Response":     t.responseEnd    - t.responseStart,
            "DOM build":    t.domComplete    - t.domLoading,
            "Page total":   t.loadEventEnd   - t.navigationStart,
        };
        log("Page timing (ms):", data);
    });
    /* ===================================================================
     * 19. UTILITY: humanise filename / class label
     * =================================================================== */
    function humanise(s) {
        return (s || "")
            .toString()
            .replace(/_/g, " ")
            .replace(/\b\w/g, ch => ch.toUpperCase())
            .trim();
    }
    window.__agro.humanise = humanise;
    /* ===================================================================
     * 20. UTILITY: pretty bytes
     * =================================================================== */
    function prettyBytes(n) {
        if (!n) return "0 B";
        const units = ["B", "KB", "MB", "GB"];
        let i = 0;
        while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
        return `${n.toFixed(n >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
    }
    window.__agro.prettyBytes = prettyBytes;
    /* ===================================================================
     * 21. UTILITY: relative time
     * =================================================================== */
    function relativeTime(iso) {
        try {
            const t = new Date(iso).getTime();
            const diff = Date.now() - t;
            const s = Math.floor(diff / 1000);
            if (s < 60) return s + "s ago";
            const m = Math.floor(s / 60);
            if (m < 60) return m + "m ago";
            const h = Math.floor(m / 60);
            if (h < 24) return h + "h ago";
            const d = Math.floor(h / 24);
            return d + "d ago";
        } catch { return iso; }
    }
    window.__agro.relativeTime = relativeTime;


    /* ===================================================================
     * 25. PROTECT THE LIVE LOOP FROM SLOW BACKEND
     *      If the server takes >3 s to respond, we double the polling
     *      interval until it recovers - keeps the UI responsive.
     * =================================================================== */
    const _origDetect = detectFrame;
    // eslint-disable-next-line no-func-assign
    detectFrame = async function (saveIt = false) {
        const t0 = performance.now();
        await _origDetect.call(this, saveIt);
        const elapsed = performance.now() - t0;
        if (elapsed > 2500 && dynInterval < 4000) {
            dynInterval = Math.min(4000, dynInterval + 500);
            warn("Backend slow - polling slowed to", dynInterval, "ms");
            stopAutoDetect();
            if (autoDetect?.checked) {
                detectTimer = setInterval(detectFrame, dynInterval);
            }
        } else if (elapsed < 700 && dynInterval > 900) {
            dynInterval = Math.max(900, dynInterval - 250);
            stopAutoDetect();
            if (autoDetect?.checked) {
                detectTimer = setInterval(detectFrame, dynInterval);
            }
        }
    };


    /* ===================================================================
     * 27. LANGUAGE TOGGLE
     * -------------------------------------------------------------------
     *   English / Kannada labels for the main app controls.
     * =================================================================== */
    const I18N = {
        en: {
            // nav
            
            "a[href='/'][data-i18n='nav_home']": "Home",
            "a[href='#detect']":     "Detect",
            "a[href='#mandi']":      "Mandi Prices",
            "a[href='#classes']":    "Seed Catalog",
            "a[href='#about']":      "About",
            "a[href='/history']":    "History",
            // hero
            ".hero-title":           ["Realtime Agriculture Seeds<br/>",
                                      "<span class='grad-text'>Detection &amp; Recognition</span> using AI"].join(""),
            ".hero-sub":             null, // keep default English (long)
            // section titles
            "#detect .section-title":      "Detect a Seed",
            "#mandi  .section-title":      "Karnataka Mandi Prices",
            "#classes .section-title":     "Seed Catalog",
            "#about .section-title":       "How it works",
            // detect cards
            "#detect .card:nth-child(1) h3": "📸Live Camera",
            "#detect .card:nth-child(2) h3": "📤Upload Photo",
            // buttons
            "#btnCameraOn":  "Turn Camera On",
            "#btnCameraOff": "Stop Camera",
            "#btnCapture":   "Capture & Save",
            "#btnBrowse":    "Browse files",
            "#btnPredict":   "Detect Seed",
            "#btnClear":     "Clear",
            "#btnOnlineRefresh": "Refresh",
            // misc
            ".camera-hint":  "Tip: Hold seed 5-15 cm from the camera, plain background, good light. Auto-detection runs every 1.5 seconds.",
            "#cameraEmpty p": "Click Turn Camera On to begin live seed detection.",
            ".drop-inner p:first-of-type": "Drag & drop a seed photo here",
            ".drop-inner .muted:not(.small)": "or",
            ".drop-inner .muted.small": "PNG / JPG / JPEG / WEBP (max 16 MB)",
            ".result-head h3": "Result",
            "#closeResult": "Close",
            ".result-class .label": "Predicted Seed",
            ".top5-block h4": "Top-5 alternatives",
            "#karnatakaBlock h3": "Karnataka Information",
            "#mandi .section-sub": "Detect or upload a seed first. This panel will show only the Karnataka mandi price for the detected seed.",
            "#mandiEmpty h4": "No seed detected yet",
            "#mandiEmpty .price-pill": "Waiting",
            "#mandiEmpty .mandi-card-body p": "Upload a seed photo or start the live camera to see the detected seed's Karnataka mandi price here.",
            // history page
            "#histSearch[placeholder]": "Filter by seed name or filename...",
            "#mandiSearch[placeholder]": "Search by seed name (e.g. tomato, chilli, mango)...",
        },
        kn: {
            "a[href='/'][data-i18n='nav_home']": "ಮುಖಪುಟ",
            "a[href='#detect']":     "ಗುರುತಿಸಿ",
            "a[href='#mandi']":      "ಮಂಡಿ ಬೆಲೆ",
            "a[href='#classes']":    "ಬೀಜ ಪಟ್ಟಿ",
            "a[href='#about']":      "ವಿವರ",
            "a[href='/history']":    "ಇತಿಹಾಸ",
            ".hero-title":           ["ರಿಯಲ್ ಟೈಮ್ ಕೃಷಿ ಬೀಜ<br/>",
                                      "<span class='grad-text'>ಗುರುತಿಸುವಿಕೆ ಮತ್ತು ಪರಿಚಯ</span> AI ಮೂಲಕ"].join(""),
            ".hero-sub":             "{n} ಬೀಜ ಪ್ರಭೇದಗಳನ್ನು CNN ensemble ಮತ್ತು YOLOv8 ಬಳಸಿ ರಿಯಲ್ ಟೈಮ್‌ನಲ್ಲಿ ಗುರುತಿಸಿ. ಕರ್ನಾಟಕ ಮಂಡಿ ಬೆಲೆ, ಮಣ್ಣಿನ ಸೂಕ್ತತೆ ಮತ್ತು ಉತ್ತಮ ಬೆಳೆ ಕಾಲವನ್ನು ತಕ್ಷಣ ಪಡೆಯಿರಿ.",
            "#detect .section-title":      "ಬೀಜವನ್ನು ಗುರುತಿಸಿ",
            "#mandi  .section-title":      "ಕರ್ನಾಟಕ ಮಂಡಿ ಬೆಲೆಗಳು",
            "#classes .section-title":     "ಬೀಜ ಪಟ್ಟಿ",
            "#about .section-title":       "ಇದು ಹೇಗೆ ಕೆಲಸ ಮಾಡುತ್ತದೆ",
            "#detect .card:nth-child(1) h3": "📸ಲೈವ್ ಕ್ಯಾಮೆರಾ",
            "#detect .card:nth-child(2) h3": "📤ಫೋಟೋ ಅಪ್ಲೋಡ್",
            "#btnCameraOn":  "ಕ್ಯಾಮೆರಾ ಆನ್",
            "#btnCameraOff": "ಕ್ಯಾಮೆರಾ ನಿಲ್ಲಿಸಿ",
            "#btnCapture":   "ಸೆರೆಹಿಡಿದು ಉಳಿಸಿ",
            "#btnBrowse":    "ಫೈಲ್ ಆಯ್ಕೆ",
            "#btnPredict":   "ಬೀಜ ಗುರುತಿಸಿ",
            "#btnClear":     "ತೆರವುಗೊಳಿಸಿ",
            "#btnOnlineRefresh": "ರಿಫ್ರೆಶ್",
            ".camera-hint":  "ಸೂಚನೆ: ಬೀಜವನ್ನು ಕ್ಯಾಮೆರಾದಿಂದ 5-15 ಸೆಂ.ಮೀ. ದೂರದಲ್ಲಿ, ಸರಳ ಹಿನ್ನೆಲೆ ಮತ್ತು ಉತ್ತಮ ಬೆಳಕಿನಲ್ಲಿ ಹಿಡಿಯಿರಿ. ಆಟೋ ಗುರುತಿಸುವಿಕೆ ಪ್ರತಿ 1.5 ಸೆಕೆಂಡಿಗೆ ನಡೆಯುತ್ತದೆ.",
            "#cameraStatus": "ಕ್ಯಾಮೆರಾ ಆಫ್",
            "#cameraEmpty p": "ಲೈವ್ ಬೀಜ ಗುರುತಿಸುವಿಕೆಗೆ ಕ್ಯಾಮೆರಾ ಆನ್ ಕ್ಲಿಕ್ ಮಾಡಿ.",
            ".drop-inner p:first-of-type": "ಬೀಜದ ಫೋಟೋವನ್ನು ಇಲ್ಲಿ ಎಳೆದು ಬಿಡಿ",
            ".drop-inner .muted:not(.small)": "ಅಥವಾ",
            ".drop-inner .muted.small": "PNG / JPG / JPEG / WEBP (ಗರಿಷ್ಠ 16 MB)",
            ".result-head h3": "ಫಲಿತಾಂಶ",
            "#closeResult": "ಮುಚ್ಚಿ",
            ".result-class .label": "ಗುರುತಿಸಿದ ಬೀಜ",
            ".top5-block h4": "ಟಾಪ್-5 ಪರ್ಯಾಯಗಳು",
            "#karnatakaBlock h3": "ಕರ್ನಾಟಕ ಮಾಹಿತಿ",
            "#mandi .section-sub": "ಮೊದಲು ಬೀಜವನ್ನು ಗುರುತಿಸಿ ಅಥವಾ ಅಪ್ಲೋಡ್ ಮಾಡಿ. ಇಲ್ಲಿ ಗುರುತಿಸಿದ ಬೀಜದ ಕರ್ನಾಟಕ ಮಂಡಿ ಬೆಲೆ ಮಾತ್ರ ತೋರಿಸಲಾಗುತ್ತದೆ.",
            "#mandiEmpty h4": "ಇನ್ನೂ ಬೀಜ ಗುರುತಿಸಲಾಗಿಲ್ಲ",
            "#mandiEmpty .price-pill": "ಕಾಯುತ್ತಿದೆ",
            "#mandiEmpty .mandi-card-body p": "ಬೀಜದ ಫೋಟೋ ಅಪ್ಲೋಡ್ ಮಾಡಿ ಅಥವಾ ಲೈವ್ ಕ್ಯಾಮೆರಾ ಆರಂಭಿಸಿ. ಗುರುತಿಸಿದ ಬೀಜದ ಕರ್ನಾಟಕ ಮಂಡಿ ಬೆಲೆ ಇಲ್ಲಿ ಕಾಣಿಸುತ್ತದೆ.",
            "#histSearch[placeholder]": "ಬೀಜದ ಹೆಸರು ಅಥವಾ ಫೈಲ್ ಹೆಸರಿನಿಂದ ಹುಡುಕಿ...",
            "#mandiSearch[placeholder]": "ಬೀಜದ ಹೆಸರಿನಿಂದ ಹುಡುಕಿ...",
        },
    };

    const NO_SEED_MSG = {
        en: "No seed detected. Please bring a seed in front of the camera.",
        kn: "ಬೀಜ ಕಂಡುಬಂದಿಲ್ಲ. ದಯವಿಟ್ಟು ಬೀಜವನ್ನು ಕ್ಯಾಮೆರಾ ಮುಂದೆ ತೋರಿಸಿ.",
    };

    
    let currentLang = (() => {
        try { return localStorage.getItem("lang") || "en"; } catch { return "en"; }
    })();

    function applyLang(lang) {
        currentLang = (lang === "kn") ? "kn" : "en";
        try { localStorage.setItem("lang", currentLang); } catch {}
        document.documentElement.lang = currentLang;
        lang = currentLang;
        const dict = I18N[lang] || I18N.en;
        const nClasses = (document.querySelector("[data-count]")?.dataset.count) || "54";
        Object.entries(dict).forEach(([sel, val]) => {
            if (val == null) return;
            // Handle [placeholder] selectors -> set placeholder attr
            const placeholderMatch = sel.match(/^(.+?)\[placeholder\]$/);
            if (placeholderMatch) {
                document.querySelectorAll(placeholderMatch[1]).forEach(el => {
                    el.setAttribute("placeholder", val);
                });
                return;
            }
            document.querySelectorAll(sel).forEach(el => {
                let txt = val.replace("{n}", nClasses);
                if (val.includes("<")) el.innerHTML = txt;
                else el.textContent = txt;
            });
        });
        const langSelect = document.getElementById("langSelect");
        if (langSelect) langSelect.value = currentLang;
        if (cameraStatus) cameraStatus.textContent = stream ? cameraText("live") : cameraText("off");
        translateSeedLabels();
        if (lastResult?.karnataka_info && Object.keys(lastResult.karnataka_info).length) {
            renderDetectedMandiCard(lastResult.predicted_class, lastResult.karnataka_info);
        } else if (mandiGrid) {
            renderDetectedMandiCard("", null);
        }
    }

    document.getElementById("langSelect")?.addEventListener("change", (e) => {
        applyLang(e.target.value);
        toast(currentLang === "kn" ? "ಕನ್ನಡ ಭಾಷೆ ಆಯ್ಕೆಯಾಗಿದೆ" : "English selected", "ok");
    });

    // Apply on first load
    applyLang(currentLang);


    /* ===================================================================
     * 28. NO-SEED FRIENDLY MESSAGE
     * -------------------------------------------------------------------
     *   When the backend says "no_seed: true", we show a soft hint
     *   instead of a wrong prediction.  This is what stops the system
     *   from labelling persons / hands / walls as a random seed.
     * =================================================================== */
    const _origRender2 = renderResult;
    // eslint-disable-next-line no-func-assign
    renderResult = function (res, imageSrc) {
        if (res && res.no_seed) {
            // Show a gentle empty-state inside the result panel
            if (resultPanel) resultPanel.style.display = "block";
            if (resultImg && imageSrc) resultImg.src = imageSrc;
            if (resultMeta) resultMeta.textContent =
                `${res.timestamp || ""} - ${res.elapsed_ms || 0} ms`;
            if (resultClass) resultClass.textContent = currentLang === "kn" ? "ಫ್ರೇಮ್‌ನಲ್ಲಿ ಬೀಜ ಇಲ್ಲ" : "No seed in frame";
            if (confFill) { confFill.style.width = "0%"; confFill.style.background = "var(--text-mute)"; }
            if (confText) confText.textContent = "-";
            if (resultSource) resultSource.textContent = res.source || "no_seed";
            if (top5List) top5List.innerHTML = `<li class="muted small">${NO_SEED_MSG[currentLang] || NO_SEED_MSG.en}</li>`;
            if (karnatakaBlock) karnatakaBlock.style.display = "none";
            if (webBlock) webBlock.style.display = "none";
            if (resultOverlay) {
                const ctx = resultOverlay.getContext("2d");
                ctx?.clearRect(0, 0, resultOverlay.width, resultOverlay.height);
            }
            renderDetectedMandiCard("", null);

            // wipe any stale YOLO overlay
            try {
                const ctx = overlay?.getContext?.("2d");
                if (ctx && overlay) ctx.clearRect(0, 0, overlay.width, overlay.height);
            } catch {}
            // clear any highlighted mandi card
            document.querySelectorAll(".mandi-card.highlight")
                    .forEach(el => el.classList.remove("highlight"));

            lastResult = res;
            lastImageSrc = imageSrc;
            ensureResultActionBar();
            return;
        }
        _origRender2(res, imageSrc);
    };


    /* ===================================================================
     * 29. END-OF-FILE GUARD
     * =================================================================== */

})();

/* ============================ end of script.js ============================ */
