/* Rocket Agronomist - client side
 * 1. Live camera with an in-browser leaf check (same colour rules as leaf_core.py)
 * 2. Capture / upload -> preview -> send to Shiny ("photo_submit")
 * 3. Live "model steps" timeline fed by the server ("step" messages)
 * 4. English / Arabic switch
 */
(function () {
  "use strict";

  // ------------------------------------------------------------------ i18n
  const I18N = {
    en: {
      brand: "Rocket Agronomist", tagline: "Check rocket leaf nitrogen with your phone camera",
      s1: "Take a photo", s2: "Results & reasoning", s3: "Ask a follow-up", steps: "Model steps",
      cam_off_text: "Point your camera at one rocket leaf. The app checks live whether a leaf is in view.",
      cam_start: "Start camera", cam_unsupported: "Camera not available here. Use Upload instead.",
      switch: "Switch", flash: "Flash", upload: "Upload", retake: "Retake", analyze: "Analyze leaf",
      auto: "Auto-capture when the leaf is steady", analyzing: "Analyzing…",
      is_ref: "This is my healthy reference plant",
      disclaimer: "Research prototype. Estimates from leaf colour, not a lab test.",
      note_ph: "Optional note, e.g. older leaves look yellow",
      chat_ph: "Ask about this leaf, fertilizer or weather…",
      st_noleaf: "No leaf in view", st_dark: "Too dark – turn on the flash",
      st_small: "Leaf too small – move closer", st_close: "Too close – move back a little",
      st_steady: "Leaf in view – hold steady", st_ready: "Ready – tap the button",
      st_auto: "Hold still… capturing", st_still_ok: "Leaf found in photo", st_still_no: "No leaf found in this photo",
      st_denied: "Camera permission denied. Use Upload instead.",
      grp_measure: "Python", grp_agent: "AI agent",
      upload_another: "Upload another", save_ref: "Save as reference",
      ref_title: "Reference plant", ref_device: "Saved on this device",
      ref_none: "No reference yet. Photograph a healthy, well-fertilized plant and tap “Save as reference”. Later photos are compared with it.",
      ref_remove: "Remove reference", ref_line: "Greenness {d} · saved {t}",
      ref_confirm: "Remove the saved reference plant?", saving_ref: "Saving reference…",
      ref_fail: "No leaf found in this photo, so it was not saved as reference.",
    },
    ar: {
      brand: "مساعد الجرجير", tagline: "افحص نيتروجين أوراق الجرجير بكاميرا هاتفك",
      s1: "التقط صورة", s2: "النتائج والتفسير", s3: "اسأل سؤال متابعة", steps: "خطوات النموذج",
      cam_off_text: "وجّه الكاميرا نحو ورقة جرجير واحدة. يتحقق التطبيق مباشرة من وجود ورقة.",
      cam_start: "تشغيل الكاميرا", cam_unsupported: "الكاميرا غير متاحة هنا. استخدم الرفع بدلًا منها.",
      switch: "تبديل", flash: "فلاش", upload: "رفع", retake: "إعادة", analyze: "حلّل الورقة",
      auto: "التقاط تلقائي عند ثبات الورقة", analyzing: "جارٍ التحليل…",
      is_ref: "هذا نباتي المرجعي السليم",
      disclaimer: "نموذج بحثي. النتائج تقديرات من لون الورقة وليست تحليلًا مخبريًا.",
      note_ph: "ملاحظة اختيارية، مثل: الأوراق القديمة مصفرّة",
      chat_ph: "اسأل عن هذه الورقة أو السماد أو الطقس…",
      st_noleaf: "لا توجد ورقة في الإطار", st_dark: "الإضاءة ضعيفة – شغّل الفلاش",
      st_small: "الورقة صغيرة – اقترب", st_close: "قريب جدًا – ابتعد قليلًا",
      st_steady: "الورقة ظاهرة – ثبّت الهاتف", st_ready: "جاهز – اضغط الزر",
      st_auto: "اثبت… جارٍ الالتقاط", st_still_ok: "تم العثور على ورقة في الصورة", st_still_no: "لا توجد ورقة في هذه الصورة",
      st_denied: "تم رفض إذن الكاميرا. استخدم الرفع.",
      grp_measure: "بايثون", grp_agent: "الوكيل الذكي",
      upload_another: "رفع صورة أخرى", save_ref: "حفظ كمرجع",
      ref_title: "النبات المرجعي", ref_device: "محفوظ على هذا الجهاز",
      ref_none: "لا يوجد مرجع بعد. صوّر نباتًا سليمًا جيد التسميد واضغط «حفظ كمرجع». ستتم مقارنة الصور التالية به.",
      ref_remove: "حذف المرجع", ref_line: "الاخضرار {d} · حُفظ {t}",
      ref_confirm: "حذف النبات المرجعي المحفوظ؟", saving_ref: "جارٍ حفظ المرجع…",
      ref_fail: "لم يتم العثور على ورقة في هذه الصورة، لذلك لم تُحفظ كمرجع.",
    },
  };
  let LANG = "en";
  const tr = (k) => (I18N[LANG] && I18N[LANG][k]) || I18N.en[k] || k;

  function applyLang(l) {
    LANG = l;
    document.documentElement.lang = l;
    document.documentElement.dir = l === "ar" ? "rtl" : "ltr";
    document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = tr(el.dataset.i18n); });
    document.querySelectorAll(".lang-btn").forEach((b) => b.classList.toggle("is-on", b.dataset.lang === l));
    const note = document.getElementById("note"); if (note) note.placeholder = tr("note_ph");
    const ci = document.querySelector("shiny-chat-input");
    if (ci) { ci.setAttribute("placeholder", tr("chat_ph"));
      const ta = ci.querySelector("textarea"); if (ta) ta.placeholder = tr("chat_ph"); }
    document.querySelectorAll(".step-group").forEach((g) => { g.textContent = tr(g.dataset.key); });
    if (window.Shiny && Shiny.setInputValue) Shiny.setInputValue("lang", l);
    renderStatus(lastStatus);
    renderRef(loadRef());
  }

  // ------------------------------------------------------------------ camera
  const $ = (id) => document.getElementById(id);
  let stream = null, track = null, facing = "environment", torchOn = false, torchWanted = false;
  let loopTimer = null, prevGray = null, readySince = 0, lastStatus = null, busy = false;
  const sample = document.createElement("canvas");
  sample.width = 160; sample.height = 120;
  const sctx = sample.getContext("2d", { willReadFrequently: true });

  function setState(s) { $("cam").dataset.state = s; }

  // Same colour rules as leaf_core.detect_leaf (ExG + green hue gate)
  function analyzePixels(img) {
    const d = img.data, n = d.length / 4;
    let leaf = 0, vSum = 0;
    const gray = new Float32Array(n);
    for (let i = 0, p = 0; i < d.length; i += 4, p++) {
      const r = d[i] / 255, g = d[i + 1] / 255, b = d[i + 2] / 255;
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b), dl = mx - mn;
      vSum += mx; gray[p] = 0.299 * r + 0.587 * g + 0.114 * b;
      const sum = r + g + b + 1e-6, exg = (2 * g - r - b) / sum;
      if (exg <= 0.05 || mx < 0.118 || mx === 0 || dl / mx < 0.157) continue;
      let h;
      if (mx === r) h = 60 * (((g - b) / dl) % 6);
      else if (mx === g) h = 60 * ((b - r) / dl + 2);
      else h = 60 * ((r - g) / dl + 4);
      if (h < 0) h += 360;
      if (h >= 34 && h <= 170) leaf++;
    }
    let motion = 0;
    if (prevGray && prevGray.length === n) {
      for (let p = 0; p < n; p++) motion += Math.abs(gray[p] - prevGray[p]);
      motion /= n;
    }
    prevGray = gray;
    return { leafFrac: leaf / n, bright: vSum / n, motion };
  }

  function judge(m, live) {
    if (m.bright < 0.15) return { key: "st_dark", kind: "warn", ok: false };
    if (m.leafFrac < 0.05) return { key: live ? "st_noleaf" : "st_still_no", kind: "bad", ok: false };
    if (m.leafFrac < 0.10) return { key: "st_small", kind: "warn", ok: true };
    if (m.leafFrac > 0.92) return { key: "st_close", kind: "warn", ok: true };
    if (!live) return { key: "st_still_ok", kind: "ok", ok: true };
    if (m.motion > 0.035) return { key: "st_steady", kind: "warn", ok: true };
    return { key: "st_ready", kind: "ok", ok: true };
  }

  function renderStatus(s) {
    lastStatus = s;
    const el = $("cam-status"), fill = $("cam-meter-fill");
    if (!el || !s) return;
    el.textContent = tr(s.key);
    el.dataset.kind = s.kind;
    if (s.leafFrac !== undefined) {
      fill.style.width = Math.min(100, (s.leafFrac / 0.35) * 100).toFixed(0) + "%";
      fill.dataset.kind = s.kind;
    }
  }

  function loop() {
    const v = $("cam-video");
    if (!stream || v.readyState < 2 || $("cam").dataset.state !== "live") return;
    sctx.drawImage(v, 0, 0, sample.width, sample.height);
    const m = analyzePixels(sctx.getImageData(0, 0, sample.width, sample.height));
    const s = Object.assign(judge(m, true), { leafFrac: m.leafFrac });
    $("cam-capture").classList.toggle("is-ready", s.key === "st_ready");
    if ($("cam-auto").checked && s.key === "st_ready") {
      if (!readySince) readySince = performance.now();
      s.key = "st_auto";
      if (performance.now() - readySince > 1200) { readySince = 0; capture(); return; }
    } else readySince = 0;
    renderStatus(s);
  }

  async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      $("cam-unsupported").hidden = false; $("cam-start").hidden = true; return;
    }
    stopCamera();
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: facing }, width: { ideal: 1920 }, height: { ideal: 1440 } },
        audio: false,
      });
    } catch (e) {
      $("cam-still").hidden = true;
      setState("off");
      $("cam-unsupported").hidden = false;
      $("cam-unsupported").textContent = tr("st_denied");
      return;
    }
    track = stream.getVideoTracks()[0];
    const v = $("cam-video");
    v.srcObject = stream;
    await v.play().catch(() => {});
    const caps = track.getCapabilities ? track.getCapabilities() : {};
    $("cam-flash").hidden = !caps.torch;
    torchOn = false; $("cam-flash").classList.remove("is-on");
    if (torchWanted && caps.torch) await setTorch(true);
    $("cam-capture").disabled = false;
    setState("live");
    prevGray = null;
    loopTimer = setInterval(loop, 250);
  }

  function stopCamera() {
    if (loopTimer) clearInterval(loopTimer);
    loopTimer = null;
    // Some phones keep the torch lit while the track lives, so turn it off
    // explicitly first, then stop the track (stopping releases the torch too).
    if (track && torchOn) {
      try { track.applyConstraints({ advanced: [{ torch: false }] }); } catch (e) { /* ignore */ }
    }
    torchOn = false;
    $("cam-flash").classList.remove("is-on");
    if (stream) stream.getTracks().forEach((t) => t.stop());
    const v = $("cam-video"); if (v) v.srcObject = null;
    stream = null; track = null;
  }

  async function setTorch(on) {
    if (!track) return;
    try { await track.applyConstraints({ advanced: [{ torch: on }] }); torchOn = on; }
    catch (e) { torchOn = false; }
    $("cam-flash").classList.toggle("is-on", torchOn);
  }

  async function toggleTorch() {
    await setTorch(!torchOn);
    torchWanted = torchOn;          // remembered for the next photo
  }

  let stillData = null;

  function showStill(dataUrl) {
    stillData = dataUrl;
    const img = $("cam-still");
    img.onload = () => {
      // re-check the frozen photo with the same rules
      sctx.drawImage(img, 0, 0, sample.width, sample.height);
      prevGray = null;
      const m = analyzePixels(sctx.getImageData(0, 0, sample.width, sample.height));
      renderStatus(Object.assign(judge(m, false), { leafFrac: m.leafFrac }));
    };
    img.src = dataUrl;
    img.hidden = false;
    setState("still");
  }

  function capture() {
    const v = $("cam-video");
    if (!stream || v.videoWidth === 0) return;
    const scale = Math.min(1, 1600 / Math.max(v.videoWidth, v.videoHeight));
    const c = document.createElement("canvas");
    c.width = Math.round(v.videoWidth * scale); c.height = Math.round(v.videoHeight * scale);
    c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
    stopCamera();                   // releases the camera, so the flash goes off
    $("cam").classList.add("flash-anim");
    setTimeout(() => $("cam").classList.remove("flash-anim"), 300);
    showStill(c.toDataURL("image/jpeg", 0.92));
  }

  function fromFile(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        const scale = Math.min(1, 1600 / Math.max(img.width, img.height));
        const c = document.createElement("canvas");
        c.width = Math.round(img.width * scale); c.height = Math.round(img.height * scale);
        c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
        stopCamera();
        showStill(c.toDataURL("image/jpeg", 0.92));
      };
      img.onerror = () => {   // e.g. HEIC the browser cannot draw: send raw, server decodes
        stopCamera();
        stillData = reader.result;
        $("cam-still").hidden = true;
        setState("still");
        renderStatus({ key: "st_still_ok", kind: "ok" });
      };
      img.src = reader.result;
    };
    reader.readAsDataURL(file);
  }

  function retake() {
    $("cam-still").hidden = true;
    stillData = null;
    startCamera();
  }

  function submit(asReference) {
    if (!stillData || busy) return;
    busy = true;
    $("cam-busy").hidden = false;
    $("cam-busy").querySelector("span:last-child").textContent = tr(asReference ? "saving_ref" : "analyzing");
    ["cam-analyze", "cam-save-ref"].forEach((id) => { $(id).disabled = true; });
    Shiny.setInputValue("photo_submit",
      { data: stillData, as_reference: !!asReference, ts: Date.now() }, { priority: "event" });
    if (!asReference && window.matchMedia("(max-width: 900px)").matches) {
      setTimeout(() => document.querySelector(".results").scrollIntoView({ behavior: "smooth" }), 150);
    }
  }
  const analyze = () => submit(false);
  const saveAsReference = () => submit(true);

  // ------------------------------------------------------------------ reference (browser storage)
  const REF_KEY = "rocket_reference_v1";

  function loadRef() {
    try { return JSON.parse(localStorage.getItem(REF_KEY) || "null"); } catch (e) { return null; }
  }

  function storeRef(ref) {
    try {
      if (ref) localStorage.setItem(REF_KEY, JSON.stringify(ref));
      else localStorage.removeItem(REF_KEY);
    } catch (e) { /* storage full or blocked: keep it for this visit only */ }
  }

  function renderRef(ref) {
    const panel = $("ref-panel");
    if (!panel) return;
    panel.dataset.has = ref ? "1" : "0";
    if (!ref) return;
    $("ref-thumb").src = "data:image/jpeg;base64," + ref.thumb;
    const when = new Date(ref.saved_at).toLocaleString(LANG === "ar" ? "ar" : "en",
      { dateStyle: "medium", timeStyle: "short" });
    $("ref-line").textContent = tr("ref_line")
      .replace("{d}", Number(ref.features.dgci).toFixed(3)).replace("{t}", when);
  }

  function sendRefToServer(ref) {
    if (window.Shiny && Shiny.setInputValue) {
      Shiny.setInputValue("stored_reference", ref ? { features: ref.features } : null, { priority: "event" });
    }
  }

  function removeRef() {
    if (!window.confirm(tr("ref_confirm"))) return;
    storeRef(null); renderRef(null); sendRefToServer(null);
  }

  // ------------------------------------------------------------------ steps
  const STATUS_ICON = { running: "", done: "✓", warn: "!", error: "✕", skip: "–" };

  function upsertStep(ev) {
    const ol = $("steps");
    let li = ol.querySelector(`li[data-id="${CSS.escape(ev.id)}"]`);
    if (!li) {
      li = document.createElement("li");
      li.dataset.id = ev.id;
      li.innerHTML = '<span class="st-icon"></span><div class="st-body"><div class="st-top">' +
        '<span class="st-label"></span><span class="step-group"></span><span class="st-ms"></span></div>' +
        '<div class="st-detail"></div></div>';
      ol.appendChild(li);
      li.dataset.t0 = performance.now();
    }
    if (ev.label) li.querySelector(".st-label").textContent = ev.label;
    if (ev.group) {
      const g = li.querySelector(".step-group");
      g.dataset.key = "grp_" + ev.group; g.textContent = tr(g.dataset.key);
      li.dataset.group = ev.group;
    }
    if (ev.status) {
      li.dataset.status = ev.status;
      li.querySelector(".st-icon").textContent = STATUS_ICON[ev.status] || "";
      if (ev.status !== "running") {
        const ms = ev.ms || Math.round(performance.now() - Number(li.dataset.t0));
        li.querySelector(".st-ms").textContent = ms < 5 ? "" : ms >= 1000 ? (ms / 1000).toFixed(1) + " s" : ms + " ms";
      }
    }
    if (ev.detail !== undefined) li.querySelector(".st-detail").textContent = ev.detail;
  }

  // ------------------------------------------------------------------ wiring
  function wire() {
    $("cam-start").addEventListener("click", startCamera);
    $("cam-capture").addEventListener("click", capture);
    $("cam-switch").addEventListener("click", () => {
      facing = facing === "environment" ? "user" : "environment"; startCamera();
    });
    $("cam-flash").addEventListener("click", toggleTorch);
    $("cam-file").addEventListener("change", (e) => { fromFile(e.target.files[0]); e.target.value = ""; });
    $("cam-retake").addEventListener("click", retake);
    $("cam-analyze").addEventListener("click", analyze);
    $("cam-save-ref").addEventListener("click", saveAsReference);
    $("cam-file2").addEventListener("change", (e) => { fromFile(e.target.files[0]); e.target.value = ""; });
    $("ref-remove").addEventListener("click", removeRef);
    renderRef(loadRef());
    document.querySelectorAll(".lang-btn").forEach((b) =>
      b.addEventListener("click", () => applyLang(b.dataset.lang)));

    // tap photo to switch original <-> detected leaf
    document.addEventListener("click", (e) => {
      const img = e.target.closest(".shot-toggle"); if (!img) return;
      img.src = img.src === img.dataset.a ? img.dataset.b : img.dataset.a;
    });
    document.addEventListener("keydown", (e) => {
      if ((e.key === "Enter" || e.key === " ") && e.target.classList.contains("shot-toggle")) {
        e.preventDefault(); e.target.click();
      }
    });
  }

  let resumeOnShow = false;
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      resumeOnShow = $("cam").dataset.state === "live";
      stopCamera();
    } else if (resumeOnShow) {
      resumeOnShow = false;
      startCamera();
    }
  });
  window.addEventListener("pagehide", stopCamera);

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", wire);
  else wire();

  // Shiny fires its events through jQuery, so listen with jQuery
  // sessioninitialized fires after Shiny's init handshake; sending "event"
  // inputs earlier (on shiny:connected) makes the server close the session.
  window.jQuery(document).on("shiny:sessioninitialized", () => {
    applyLang(LANG);
    sendRefToServer(loadRef());     // survives reloads: the browser keeps it
  });

  if (window.Shiny) {
    Shiny.addCustomMessageHandler("steps_reset", (msg) => {
      $("steps").innerHTML = "";
      $("steps-title").textContent = msg.title || "";
    });
    Shiny.addCustomMessageHandler("step", upsertStep);
    Shiny.addCustomMessageHandler("analysis_done", (_msg) => {
      busy = false;
      $("cam-busy").hidden = true;
      ["cam-analyze", "cam-save-ref"].forEach((id) => { $(id).disabled = false; });
    });
    Shiny.addCustomMessageHandler("reference_saved", (msg) => {
      const ref = { features: msg.features, thumb: msg.thumb, saved_at: Date.now() };
      storeRef(ref); renderRef(ref);
      const panel = $("ref-panel");
      panel.classList.remove("just-saved"); void panel.offsetWidth; panel.classList.add("just-saved");
    });
    Shiny.addCustomMessageHandler("reference_failed", (_msg) => {
      renderStatus({ key: "ref_fail", kind: "bad" });
    });
  }
})();
