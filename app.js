/* ═══════════════════════════════════════════════════════════
   CHAT · ECCV 2026 — behaviour
   ═══════════════════════════════════════════════════════════ */
(function () {
  "use strict";

  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var hasGSAP = typeof window.gsap !== "undefined" && typeof window.ScrollTrigger !== "undefined";

  if (hasGSAP) gsap.registerPlugin(ScrollTrigger);

  /* ── 0. fallback: if GSAP is blocked, nothing stays invisible ────── */
  function showAll() {
    document.querySelectorAll("[data-anim]").forEach(function (el) {
      el.style.opacity = 1;
      el.style.transform = "none";
    });
  }
  if (!hasGSAP || reduced) showAll();

  /* ── 1. tabs: the segmented control ──────────────────────────────── */
  var tabs = [].slice.call(document.querySelectorAll(".segmented button"));
  var clips = [].slice.call(document.querySelectorAll(".player video"));
  var notes = [].slice.call(document.querySelectorAll(".clip-note p"));

  function play(v) {
    if (!v) return;
    var p = v.play();
    if (p && p.catch) p.catch(function () { /* autoplay may be blocked */ });
  }

  function select(key) {
    tabs.forEach(function (b) {
      var on = b.dataset.clip === key;
      b.setAttribute("aria-selected", String(on));
      b.tabIndex = on ? 0 : -1;
    });
    clips.forEach(function (v) {
      var on = v.dataset.clip === key;
      v.classList.toggle("on", on);
      v.muted = !soundOn; v.volume = level;
      if (on) { if (v.preload === "none") v.preload = "auto"; play(v); }
      else { v.pause(); v.currentTime = 0; }
    });
    notes.forEach(function (p) { p.classList.toggle("on", p.dataset.note === key); });
    drawTurns(key); turnsIdle = false;
  }

  function focusTab(i) {
    var n = tabs.length;
    var t = tabs[((i % n) + n) % n];
    t.focus();
    select(t.dataset.clip);
  }

  tabs.forEach(function (b, i) {
    b.addEventListener("click", function () { select(b.dataset.clip); });
    b.addEventListener("keydown", function (e) {
      if (e.key === "ArrowRight" || e.key === "ArrowDown") { e.preventDefault(); focusTab(i + 1); }
      else if (e.key === "ArrowLeft" || e.key === "ArrowUp") { e.preventDefault(); focusTab(i - 1); }
      else if (e.key === "Home") { e.preventDefault(); focusTab(0); }
      else if (e.key === "End") { e.preventDefault(); focusTab(tabs.length - 1); }
    });
  });

  // Play only what is on screen.
  var player = document.querySelector(".player");
  if (player && "IntersectionObserver" in window) {
    new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        var cur = clips.filter(function (v) { return v.classList.contains("on"); })[0];
        if (e.isIntersecting) play(cur);
        else clips.forEach(function (v) { v.pause(); });
      });
    }, { threshold: 0.25 }).observe(player);
  }

  /* ── 1b. sound and level ────────────────────────────────────────
     Clips start muted because browsers require it. The button unmutes;
     the slider sets the level, because these clips carry speech at very
     different loudnesses and one fixed volume suits nobody. Both are
     remembered per viewer.
     ─────────────────────────────────────────────────────────────── */
  var volume  = document.getElementById("volume");
  var soundBtn = document.getElementById("soundBtn");
  var volRange = document.getElementById("volRange");
  var soundOn = false;
  var level = 0.7;

  try {
    var savedLevel = localStorage.getItem("chat.volume");
    if (savedLevel !== null) {
      var n = parseFloat(savedLevel);
      if (n >= 0 && n <= 1) level = n;
    }
  } catch (err) {}

  function applySound() {
    clips.forEach(function (v) { v.muted = !soundOn; v.volume = level; });
    if (volume) volume.classList.toggle("on", soundOn);
    if (soundBtn) {
      soundBtn.setAttribute("aria-pressed", String(soundOn));
      soundBtn.setAttribute("aria-label", soundOn ? "Turn sound off" : "Turn sound on");
    }
    if (volRange) {
      volRange.value = String(Math.round(level * 100));
      volRange.style.setProperty("--fill", Math.round(level * 100) + "%");
    }
    try { localStorage.setItem("chat.volume", String(level)); } catch (err2) {}
  }
  applySound();

  if (soundBtn) {
    soundBtn.addEventListener("click", function () {
      soundOn = !soundOn;
      // Unmuting at zero would look broken, so give it an audible level.
      if (soundOn && level < 0.05) level = 0.7;
      applySound();
      var cur = clips.filter(function (v) { return v.classList.contains("on"); })[0];
      if (soundOn && cur) play(cur);
    });
  }
  if (volRange) {
    volRange.addEventListener("input", function () {
      level = Math.max(0, Math.min(1, parseInt(volRange.value, 10) / 100));
      // Dragging the slider up is itself a request for sound.
      if (level > 0 && !soundOn) {
        soundOn = true;
        var cur2 = clips.filter(function (v) { return v.classList.contains("on"); })[0];
        if (cur2) play(cur2);
      }
      if (level === 0) soundOn = false;
      applySound();
    });
  }

  /* ── 1c. turn-taking timeline ────────────────────────────────────
     The unit boundaries the generator itself produced, on one axis, moving
     with the clip. The point it makes is the paper's: a back-channel bar
     sits inside the other speaker's sentence bar, on a timeline both faces
     share. No timings are drawn by hand.
     ─────────────────────────────────────────────────────────────── */
  var TURNS = {"cafe":[{"s":"i","k":"sent","a":0.0,"b":3.38},{"s":"j","k":"word","a":1.44,"b":1.94},{"s":"j","k":"sent","a":3.57,"b":8.34},{"s":"i","k":"word","a":5.7,"b":6.2},{"s":"i","k":"sent","a":8.56,"b":9.96}],"hallway":[{"s":"i","k":"sent","a":0.0,"b":5.2},{"s":"j","k":"word","a":2.35,"b":2.85},{"s":"j","k":"sent","a":5.45,"b":9.96},{"s":"i","k":"word","a":7.86,"b":8.36}],"office":[{"s":"i","k":"sent","a":0.0,"b":7.09},{"s":"j","k":"word","a":3.3,"b":3.8},{"s":"j","k":"sent","a":7.31,"b":9.96}],"kitchen":[{"s":"i","k":"sent","a":0.0,"b":6.59},{"s":"j","k":"word","a":3.05,"b":3.55},{"s":"j","k":"sent","a":6.78,"b":9.96}]};
  var SPAN = 9.96;
  var turnsFig = document.getElementById("turns");
  var lanes = turnsFig ? { i: turnsFig.querySelector('[data-lane="i"]'),
                           j: turnsFig.querySelector('[data-lane="j"]') } : null;
  var bars = [];

  function drawTurns(key) {
    if (!turnsFig || !lanes) return;
    var units = TURNS[key];
    bars = [];
    lanes.i.innerHTML = ""; lanes.j.innerHTML = "";
    // No published unit list for a clip means no timeline, rather than a guess.
    if (!units) { turnsFig.classList.remove("ready"); return; }
    units.forEach(function (u) {
      var el = document.createElement("span");
      if (u.k === "word") el.className = "word";
      el.style.setProperty("--a", (u.a / SPAN).toFixed(4));
      el.style.setProperty("--w", Math.max((u.b - u.a) / SPAN, 0.006).toFixed(4));
      lanes[u.s].appendChild(el);
      bars.push({ el: el, a: u.a, b: u.b });
    });
    turnsFig.classList.add("ready");
  }

  var turnsIdle = false;
  function tickTurns() {
    if (turnsFig && bars.length) {
      var v = clips.filter(function (c) { return c.classList.contains("on"); })[0];
      if (v && !v.paused && v.duration) {
        turnsIdle = false;
        var t = v.currentTime;
        turnsFig.style.setProperty("--t", Math.min(t / SPAN, 1).toFixed(4));
        turnsFig.style.setProperty("--head", "0.55");
        bars.forEach(function (b) { b.el.classList.toggle("live", t >= b.a && t <= b.b); });
      } else if (!turnsIdle) {
        // Clear once when playback stops, then stop touching the DOM every frame.
        turnsIdle = true;
        turnsFig.style.setProperty("--head", "0");
        bars.forEach(function (b) { b.el.classList.remove("live"); });
      }
    }
    requestAnimationFrame(tickTurns);
  }
  if (turnsFig) { drawTurns(clips[0] && clips[0].dataset.clip); requestAnimationFrame(tickTurns); }

  /* ── 1d. the conversation ring ───────────────────────────────────
     A circular waveform around the wordmark. The left arc is one speaker,
     the right arc the other, and which arc is live follows the real turn
     schedule of the first demo clip, looping. It is silent, and it is the
     same structure the timeline under the player draws, so the hero states
     the paper's idea before a word of it is read.
     ─────────────────────────────────────────────────────────────── */
  (function () {
    var cv = document.getElementById("heroRing");
    if (!cv) return;
    var ctx = cv.getContext("2d");
    var N = 132, SPAN = 9.96;
    var schedule = (TURNS && TURNS.cafe) || [];
    var phase = [];
    for (var k = 0; k < N; k++) phase.push((k * 12.9898) % 1);

    var W = 0, Hgt = 0, dpr = 1;
    function size() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      var r = cv.getBoundingClientRect();
      W = r.width; Hgt = r.height;
      cv.width = Math.round(W * dpr); cv.height = Math.round(Hgt * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    size();
    window.addEventListener("resize", size);

    var gi = 0.16, gj = 0.16;
    function draw(now) {
      var t = (now / 1000) % SPAN;
      var wantI = 0.16, wantJ = 0.16;
      for (var u = 0; u < schedule.length; u++) {
        var s = schedule[u];
        if (t >= s.a && t <= s.b) {
          var lvl = s.k === "word" ? 0.62 : 1;
          if (s.s === "i") wantI = Math.max(wantI, lvl); else wantJ = Math.max(wantJ, lvl);
        }
      }
      gi += (wantI - gi) * 0.12; gj += (wantJ - gj) * 0.12;

      ctx.clearRect(0, 0, W, Hgt);
      var cx = W / 2, cy = Hgt / 2;
      var S = Math.min(W, Hgt);
      var R = S * 0.375;
      var spin = (now / 1000) * 0.026;
      for (var i = 0; i < N; i++) {
        var th = (i / N) * Math.PI * 2 + spin;
        // Crossfade the two speakers around the circle rather than cutting at
        // the poles, so the ring reads as one instrument with two ends.
        var w = (Math.sin(th) + 1) / 2;
        var g = gi * (1 - w) + gj * w;
        var p = phase[i] * Math.PI * 2;
        var env = 0.46 + 0.28 * Math.sin(now / 560 + p * 3.1)
                       + 0.26 * Math.sin(now / 230 + p * 7.7);
        var len = S * 0.010 + env * g * S * 0.052;
        var x0 = cx + Math.sin(th) * R, y0 = cy - Math.cos(th) * R;
        var x1 = cx + Math.sin(th) * (R + len), y1 = cy - Math.cos(th) * (R + len);
        var a = 0.05 + g * 0.15;
        var mix = Math.min(Math.max((g - 0.2) / 0.5, 0), 1);
        ctx.strokeStyle = "rgba(" + Math.round(134 - 134 * mix) + ","
                        + Math.round(134 - 21 * mix) + ","
                        + Math.round(139 + 88 * mix) + "," + a.toFixed(3) + ")";
        ctx.lineWidth = 1.6; ctx.lineCap = "round";
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      }
    }

    if (reduced) { draw(0); return; }
    var live = true, raf = 0;
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (es) {
        es.forEach(function (e) {
          live = e.isIntersecting;
          if (live && !raf) raf = requestAnimationFrame(loop);
        });
      }, { threshold: 0 }).observe(cv);
    }
    function loop(now) {
      draw(now);
      raf = live ? requestAnimationFrame(loop) : 0;
    }
    raf = requestAnimationFrame(loop);
  })();

  /* ── 2. copy BibTeX ──────────────────────────────────────────────── */
  var copy = document.getElementById("copybib");
  if (copy) {
    copy.addEventListener("click", function () {
      var text = document.querySelector(".bib code").innerText;
      var done = function () {
        copy.textContent = "Copied";
        copy.classList.add("done");
        setTimeout(function () { copy.textContent = "Copy"; copy.classList.remove("done"); }, 1800);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () {});
      } else {
        var ta = document.createElement("textarea");
        ta.value = text; document.body.appendChild(ta); ta.select();
        try { document.execCommand("copy"); done(); } catch (err) {}
        document.body.removeChild(ta);
      }
    });
  }

  if (!hasGSAP || reduced) return;

  /* ── 3. entrance + scroll reveals ────────────────────────────────
     Short, critically damped, no overshoot. Motion here exists to
     stage the reading order, not to be noticed.
     ─────────────────────────────────────────────────────────────── */
  var hero = [].slice.call(document.querySelectorAll(".hero [data-anim]"));
  gsap.to(hero, {
    opacity: 1, y: 0,
    duration: 0.9, ease: "power3.out",
    stagger: 0.08, delay: 0.05
  });

  var rest = [].slice.call(document.querySelectorAll("[data-anim]")).filter(function (el) {
    return !el.closest(".hero");
  });
  rest.forEach(function (el) {
    gsap.to(el, {
      opacity: 1, y: 0,
      duration: 0.85, ease: "power3.out",
      delay: (parseFloat(el.dataset.delay) || 0) * 0.09,
      scrollTrigger: { trigger: el, start: "top 88%", once: true }
    });
  });

  /* ── 4. teaser: the product reveal ───────────────────────────────
     Scales up into place as it enters. Apple's move: the object
     arrives rather than fades.
     ─────────────────────────────────────────────────────────────── */
  var teaser = document.getElementById("teaserFig");
  if (teaser) {
    gsap.fromTo(teaser,
      { scale: 0.9, opacity: 0, y: 40 },
      {
        scale: 1, opacity: 1, y: 0,
        ease: "none",
        scrollTrigger: {
          trigger: teaser,
          start: "top 92%",
          end: "top 42%",
          scrub: 0.6
        }
      }
    );
  }

  /* ── 5. statement: words rise in sequence ────────────────────────── */
  var lead = document.querySelector(".lead");
  if (lead) {
    // Wrap words without disturbing the two spans that carry meaning.
    [].slice.call(lead.childNodes).forEach(function (node) {
      if (node.nodeType === 3 && node.textContent.trim()) {
        var frag = document.createDocumentFragment();
        node.textContent.split(/(\s+)/).forEach(function (tok) {
          if (!tok.trim()) { frag.appendChild(document.createTextNode(tok)); return; }
          var s = document.createElement("span");
          s.className = "word"; s.textContent = tok;
          frag.appendChild(s);
        });
        lead.replaceChild(frag, node);
      } else if (node.nodeType === 1 && node.classList.contains("dim")) {
        var inner = document.createDocumentFragment();
        node.textContent.split(/(\s+)/).forEach(function (tok) {
          if (!tok.trim()) { inner.appendChild(document.createTextNode(tok)); return; }
          var s = document.createElement("span");
          s.className = "word"; s.textContent = tok;
          inner.appendChild(s);
        });
        node.textContent = "";
        node.appendChild(inner);
      }
    });
    lead.style.opacity = 1;
    gsap.from(lead.querySelectorAll(".word"), {
      opacity: 0, y: "0.42em",
      duration: 0.75, ease: "power3.out", stagger: 0.022,
      scrollTrigger: { trigger: lead, start: "top 78%", once: true }
    });
  }

  /* ── 6. pinned pipeline ──────────────────────────────────────────
     One stage highlighted at a time; the orb shifts hue and scale as
     text becomes audio becomes video.
     ─────────────────────────────────────────────────────────────── */
  var pinSection = document.querySelector(".pin-section");
  var stages = [].slice.call(document.querySelectorAll(".stages li"));
  var labels = [].slice.call(document.querySelectorAll(".pin-label"));
  var orb1 = document.querySelector(".orb.o1");

  function setStage(i) {
    stages.forEach(function (li, k) { li.classList.toggle("active", k === i); });
    labels.forEach(function (l, k) { l.classList.toggle("on", k === i); });
    if (!orb1) return;
    var looks = [
      { scale: 0.82, background: "radial-gradient(circle at 34% 30%, #7fc4ff, #0071e3 62%, #0055b3)" },
      { scale: 1.0,  background: "radial-gradient(circle at 34% 30%, #a8d8ff, #2b8cf0 60%, #0a63c9)" },
      { scale: 1.16, background: "radial-gradient(circle at 34% 30%, #cfe9ff, #55a6f5 58%, #1d6fd6)" }
    ][i];
    gsap.to(orb1, { scale: looks.scale, duration: 0.7, ease: "power2.out" });
    orb1.style.background = looks.background;
  }

  if (pinSection && stages.length && window.matchMedia("(min-width:901px)").matches) {
    setStage(0);
    ScrollTrigger.create({
      trigger: pinSection,
      start: "top top",
      end: "+=" + (stages.length * 60) + "%",
      pin: ".pin-inner",
      pinSpacing: true,
      onUpdate: function (self) {
        var i = Math.min(stages.length - 1, Math.floor(self.progress * stages.length));
        if (i !== setStage.last) { setStage(i); setStage.last = i; }
      }
    });
  } else {
    stages.forEach(function (li) { li.classList.add("active"); });
  }

  /* ── 7. stat counters ────────────────────────────────────────────── */
  document.querySelectorAll(".stat b[data-count]").forEach(function (el) {
    var target = parseFloat(el.dataset.count);
    var obj = { v: 0 };
    gsap.to(obj, {
      v: target,
      duration: 1.6, ease: "power2.out",
      onUpdate: function () { el.textContent = Math.round(obj.v).toLocaleString("en-US"); },
      scrollTrigger: { trigger: el, start: "top 86%", once: true }
    });
  });

  /* ── 8. mechanism panels ─────────────────────────────────────────
     Three diagrams, each scrubbed by scroll. The variable being
     scrubbed is the one the paper's own rule is written in.
     ─────────────────────────────────────────────────────────────── */
  var SVGNS = "http://www.w3.org/2000/svg";
  function el(tag, attrs, parent) {
    var n = document.createElementNS(SVGNS, tag);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }

  /* 8a · shared timeline -------------------------------------------------
     Turns alternate; the holder of the turn speaks while the other drops to
     short back-channels drawn from W_inter. */
  (function timeline() {
    var svg = document.querySelector("#mechTimeline svg");
    if (!svg) return;
    var grid = svg.querySelector(".tl-grid");
    var blocks = svg.querySelector(".tl-blocks");
    var head = svg.querySelector(".tl-head");
    var X0 = 70, X1 = 612, W = X1 - X0;
    var ROW = { i: 44, j: 138 }, H = 26;

    for (var g = 0; g <= 10; g++) {
      var gx = X0 + (W * g) / 10;
      el("line", { x1: gx, y1: 26, x2: gx, y2: 184 }, grid);
    }

    // [speaker, start, length] in normalised time, plus back-channels
    var TURNS = [
      ["i", 0.00, 0.20], ["j", 0.21, 0.15], ["i", 0.37, 0.12],
      ["j", 0.50, 0.22], ["i", 0.73, 0.14], ["j", 0.88, 0.12]
    ];
    var WORDS = ["uh-huh", "hmm", "okay", "yeah"];
    var items = [];

    TURNS.forEach(function (t, idx) {
      var who = t[0], other = who === "i" ? "j" : "i";
      var x = X0 + W * t[1], w = W * t[2];
      var r = el("rect", { x: x, y: ROW[who], width: w, height: H, class: "say" }, blocks);
      items.push({ node: r, a: t[1], b: t[1] + t[2] });

      // one back-channel from the listener, mid-turn
      if (idx < TURNS.length - 1) {
        var bw = 34, bx = x + w * 0.55;
        var b = el("rect", { x: bx, y: ROW[other] + 7, width: bw, height: 12, class: "bc" }, blocks);
        var lab = el("text", { x: bx + bw / 2, y: ROW[other] + 34 }, blocks);
        lab.textContent = WORDS[idx % WORDS.length];
        var ba = (bx - X0) / W;
        items.push({ node: b, a: ba, b: ba + bw / W });
      }
    });

    function render(p) {
      var hx = X0 + W * p;
      head.setAttribute("x1", hx); head.setAttribute("x2", hx);
      items.forEach(function (it) {
        it.node.classList.toggle("live", p >= it.a && p <= it.b);
      });
    }
    render(0);

    if (reduced) { render(0.42); return; }
    var st = { p: 0 };
    gsap.to(st, {
      p: 1, ease: "none",
      onUpdate: function () { render(st.p); },
      scrollTrigger: { trigger: "#mechTimeline", start: "top 78%", end: "bottom 46%", scrub: 0.5 }
    });
  })();

  /* 8b · coarse-to-fine ---------------------------------------------------
     l(t) = max(1, ceil(L*t/T)); the injected set is every scale from l(t)
     up to L. Higher l is lower spatial resolution, so the coarsest scale is
     alone at t = T and the finest joins last. */
  (function scales() {
    var svg = document.querySelector("#mechScale svg");
    if (!svg) return;
    var rows = svg.querySelector(".sc-rows");
    var head = svg.querySelector(".sc-head");
    var read = svg.querySelector(".sc-read");
    var X0 = 70, X1 = 612, W = X1 - X0;
    var L = 3, T = 50;                       // matches the released checkpoint
    var cells = [];

    for (var l = 1; l <= L; l++) {
      var y = 26 + (l - 1) * 58;
      var n = [16, 8, 4][l - 1];             // finer scale = more, smaller cells
      var cw = (W - (n - 1) * 4) / n;
      var lab = el("text", { x: X0 - 14, y: y + 22, class: "sc-rowlab" }, rows);
      lab.textContent = "l = " + l;
      var row = [];
      for (var c = 0; c < n; c++) {
        row.push(el("rect", {
          x: X0 + c * (cw + 4), y: y, width: cw, height: 34
        }, rows));
      }
      cells.push(row);
    }

    function render(p) {
      var t = Math.round(T * (1 - p));               // t runs T -> 0
      var lt = Math.max(1, Math.ceil((L * t) / T));  // the paper's schedule
      cells.forEach(function (row, idx) {
        var l = idx + 1;
        var on = l >= lt;                            // union from l(t) to L
        row.forEach(function (r) { r.classList.toggle("on", on); });
      });
      var hx = X0 + W * p;
      head.setAttribute("x1", hx); head.setAttribute("x2", hx);
      read.textContent = "t = " + t + "   l(t) = " + lt;
    }
    render(0);

    if (reduced) { render(0.55); return; }
    var st = { p: 0 };
    gsap.to(st, {
      p: 1, ease: "none",
      onUpdate: function () { render(st.p); },
      scrollTrigger: { trigger: "#mechScale", start: "top 78%", end: "bottom 46%", scrub: 0.5 }
    });
  })();

  /* 8c · boundary blend ---------------------------------------------------
     alpha(tau) = 0.5*exp(-(tau-1)^2 / 2 sigma^2), sigma = W/3, so the weight
     is exactly 0.5 at the boundary and decays outwards. */
  (function blend() {
    var svg = document.querySelector("#mechBlend svg");
    if (!svg) return;
    var bars = svg.querySelector(".bl-bars");
    var curve = svg.querySelector(".bl-curve");
    var readout = svg.querySelector(".bl-a");
    // Geometry is sized so both halves fit inside the 640-wide viewBox:
    // N * CW must not exceed MID.
    var MID = 320, TOP = 82, BH = 78, N = 22, CW = 14;
    var BASE = TOP - 8;          // the cross-fade curve sits above the bars

    var left = [], right = [];
    for (var k = 0; k < N; k++) {
      left.push(el("rect", { x: MID - (k + 1) * CW, y: TOP, width: CW - 2, height: BH }, bars));
      right.push(el("rect", { x: MID + k * CW + 2, y: TOP, width: CW - 2, height: BH }, bars));
    }

    // the two segments start as distinct greys; blending pulls them together
    var A = [214, 217, 223], B = [151, 157, 168];
    function mix(a, b, w) {
      return "rgb(" + a.map(function (v, i) { return Math.round(v + (b[i] - v) * w); }).join(",") + ")";
    }

    function render(p) {
      var Wwin = Math.max(1, Math.round(10 * p));   // window grows 0 -> 10 frames
      var sigma = Wwin / 3;
      var pts = [];
      for (var k = 0; k < N; k++) {
        var tau = k + 1;
        var a = tau <= Wwin ? 0.5 * Math.exp(-Math.pow(tau - 1, 2) / (2 * sigma * sigma)) : 0;
        left[k].setAttribute("fill", mix(A, B, a));
        right[k].setAttribute("fill", mix(B, A, a));
        if (tau <= Math.max(Wwin, 1)) {
          pts.push([MID - (k + 0.5) * CW, BASE - a * 52]);
          pts.push([MID + (k + 0.5) * CW, BASE - a * 52]);
        }
      }
      pts.sort(function (u, v) { return u[0] - v[0]; });
      if (pts.length > 2) {
        var d = "M " + pts[0][0].toFixed(1) + " " + BASE;
        pts.forEach(function (q) { d += " L " + q[0].toFixed(1) + " " + q[1].toFixed(1); });
        d += " L " + pts[pts.length - 1][0].toFixed(1) + " " + BASE + " Z";
        curve.setAttribute("d", d);
      } else curve.setAttribute("d", "");
      readout.textContent = p > 0.04 ? "W = " + Wwin + "   α(1) = 0.50" : "no blending";
    }
    render(0);

    if (reduced) { render(1); return; }
    var st = { p: 0 };
    gsap.to(st, {
      p: 1, ease: "none",
      onUpdate: function () { render(st.p); },
      scrollTrigger: { trigger: "#mechBlend", start: "top 76%", end: "bottom 50%", scrub: 0.5 }
    });
  })();

  /* ── 7b. walk the pipeline figure, one stage at a time ───────────
     The figure carries its own (a) (b) (c) labels, so the walk-through adds
     no new text: it veils the two panels you are not being shown, holds,
     moves on, then hands the whole figure back. Runs once, on arrival.
     ─────────────────────────────────────────────────────────────── */
  (function () {
    var dims = [].slice.call(document.querySelectorAll(".fig-dim"));
    if (dims.length !== 3 || reduced) return;
    var VEIL = 0.72, HOLD = 0.95;
    var tl = gsap.timeline({
      scrollTrigger: { trigger: ".fig-marks", start: "top 72%", once: true },
      defaults: { duration: 0.45, ease: "power2.inOut" }
    });
    [0, 1, 2].forEach(function (k) {
      var others = dims.filter(function (_, n) { return n !== k; });
      tl.to(others, { opacity: VEIL }, k === 0 ? 0.35 : ">" + HOLD)
        .to(others, { opacity: 0 }, ">" + HOLD);
    });
  })();

  /* ── 8. keep positions honest once media has loaded ──────────────── */
  window.addEventListener("load", function () { ScrollTrigger.refresh(); });
})();
