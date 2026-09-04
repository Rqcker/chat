/* ══════════════════════════════════════════════════════════════
   CHAT · ECCV 2026 — project page behaviour
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";

  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ── 1. entrance + scroll reveals ─────────────────────────── */
  function revealAll(sel) {
    document.querySelectorAll(sel).forEach(function (el) { el.classList.add("in"); });
  }

  if (reduced) {
    revealAll(".rev, .reveal");
  } else {
    requestAnimationFrame(function () { revealAll(".rev"); });

    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
      });
    }, { rootMargin: "0px 0px -12% 0px", threshold: 0.08 });

    document.querySelectorAll(".reveal").forEach(function (el) { io.observe(el); });
  }

  /* ── 2. scroll progress rail ──────────────────────────────── */
  var bar = document.getElementById("scrollbar");
  var ticking = false;
  function onScroll() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(function () {
      var h = document.documentElement.scrollHeight - window.innerHeight;
      bar.style.width = (h > 0 ? (window.scrollY / h) * 100 : 0) + "%";
      ticking = false;
    });
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* ── 3. dock active section ───────────────────────────────── */
  var dockLinks = Array.prototype.slice.call(document.querySelectorAll(".dock a"));
  var targets = dockLinks
    .map(function (a) { return document.getElementById(a.dataset.dock); })
    .filter(Boolean);

  if (targets.length) {
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        dockLinks.forEach(function (a) {
          a.classList.toggle("active", a.dataset.dock === e.target.id);
        });
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    targets.forEach(function (t) { spy.observe(t); });
  }

  /* ── 4. demo tabs ─────────────────────────────────────────── */
  var tabs = Array.prototype.slice.call(document.querySelectorAll(".tabs button"));
  var panels = Array.prototype.slice.call(document.querySelectorAll(".players video"));
  var notes = Array.prototype.slice.call(document.querySelectorAll(".stage-note p"));

  function play(v) {
    if (!v) return;
    var p = v.play();
    if (p && typeof p.catch === "function") p.catch(function () { /* autoplay blocked */ });
  }

  function select(key) {
    tabs.forEach(function (b) {
      var on = b.dataset.tab === key;
      b.setAttribute("aria-selected", String(on));
      b.tabIndex = on ? 0 : -1;   // roving tabindex
    });
    panels.forEach(function (v) {
      var on = v.dataset.panel === key;
      v.classList.toggle("on", on);
      if (on) { if (v.preload === "none") v.preload = "auto"; play(v); }
      else { v.pause(); }
    });
    notes.forEach(function (p) { p.classList.toggle("on", p.dataset.note === key); });
  }

  function focusTab(i) {
    var n = tabs.length;
    var t = tabs[((i % n) + n) % n];
    t.focus();
    select(t.dataset.tab);
  }

  tabs.forEach(function (b, i) {
    b.addEventListener("click", function () { select(b.dataset.tab); });
    // Arrow-key navigation, as a tablist is expected to provide.
    b.addEventListener("keydown", function (e) {
      if (e.key === "ArrowRight" || e.key === "ArrowDown") { e.preventDefault(); focusTab(i + 1); }
      else if (e.key === "ArrowLeft" || e.key === "ArrowUp") { e.preventDefault(); focusTab(i - 1); }
      else if (e.key === "Home") { e.preventDefault(); focusTab(0); }
      else if (e.key === "End") { e.preventDefault(); focusTab(tabs.length - 1); }
    });
  });

  // Only run the visible clip, and only while the stage is on screen.
  var stage = document.querySelector(".players");
  if (stage) {
    var vio = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        var current = panels.filter(function (v) { return v.classList.contains("on"); })[0];
        if (e.isIntersecting) play(current);
        else panels.forEach(function (v) { v.pause(); });
      });
    }, { threshold: 0.25 });
    vio.observe(stage);
  }

  /* ── 5. copy BibTeX ───────────────────────────────────────── */
  var copyBtn = document.getElementById("copybib");
  if (copyBtn) {
    copyBtn.addEventListener("click", function () {
      var text = document.querySelector(".bibtex code").innerText;
      var done = function () {
        copyBtn.textContent = "copied";
        copyBtn.classList.add("done");
        setTimeout(function () {
          copyBtn.textContent = "copy";
          copyBtn.classList.remove("done");
        }, 1800);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () { });
      } else {
        var ta = document.createElement("textarea");
        ta.value = text; document.body.appendChild(ta); ta.select();
        try { document.execCommand("copy"); done(); } catch (err) { }
        document.body.removeChild(ta);
      }
    });
  }

  /* ── 6. hero: turn-taking waveform ────────────────────────────
     Two mirrored ribbons, one per speaker. Amplitude alternates on a
     conversational cadence: while one speaks the other listens, with a
     brief overlap at each hand-over. This is the paper's mechanism drawn
     as the background, not a generic particle field.
     ─────────────────────────────────────────────────────────── */
  var cv = document.getElementById("turnwave");
  if (!cv) return;
  var ctx = cv.getContext("2d");

  var COOL = [63, 169, 245];
  var WARM = [240, 164, 92];

  var W = 0, H = 0, dpr = 1;
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = cv.clientWidth; H = cv.clientHeight;
    cv.width = Math.round(W * dpr);
    cv.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener("resize", resize);

  // Turn schedule: [speaker, seconds]. 0 = speaker i, 1 = speaker j.
  var TURNS = [[0, 3.1], [1, 2.4], [0, 1.8], [1, 3.4], [0, 2.6], [1, 2.0]];
  var CYCLE = TURNS.reduce(function (s, t) { return s + t[1]; }, 0);

  // Smooth 0..1 activity for a given speaker at time t (seconds).
  function activity(speaker, t) {
    var u = ((t % CYCLE) + CYCLE) % CYCLE;
    var acc = 0, level = 0;
    for (var i = 0; i < TURNS.length; i++) {
      var who = TURNS[i][0], dur = TURNS[i][1];
      if (u >= acc && u < acc + dur) {
        var local = (u - acc) / dur;
        // ease in and out of each turn so hand-overs overlap softly
        var env = Math.min(1, local / 0.18) * Math.min(1, (1 - local) / 0.18);
        level = who === speaker ? env : 0;
        break;
      }
      acc += dur;
    }
    return 0.16 + 0.84 * level; // listeners keep a low idle motion
  }

  function ribbon(t, speaker, rgb, baseY, dir) {
    var amp = activity(speaker, t);
    var pts = 84;
    var span = W * 1.06;
    var x0 = -W * 0.03;

    ctx.beginPath();
    for (var i = 0; i <= pts; i++) {
      var p = i / pts;
      var x = x0 + span * p;
      // window so the ribbon fades at both edges
      var win = Math.sin(Math.PI * p);
      var phase = t * (speaker ? 1.35 : 1.62) + p * (speaker ? 7.4 : 9.1);
      var y =
        Math.sin(phase) * 44 +
        Math.sin(phase * 0.47 + 1.7) * 27 +
        Math.sin(phase * 2.11 + 0.6) * 11;
      ctx.lineTo(x, baseY + dir * y * win * amp);
    }

    var g = ctx.createLinearGradient(0, 0, W, 0);
    g.addColorStop(0, "rgba(" + rgb.join(",") + ",0)");
    g.addColorStop(0.28, "rgba(" + rgb.join(",") + "," + (0.52 * amp + 0.07).toFixed(3) + ")");
    g.addColorStop(0.72, "rgba(" + rgb.join(",") + "," + (0.52 * amp + 0.07).toFixed(3) + ")");
    g.addColorStop(1, "rgba(" + rgb.join(",") + ",0)");
    ctx.strokeStyle = g;
    ctx.lineWidth = 1.25;
    ctx.stroke();
  }

  function frame(now) {
    var t = now / 1000;
    ctx.clearRect(0, 0, W, H);

    var mid = H * 0.5;
    // stacked ribbons give the band a woven, textile feel
    for (var k = 0; k < 7; k++) {
      var off = (k - 3) * (H * 0.036);
      var fade = 1 - Math.abs(k - 3) / 4.6;
      ctx.globalAlpha = 0.46 * fade;
      ribbon(t + k * 0.42, 0, COOL, mid + off - H * 0.10, -1);
      ribbon(t + k * 0.42, 1, WARM, mid + off + H * 0.10, 1);
    }
    ctx.globalAlpha = 1;

    raf = requestAnimationFrame(frame);
  }

  var raf = null;
  if (reduced) {
    // one composed static frame
    var t0 = 1.2;
    ctx.clearRect(0, 0, W, H);
    var mid0 = H * 0.5;
    for (var k0 = 0; k0 < 7; k0++) {
      var off0 = (k0 - 3) * (H * 0.036);
      ctx.globalAlpha = 0.46 * (1 - Math.abs(k0 - 3) / 4.6);
      ribbon(t0 + k0 * 0.42, 0, COOL, mid0 + off0 - H * 0.10, -1);
      ribbon(t0 + k0 * 0.42, 1, WARM, mid0 + off0 + H * 0.10, 1);
    }
    ctx.globalAlpha = 1;
  } else {
    raf = requestAnimationFrame(frame);
    // stop painting once the hero has scrolled away
    var hio = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting && raf === null) raf = requestAnimationFrame(frame);
        else if (!e.isIntersecting && raf !== null) { cancelAnimationFrame(raf); raf = null; }
      });
    }, { threshold: 0.01 });
    hio.observe(cv);
  }
})();
