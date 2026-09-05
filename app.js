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
      if (on) { if (v.preload === "none") v.preload = "auto"; play(v); }
      else v.pause();
    });
    notes.forEach(function (p) { p.classList.toggle("on", p.dataset.note === key); });
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

  /* ── 8. keep positions honest once media has loaded ──────────────── */
  window.addEventListener("load", function () { ScrollTrigger.refresh(); });
})();
