"use strict";
/* Boot: each section initialises independently so one failure cannot break the page. */
(function boot() {
  const run = (name, fn) => { try { return fn(); } catch (e) { console.error("init failed:", name, e); const s = document.getElementById(name); if (s) s.insertAdjacentHTML("afterbegin", `<div class="callout warn"><b>This part failed to load</b>${esc(String(e && e.message || e))}</div>`); } };
  bindK();
  run("start", setupShell);
  run("start", initStart); run("story", initStory); run("pipeline", initPipeline); run("theory", initTheory);
  run("lab", initLab); run("arena", initArena); run("results", initResults); run("choose", initChoose);
  run("lit", initLit); run("defend", initDefend); run("present", initPresent); run("files", initFiles); run("glossary", initGlossary);
  window.__studyGuideReady = true;
})();
