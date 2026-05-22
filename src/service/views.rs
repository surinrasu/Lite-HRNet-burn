use ann::tensor::backend::Backend;
use hypertext::{Raw, prelude::*};

use crate::SearchHit;

use super::RetrievalService;

pub(super) fn render_home(service: &RetrievalService<impl Backend>, error: Option<&str>) -> String {
    let sample_count = service.dataset.len();
    let candidate_count = service.index.entries.len();
    let top_k = service.default_top_k;
    let example_images = super::example_image_names();

    maud! {
        !DOCTYPE
        html lang="en" {
            head {
                meta charset="utf-8";
                meta name="viewport" content="width=device-width, initial-scale=1";
                link rel="stylesheet" href="/assets/material-symbols.css";
                link rel="stylesheet" href="/assets/beer.min.css";
                script type="module" src="/assets/beer.min.js" {}
                style { (Raw::dangerously_create(APP_CSS)) }
                title { "Oracle Pose Retrieval" }
            }
            body.dark {
                main.responsive.max {
                    header class="app-bar" {
                        h5 { "Oracle Pose Retrieval" }
                        div class="stats" {
                            span class="stat-pill" { (sample_count) " pairs" }
                            span class="stat-pill" { (candidate_count) " candidates" }
                            @if service.live {
                                span class="stat-pill live-pill" { "live" }
                            }
                        }
                        button class="theme-toggle circle transparent" type="button" data-theme-toggle aria-label="Toggle color theme" title="Toggle color theme" {
                            i data-theme-icon { "dark_mode" }
                        }
                    }

                    @if let Some(error) = error {
                        article.border.error {
                            i { "error" }
                            span { (error) }
                        }
                    }

                    section class="search-panel" {
                        article class="query-card" {
                            h6 { "Query by upload" }
                            form class="query-form" method="post" action="/search" enctype="multipart/form-data" {
                                label class="file-picker" for="query-image" {
                                    i { "upload_file" }
                                    span { "Choose pose image" }
                                }
                                input id="query-image" class="file-input" type="file" name="image" accept="image/png,image/jpeg,image/webp";
                                div class="query-controls" {
                                    div class="topk-control" {
                                        div.field.border.round {
                                            input type="number" name="k" min="1" max="50" value=(top_k);
                                            label { "top k" }
                                        }
                                    }
                                    div.actions {
                                        button.round type="submit" {
                                            i { "search" }
                                            span { "Search" }
                                        }
                                    }
                                }
                            }
                        }
                        article class="query-card" {
                            h6 { "Query by data sample" }
                            form class="query-form" method="get" action="/search" {
                                div class="query-controls sample-controls" {
                                    div class="sample-control" {
                                        div.field.border.round {
                                            input type="number" name="sample" min="0" max=(sample_count.saturating_sub(1)) value="0";
                                            label { "sample id" }
                                        }
                                    }
                                    div class="topk-control" {
                                        div.field.border.round {
                                            input type="number" name="k" min="1" max="50" value=(top_k);
                                            label { "top k" }
                                        }
                                    }
                                    div.actions {
                                        button.round type="submit" {
                                            i { "play_arrow" }
                                            span { "Run" }
                                        }
                                    }
                                }
                            }
                        }
                        @if service.live {
                            article class="query-card live-card" {
                                h6 { "Live video" }
                                div class="live-stage" {
                                    video id="live-video" "autoplay"="autoplay" "playsinline"="playsinline" "muted"="muted" {}
                                    canvas id="live-canvas" "hidden"="hidden" {}
                                    div id="live-placeholder" class="live-placeholder" {
                                        i { "videocam" }
                                    }
                                }
                                div class="query-controls live-controls" {
                                    div class="topk-control" {
                                        div.field.border.round {
                                            input id="live-top-k" type="number" min="1" max="50" value=(top_k);
                                            label { "top k" }
                                        }
                                    }
                                    div class="actions live-actions" {
                                        button id="live-start" class="round" type="button" {
                                            i { "play_arrow" }
                                            span { "Start" }
                                        }
                                        button id="live-stop" class="round border" type="button" "disabled"="disabled" {
                                            i { "stop" }
                                            span { "Stop" }
                                        }
                                    }
                                }
                                div id="live-status" class="live-status" { "idle" }
                                div id="live-results" class="live-results" {}
                            }
                        }
                    }

                    @if !example_images.is_empty() {
                        div class="example-gallery" aria-label="Example images" {
                            @for name in example_images.iter() {
                                a class="example-card" href=(format!("{}{}", super::EXAMPLE_ASSET_PREFIX, name)) aria-label=(format!("Example {name}")) {
                                    img src=(format!("{}{}", super::EXAMPLE_ASSET_PREFIX, name)) alt=(format!("Example {name}")) "loading"="lazy";
                                }
                            }
                        }
                    }
                }
                @if service.live {
                    script { (Raw::dangerously_create(LIVE_JS)) }
                }
                script { (Raw::dangerously_create(THEME_JS)) }
            }
        }
    }
    .render()
    .as_inner()
    .to_string()
}

pub(super) fn render_results(
    service: &RetrievalService<impl Backend>,
    hits: &[SearchHit],
    source: &str,
    top_k: usize,
) -> String {
    let sample_count = service.dataset.len();
    let candidate_count = service.index.entries.len();

    maud! {
        !DOCTYPE
        html lang="en" {
            head {
                meta charset="utf-8";
                meta name="viewport" content="width=device-width, initial-scale=1";
                link rel="stylesheet" href="/assets/material-symbols.css";
                link rel="stylesheet" href="/assets/beer.min.css";
                script type="module" src="/assets/beer.min.js" {}
                style { (Raw::dangerously_create(APP_CSS)) }
                title { "Oracle Pose Retrieval" }
            }
            body.dark {
                main.responsive.max {
                    header class="app-bar" {
                        a class="back-link" href="/" aria-label="Back" {
                            i { "arrow_back" }
                        }
                        h5 { "Results" }
                        div class="stats" {
                            span class="stat-pill" { (sample_count) " pairs" }
                            span class="stat-pill" { (candidate_count) " candidates" }
                        }
                        button class="theme-toggle circle transparent" type="button" data-theme-toggle aria-label="Toggle color theme" title="Toggle color theme" {
                            i data-theme-icon { "dark_mode" }
                        }
                    }
                    article class="result-summary" {
                        h6 { (source) }
                        p { "Showing top " (top_k) " candidates by shared embedding similarity." }
                    }
                    div class="result-grid" {
                        @for (rank, hit) in hits.iter().enumerate() {
                            article class="result-card" {
                                div.rank { "#" (rank + 1) }
                                img src=(format!("/candidate/{}", hit.index)) alt=(hit.entry.id.clone()) "loading"="lazy";
                                h6 { (hit.entry.character.clone().unwrap_or_else(|| hit.entry.id.clone())) }
                                p {
                                    (hit.entry.id)
                                    @if let Some(codepoint) = &hit.entry.codepoint {
                                        " " (codepoint)
                                    }
                                }
                                progress max="1" value=(format!("{:.4}", hit.score.max(0.0))) {}
                                small { "score " (format!("{:.4}", hit.score)) }
                            }
                        }
                    }
                }
                script { (Raw::dangerously_create(THEME_JS)) }
            }
        }
    }
    .render()
    .as_inner()
    .to_string()
}

const THEME_JS: &str = r##"
(() => {
  const storageKey = "pose-obc-theme";
  const buttons = Array.from(document.querySelectorAll("[data-theme-toggle]"));
  const media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;

  function storedTheme() {
    try {
      const value = window.localStorage.getItem(storageKey);
      return value === "light" || value === "dark" ? value : null;
    } catch (error) {
      return null;
    }
  }

  function preferredTheme() {
    return storedTheme() || (media && media.matches ? "dark" : "light");
  }

  function applyTheme(theme) {
    document.body.classList.toggle("dark", theme === "dark");
    document.body.classList.toggle("light", theme === "light");
    for (const button of buttons) {
      const icon = button.querySelector("[data-theme-icon]");
      const label = theme === "dark" ? "Switch to light mode" : "Switch to dark mode";
      if (icon) icon.textContent = theme === "dark" ? "light_mode" : "dark_mode";
      button.setAttribute("aria-label", label);
      button.setAttribute("title", label);
    }
  }

  function persistTheme(theme) {
    try {
      window.localStorage.setItem(storageKey, theme);
    } catch (error) {}
  }

  applyTheme(preferredTheme());
  for (const button of buttons) {
    button.addEventListener("click", () => {
      const next = document.body.classList.contains("dark") ? "light" : "dark";
      persistTheme(next);
      applyTheme(next);
    });
  }
  if (media) {
    media.addEventListener("change", () => {
      if (!storedTheme()) applyTheme(preferredTheme());
    });
  }
})();
"##;

const LIVE_JS: &str = r##"
(() => {
  const video = document.querySelector("#live-video");
  const canvas = document.querySelector("#live-canvas");
  const placeholder = document.querySelector("#live-placeholder");
  const startButton = document.querySelector("#live-start");
  const stopButton = document.querySelector("#live-stop");
  const topKInput = document.querySelector("#live-top-k");
  const status = document.querySelector("#live-status");
  const results = document.querySelector("#live-results");
  if (!video || !canvas || !startButton || !stopButton || !topKInput || !status || !results) return;

  const frameIntervalMs = 700;
  const maxFrameSide = 640;
  let stream = null;
  let timer = null;
  let running = false;
  let inFlight = false;

  function setStatus(value) {
    status.textContent = value;
  }

  function setRunning(value) {
    running = value;
    startButton.disabled = value;
    stopButton.disabled = !value;
    if (placeholder) placeholder.hidden = value;
  }

  function stopLive() {
    if (timer) {
      window.clearInterval(timer);
      timer = null;
    }
    if (stream) {
      for (const track of stream.getTracks()) track.stop();
      stream = null;
    }
    video.srcObject = null;
    setRunning(false);
    setStatus("idle");
  }

  async function startLive() {
    if (running) return;
    startButton.disabled = true;
    setStatus("starting");
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } },
        audio: false
      });
      video.srcObject = stream;
      await video.play();
      setRunning(true);
      setStatus("running");
      timer = window.setInterval(searchFrame, frameIntervalMs);
      searchFrame();
    } catch (error) {
      stopLive();
      setStatus("camera unavailable");
    }
  }

  async function searchFrame() {
    if (!running || inFlight || !video.videoWidth || !video.videoHeight) return;
    inFlight = true;
    setStatus("scoring");
    try {
      const scale = Math.min(1, maxFrameSide / Math.max(video.videoWidth, video.videoHeight));
      canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
      canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
      const context = canvas.getContext("2d", { willReadFrequently: false });
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.82));
      if (!blob) throw new Error("frame encode failed");

      const params = new URLSearchParams({ k: topKInput.value || "8" });
      const response = await fetch(`/live/search?${params.toString()}`, {
        method: "POST",
        headers: { "Content-Type": "image/jpeg" },
        body: blob
      });
      if (!response.ok) throw new Error(await response.text());
      renderHits((await response.json()).hits || []);
      if (running) setStatus("running");
    } catch (error) {
      if (running) setStatus(liveErrorStatus(error));
      console.warn("Live frame search failed", error);
    } finally {
      inFlight = false;
    }
  }

  function liveErrorStatus(error) {
    const message = error && error.message ? String(error.message) : "unknown error";
    if (message.includes("SpinePose did not detect a person")) return "no pose detected";
    const trimmed = message.length > 220 ? `${message.slice(0, 217)}...` : message;
    return `frame error: ${trimmed}`;
  }

  function renderHits(hits) {
    const items = hits.map((hit) => {
      const item = document.createElement("div");
      item.className = "live-result";

      const image = document.createElement("img");
      image.src = hit.image_url;
      image.alt = hit.id;

      const body = document.createElement("div");
      body.className = "live-result-body";

      const title = document.createElement("strong");
      title.textContent = `#${hit.rank} ${hit.character || hit.id}`;

      const meta = document.createElement("span");
      meta.textContent = `${hit.id}${hit.codepoint ? " " + hit.codepoint : ""}`;

      const score = document.createElement("small");
      const numericScore = Number(hit.score) || 0;
      score.textContent = `score ${numericScore.toFixed(4)}`;

      const progress = document.createElement("progress");
      progress.max = 1;
      progress.value = Math.max(0, Math.min(1, numericScore));

      body.append(title, meta, progress, score);
      item.append(image, body);
      return item;
    });
    results.replaceChildren(...items);
  }

  startButton.addEventListener("click", startLive);
  stopButton.addEventListener("click", stopLive);
  window.addEventListener("beforeunload", stopLive);
})();
"##;

const APP_CSS: &str = r#"
html, body { max-width: 100%; overflow-x: hidden; }
body.dark { color-scheme: dark; }
body.light { color-scheme: light; }
main.max { width: min(100%, 1180px); max-width: 1180px; padding: 1rem; overflow-x: hidden; }
i { font-family: "Material Symbols Outlined"; font-weight: normal; font-style: normal; line-height: 1; letter-spacing: normal; text-transform: none; display: inline-block; white-space: nowrap; }
.app-bar { min-height: 4.25rem; display: flex; align-items: center; gap: 1rem; padding: 0 1rem; margin-bottom: 1rem; border-radius: .5rem; background: var(--surface-container); overflow: hidden; }
.app-bar h5 { margin: 0; line-height: 1.15; }
.back-link { width: 2.75rem; height: 2.75rem; flex: 0 0 2.75rem; display: inline-flex; align-items: center; justify-content: center; border-radius: 50%; color: var(--on-surface); text-decoration: none; }
.back-link:hover { background: var(--surface-container-highest); }
.back-link i { display: block; font-size: 1.9rem; line-height: 1; }
.theme-toggle { width: 2.75rem; height: 2.75rem; flex: 0 0 2.75rem; color: var(--on-surface); }
.theme-toggle:hover { background: var(--surface-container-highest); }
.theme-toggle i { font-size: 1.65rem; line-height: 1; }
.stats { margin-left: auto; display: flex; flex-wrap: wrap; justify-content: flex-end; gap: .5rem; }
.stat-pill { display: inline-flex; align-items: center; min-height: 1.45rem; padding: 0 .55rem; border-radius: 999px; background: rgb(255 180 171); color: rgb(105 0 5); font-size: .82rem; font-weight: 700; line-height: 1; white-space: nowrap; }
.live-pill { background: rgb(179 229 252); color: rgb(1 87 155); }
.search-panel { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; align-items: stretch; }
.query-card { width: 100%; inline-size: 100%; min-width: 0; min-inline-size: 0; max-width: 100%; max-inline-size: 100%; min-height: 13rem; display: flex; flex-direction: column; gap: 1rem; border-radius: .5rem; overflow: hidden; }
.query-card h6 { margin: 0; }
.query-form { display: flex; flex-direction: column; gap: 1rem; height: 100%; }
.file-picker { min-height: 4.5rem; display: flex; align-items: center; gap: .75rem; padding: 1rem; border: 1px dashed var(--outline); border-radius: .5rem; background: var(--surface-container-high); color: var(--on-surface); cursor: pointer; }
.file-picker i { font-size: 2rem; color: var(--primary); }
.file-picker span { font-weight: 600; overflow-wrap: anywhere; }
.file-input { position: absolute; width: 1px; height: 1px; opacity: 0; pointer-events: none; }
.query-controls { margin-top: auto; display: grid; grid-template-columns: minmax(8rem, 12rem) 1fr; gap: .75rem; align-items: end; }
.query-controls > *, .query-card .field, .query-card input, .query-card button { min-width: 0; min-inline-size: 0; max-width: 100%; max-inline-size: 100%; }
.query-card .field, .query-card input { width: 100%; inline-size: 100%; }
.sample-controls { grid-template-columns: minmax(10rem, 1fr) minmax(7rem, 10rem) auto; }
.actions { min-width: 0; display: flex; align-items: end; justify-content: flex-end; gap: .5rem; min-height: 4rem; }
.live-card { grid-column: 1 / -1; }
.live-stage { position: relative; aspect-ratio: 16 / 9; min-height: 14rem; overflow: hidden; border-radius: .5rem; background: black; }
.live-stage video { width: 100%; height: 100%; display: block; object-fit: contain; background: black; }
.live-placeholder { position: absolute; inset: 0; display: grid; place-items: center; color: var(--on-surface-variant); background: var(--surface-container-high); }
.live-placeholder[hidden] { display: none; }
.live-placeholder i { font-size: 3rem; }
.live-controls { grid-template-columns: minmax(8rem, 12rem) 1fr; }
.live-actions { flex-wrap: wrap; }
.live-status { min-height: 1.25rem; font-size: .82rem; color: var(--on-surface-variant); }
.live-results { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: .5rem; }
.live-result { display: grid; grid-template-columns: 4.25rem minmax(0, 1fr); gap: .6rem; align-items: center; padding: .45rem 0; border-top: 1px solid var(--outline-variant); }
.live-result img { width: 4.25rem; aspect-ratio: 1 / 1; object-fit: contain; background: white; border-radius: .35rem; }
.live-result-body { min-width: 0; display: grid; gap: .15rem; }
.live-result-body strong, .live-result-body span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.live-result-body span, .live-result-body small { color: var(--on-surface-variant); font-size: .78rem; }
.live-result-body progress { width: 100%; height: .45rem; }
.example-gallery { margin-top: 1.25rem; display: flex; gap: .75rem; overflow-x: auto; overflow-y: hidden; padding: .1rem 0 .6rem; scroll-snap-type: x proximity; }
.example-card { flex: 0 0 11rem; color: inherit; text-decoration: none; border: 1px solid var(--outline-variant); border-radius: .5rem; overflow: hidden; background: var(--surface-container); padding: .35rem; scroll-snap-align: start; }
.example-card img { width: 100%; aspect-ratio: 9 / 16; object-fit: cover; border-radius: .35rem; display: block; }
.result-card img { width: 100%; aspect-ratio: 1 / 1; object-fit: contain; background: white; border-radius: .35rem; display: block; }
.result-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: .75rem; }
.result-card { position: relative; border-radius: .5rem; }
.result-card .rank { position: absolute; top: .55rem; left: .55rem; padding: .1rem .45rem; border-radius: 999px; background: var(--primary); color: var(--on-primary); font-size: .75rem; }
.result-card h6, .result-card p { overflow-wrap: anywhere; }
.result-summary { margin-bottom: 1rem; }
@media (max-width: 900px) {
  .search-panel { grid-template-columns: 1fr; }
  .sample-controls, .query-controls, .live-controls { grid-template-columns: 1fr; }
  .actions { justify-content: stretch; }
  .actions button { width: 100%; }
}
@media (max-width: 560px) {
  main.max { padding: .5rem; }
  .app-bar { align-items: flex-start; flex-direction: column; padding: 1rem; }
  .stats { margin-left: 0; justify-content: flex-start; }
  .theme-toggle { margin-left: 0; }
  .query-card { padding: 1rem; }
  .query-card h6 { font-size: 1.35rem; overflow-wrap: anywhere; }
  .live-stage { min-height: 11rem; }
  .live-results { grid-template-columns: 1fr; }
  .example-gallery { margin-top: .75rem; }
  .example-card { flex-basis: 9rem; }
  .result-grid { grid-template-columns: repeat(auto-fill, minmax(132px, 1fr)); }
}
"#;
