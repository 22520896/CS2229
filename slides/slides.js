const slides = Array.from(document.querySelectorAll(".slide"));
const previous = document.querySelector("#previous");
const next = document.querySelector("#next");
const fullscreen = document.querySelector("#fullscreen");
const counter = document.querySelector("#counter");
let position = 0;

function show(index, updateHash = true) {
  position = Math.max(0, Math.min(slides.length - 1, index));
  slides.forEach((slide, slideIndex) => {
    slide.hidden = slideIndex !== position;
  });
  previous.disabled = position === 0;
  next.disabled = position === slides.length - 1;
  counter.textContent = `${position + 1} / ${slides.length}`;
  if (updateHash) history.replaceState(null, "", `#${position + 1}`);
}

function readPosition() {
  const value = Number(location.hash.slice(1));
  show(Number.isInteger(value) && value > 0 ? value - 1 : 0, false);
}

function setPresentation(active) {
  document.body.classList.toggle("presenting", active);
  fullscreen.setAttribute("aria-pressed", String(active));
}

async function toggleFullscreen() {
  const active = !document.body.classList.contains("presenting");
  setPresentation(active);
  if (active && document.fullscreenEnabled) {
    await document.documentElement.requestFullscreen().catch(() => {});
  } else if (!active && document.fullscreenElement) {
    await document.exitFullscreen();
  }
}

previous.addEventListener("click", () => show(position - 1));
next.addEventListener("click", () => show(position + 1));
fullscreen.addEventListener("click", () => toggleFullscreen().catch(() => {}));
window.addEventListener("hashchange", readPosition);
document.addEventListener("keydown", (event) => {
  if (event.altKey || event.metaKey || event.ctrlKey) return;
  const navigation = {
    ArrowLeft: position - 1, PageUp: position - 1,
    ArrowRight: position + 1, PageDown: position + 1,
    Home: 0, End: slides.length - 1,
  };
  if (event.key in navigation) {
    event.preventDefault();
    show(navigation[event.key]);
  } else if (event.key.toLowerCase() === "f") {
    toggleFullscreen().catch(() => {});
  } else if (event.key === "Escape" && document.body.classList.contains("presenting")) {
    toggleFullscreen().catch(() => {});
  }
});
readPosition();
