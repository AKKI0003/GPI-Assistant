// static/effects.js
//
// Small, dependency-free visual effects shared across every static page.
// Deliberately vanilla JS -- no React, no build step, no CDN fetch -- so
// this keeps working on a locked-down company network exactly like the
// rest of this app (see RUNNING_REMOTELY.md). The "react bits"-style
// look (spotlight-follow cards, animated aurora backgrounds, smooth
// entrance transitions) is achieved with plain CSS custom properties
// updated from a mousemove listener, and CSS animations triggered by
// adding a class -- the same techniques those components use under the
// hood, just without React/JSX around them.

// ── Spotlight-follow effect ─────────────────────────────────────────────
// Attach to any container with children carrying the `.spotlight` class
// (see style.css's `.spotlight` rules). Tracks the pointer position as
// CSS custom properties (--spot-x/--spot-y) on each spotlighted element,
// which a radial-gradient background reads to make a soft glow follow
// the cursor across the card's surface.
function initSpotlights(root = document) {
  const targets = root.querySelectorAll(".spotlight");
  if (!targets.length) return;

  function updateFromEvent(el, clientX, clientY) {
    const rect = el.getBoundingClientRect();
    el.style.setProperty("--spot-x", `${clientX - rect.left}px`);
    el.style.setProperty("--spot-y", `${clientY - rect.top}px`);
  }

  targets.forEach((el) => {
    el.addEventListener("pointermove", (e) => updateFromEvent(el, e.clientX, e.clientY));
    el.addEventListener("pointerleave", () => {
      el.style.setProperty("--spot-x", "50%");
      el.style.setProperty("--spot-y", "50%");
    });
  });
}

// ── Entrance animation ──────────────────────────────────────────────────
// Adds a class that triggers a CSS fade+rise animation, then removes it
// once finished so the element is left in a clean state (re-triggerable,
// no leftover animation state affecting later layout/reflow).
function animateIn(el, delayMs = 0) {
  if (!el) return;
  el.classList.remove("enter-anim");
  // Force reflow so re-adding the class restarts the animation even if
  // it was already applied to this element once before.
  void el.offsetWidth;
  setTimeout(() => el.classList.add("enter-anim"), delayMs);
}

// ── Staggered entrance for a group of elements ──────────────────────────
// Used for the auth-card fields and the sidebar chat list, so items
// settle in one after another rather than all snapping in at once.
function animateStagger(elements, stepMs = 45) {
  elements.forEach((el, i) => animateIn(el, i * stepMs));
}

// Auto-init spotlight tracking for anything already on the page at load
// time; pages that add cards dynamically (app.js's chat list, upload.js's
// file rows) call initSpotlights(container) again after inserting them.
document.addEventListener("DOMContentLoaded", () => initSpotlights());
