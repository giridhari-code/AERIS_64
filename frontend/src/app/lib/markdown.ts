/**
 * Assistant-bubble markdown via the locally vendored marked + DOMPurify +
 * highlight.js (script tags in index.html). Plain-text fallback if missing.
 * Mirrors the legacy inline renderInto() exactly.
 */
export function renderInto(el: HTMLElement, text: unknown): void {
  const src = text == null ? "" : String(text);
  try {
    if (window.marked && window.DOMPurify) {
      const html = window.marked.parse(src, { gfm: true, breaks: true });
      el.innerHTML = window.DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
      el.classList.add("md");
      if (window.hljs) {
        el.querySelectorAll("pre code").forEach((code) => {
          try {
            window.hljs?.highlightElement(code);
          } catch (_) {}
        });
      }
      return;
    }
  } catch (_) {}
  el.textContent = src;
}
