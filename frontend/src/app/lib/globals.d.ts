export {};

declare global {
  interface Window {
    marked?: { parse: (src: string, opts?: unknown) => string };
    DOMPurify?: { sanitize: (html: string, opts?: unknown) => string };
    hljs?: { highlightElement: (el: Element) => void };
    /** Chat pipeline hooks — set by lib/buddy-float.ts initBuddy(). */
    sentinelBuddyReact?: (text: string, role?: string) => void;
    sentinelBuddySetVisible?: (on: boolean) => void;
    sentinelBuddyPet?: () => void;
    /** Optional API key injected by deployments. */
    __API_KEY?: string;
  }
}
