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
    /** Egg incubation: one step per real user message (App.send). */
    sentinelBuddyStep?: () => void;
    /** Command-triggered animation from chat meta (buddy_anim). */
    sentinelBuddyAnim?: (name?: string) => void;
    /** Re-fetch egg/buddy state (after login or /buddy egg). */
    sentinelBuddyRefresh?: () => void;
    /** First login: carry the guest (device) egg/buddy into the account. */
    sentinelBuddyAdopt?: (accountId: string) => void;
    /** Optional API key injected by deployments. */
    __API_KEY?: string;
  }
}
