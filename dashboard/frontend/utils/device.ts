// Single UA-based mobile check shared across CV/WS gates so the heavy live-
// tracking pipeline stays off phones. Returns false during SSR; client hooks
// re-evaluate on mount, so the desktop first-paint path isn't penalised.
export function isMobileDevice(): boolean {
  if (typeof navigator === "undefined") return false;
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i
    .test(navigator.userAgent);
}
