/**
 * Hook that wraps vite-plugin-pwa's `useRegisterSW` to expose the
 * update-available / offline-ready flags to the rest of the app.
 *
 * Registration is automatic (`injectRegister: "auto"` in vite.config.ts),
 * but we still own the poll cadence: once the SW is registered we kick
 * `registration.update()` on an hourly interval so users who keep the tab
 * open overnight pick up releases without a hard reload.
 */

import { useRegisterSW } from "virtual:pwa-register/react";

const ONE_HOUR_MS = 60 * 60 * 1000;

export function usePwaUpdate() {
  return useRegisterSW({
    onRegisteredSW(_swUrl, registration) {
      if (!registration) return;
      // Stagger first poll by 10s to avoid competing with initial boot.
      setTimeout(() => {
        registration.update().catch(() => {
          /* ignore transient network errors */
        });
        setInterval(() => {
          registration.update().catch(() => {
            /* ignore */
          });
        }, ONE_HOUR_MS);
      }, 10_000);
    },
    onRegisterError(err) {
      console.error("SW register failed", err);
    },
  });
}
