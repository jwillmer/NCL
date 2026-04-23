/**
 * Shared vessel/type/class provider.
 *
 * Replaces the per-page `listVessels()` + `listVesselTypes()` + `listVesselClasses()`
 * mount fetches on ConversationsPage and ChatPage with a single session-scoped
 * cache. The provider hydrates synchronously from sessionStorage (5-minute TTL)
 * so cross-page navigation is instant, then kicks off a fresh fetch in the
 * background to replace stale state.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import {
  Vessel,
  listVessels,
  listVesselTypes,
  listVesselClasses,
} from "@/lib/conversations";
import { useAuth } from "@/components/auth";

type VesselState = {
  vessels: Vessel[];
  types: string[];
  classes: string[];
  lookup: Record<string, string>;
  loading: boolean;
  refresh: () => Promise<void>;
};

const Ctx = createContext<VesselState | null>(null);

const STORAGE_KEY = "mtss.vessels.v1";
const STORAGE_TTL_MS = 5 * 60 * 1000;

type StoredShape = {
  vessels: Vessel[];
  types: string[];
  classes: string[];
  ts: number;
};

function loadFromStorage(): Omit<StoredShape, "ts"> | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as StoredShape;
    if (
      !parsed ||
      typeof parsed.ts !== "number" ||
      Date.now() - parsed.ts > STORAGE_TTL_MS
    ) {
      return null;
    }
    return {
      vessels: parsed.vessels ?? [],
      types: parsed.types ?? [],
      classes: parsed.classes ?? [],
    };
  } catch {
    return null;
  }
}

function saveToStorage(data: Omit<StoredShape, "ts">): void {
  try {
    sessionStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ ...data, ts: Date.now() } satisfies StoredShape),
    );
  } catch {
    /* sessionStorage unavailable — ignore */
  }
}

export function VesselProvider({ children }: { children: ReactNode }) {
  const { session, loading: authLoading } = useAuth();
  const cached = loadFromStorage();
  const [vessels, setVessels] = useState<Vessel[]>(cached?.vessels ?? []);
  const [types, setTypes] = useState<string[]>(cached?.types ?? []);
  const [classes, setClasses] = useState<string[]>(cached?.classes ?? []);
  const [loading, setLoading] = useState<boolean>(!cached);

  const refresh = useCallback(async () => {
    const [v, t, c] = await Promise.all([
      listVessels(),
      listVesselTypes(),
      listVesselClasses(),
    ]);
    setVessels(v);
    setTypes(t);
    setClasses(c);
    setLoading(false);
    saveToStorage({ vessels: v, types: t, classes: c });
  }, []);

  useEffect(() => {
    if (authLoading) return;
    if (!session) {
      setLoading(false);
      return;
    }
    refresh().catch((err) => {
      console.error("VesselProvider: refresh failed", err);
      setLoading(false);
    });
  }, [authLoading, session, refresh]);

  const lookup = useMemo(
    () => Object.fromEntries(vessels.map((v) => [v.id, v.name])),
    [vessels],
  );

  const value = useMemo<VesselState>(
    () => ({ vessels, types, classes, lookup, loading, refresh }),
    [vessels, types, classes, lookup, loading, refresh],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useVessels(): VesselState {
  const v = useContext(Ctx);
  if (!v) {
    throw new Error("useVessels must be used within a VesselProvider");
  }
  return v;
}
