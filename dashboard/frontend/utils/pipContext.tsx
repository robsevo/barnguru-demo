"use client";

import { createContext, useCallback, useContext, useState } from "react";

export interface PiPStream {
  url:       string;
  title:     string;
  gameId:    string;
  embedOnly?: boolean;
}

interface PiPCtx {
  stream:     PiPStream | null;
  activate:   (s: PiPStream) => void;
  deactivate: () => void;
}

const PiPContext = createContext<PiPCtx>({
  stream:     null,
  activate:   () => {},
  deactivate: () => {},
});

export function PiPProvider({ children }: { children: React.ReactNode }) {
  const [stream, setStream] = useState<PiPStream | null>(null);
  const activate   = useCallback((s: PiPStream) => setStream(s), []);
  const deactivate = useCallback(() => setStream(null), []);
  return (
    <PiPContext.Provider value={{ stream, activate, deactivate }}>
      {children}
    </PiPContext.Provider>
  );
}

export function usePiP() {
  return useContext(PiPContext);
}
