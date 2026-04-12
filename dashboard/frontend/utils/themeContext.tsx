"use client";

import { createContext, useContext, useState, useEffect, useCallback, useRef } from "react";

export interface TeamTheme {
  abbrev: string;
  primaryColor: string;   // hex e.g. "#AF1E2D"
  secondaryColor: string; // hex e.g. "#192168"
  logoUrl: string;
}

interface ThemeContextValue {
  theme: TeamTheme | null;
  setTeamTheme: (theme: TeamTheme) => void;
  clearTheme: () => void;
  isShowingLoader: boolean;
  cortexPinned: boolean;
  setCortexPinned: (v: boolean) => void;
  isCortexLoading: boolean;
  isGrtzkyLoading: boolean;
  previewActive: boolean;
  setPreviewTheme: (theme: TeamTheme) => void;
  clearPreview: () => void;
}

export const ThemeContext = createContext<ThemeContextValue>({
  theme: null,
  setTeamTheme: () => {},
  clearTheme: () => {},
  isShowingLoader: false,
  cortexPinned: false,
  setCortexPinned: () => {},
  isCortexLoading: false,
  isGrtzkyLoading: false,
  previewActive: false,
  setPreviewTheme: () => {},
  clearPreview: () => {},
});

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<TeamTheme | null>(null);
  const [isShowingLoader, setIsShowingLoader] = useState(false);
  const [cortexPinned, setCortexPinnedState] = useState(false);
  const [isCortexLoading, setIsCortexLoading] = useState(false);
  const [isGrtzkyLoading, setIsGrtzkyLoading] = useState(false);
  const [previewActive, setPreviewActive] = useState(false);
  // Stable ref so clearPreview always sees the current value without re-creating
  const previewActiveRef = useRef(false);

  useEffect(() => {
    try {
      const saved = localStorage.getItem("grtzky_teamTheme");
      if (saved) setThemeState(JSON.parse(saved));
      setCortexPinnedState(localStorage.getItem("grtzky_cortexPinned") === "1");
    } catch {}
  }, []);

  const setTeamTheme = useCallback((newTheme: TeamTheme) => {
    // Selecting a team theme clears Cortex pin so the two never coexist
    localStorage.removeItem("grtzky_cortexPinned");
    setCortexPinnedState(false);
    localStorage.setItem("grtzky_teamTheme", JSON.stringify(newTheme));
    setThemeState(newTheme);
    setPreviewActive(false);
    setIsShowingLoader(true);
    const t = setTimeout(() => setIsShowingLoader(false), 6000);
    return () => clearTimeout(t);
  }, []);

  const clearTheme = useCallback(() => {
    localStorage.removeItem("grtzky_teamTheme");
    setThemeState(null);
    setIsGrtzkyLoading(true);
    const t = setTimeout(() => setIsGrtzkyLoading(false), 6000);
    return () => clearTimeout(t);
  }, []);

  // Preview: apply a team theme visually without persisting to localStorage.
  // Reverts when the team page unmounts (clearPreview called in its useEffect cleanup).
  const setPreviewTheme = useCallback((newTheme: TeamTheme) => {
    previewActiveRef.current = true;
    setPreviewActive(true);
    setThemeState(newTheme);
  }, []);

  // Stable — uses ref so the cleanup closure in teams page always works correctly
  // even if captured before previewActive state was set.
  const clearPreview = useCallback(() => {
    if (!previewActiveRef.current) return;
    previewActiveRef.current = false;
    setPreviewActive(false);
    try {
      const saved = localStorage.getItem("grtzky_teamTheme");
      setThemeState(saved ? JSON.parse(saved) : null);
    } catch {
      setThemeState(null);
    }
  }, []); // stable — no deps needed because we use a ref

  const setCortexPinned = useCallback((v: boolean) => {
    if (v) {
      // Switching to Cortex — clear any active team theme first
      localStorage.removeItem("grtzky_teamTheme");
      setThemeState(null);
      localStorage.setItem("grtzky_cortexPinned", "1");
      setIsCortexLoading(true);
      const t = setTimeout(() => setIsCortexLoading(false), 6000);
      setCortexPinnedState(true);
      return () => clearTimeout(t);
    } else {
      localStorage.removeItem("grtzky_cortexPinned");
      setCortexPinnedState(false);
      setIsGrtzkyLoading(true);
      const t = setTimeout(() => setIsGrtzkyLoading(false), 6000);
      return () => clearTimeout(t);
    }
  }, []);

  return (
    <ThemeContext.Provider value={{ theme, setTeamTheme, clearTheme, isShowingLoader, cortexPinned, setCortexPinned, isCortexLoading, isGrtzkyLoading, previewActive, setPreviewTheme, clearPreview }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  return useContext(ThemeContext);
}
