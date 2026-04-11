"use client";

import { createContext, useContext, useState, useEffect, useCallback } from "react";

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
}

export const ThemeContext = createContext<ThemeContextValue>({
  theme: null,
  setTeamTheme: () => {},
  clearTheme: () => {},
  isShowingLoader: false,
  cortexPinned: false,
  setCortexPinned: () => {},
});

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<TeamTheme | null>(null);
  const [isShowingLoader, setIsShowingLoader] = useState(false);
  const [cortexPinned, setCortexPinnedState] = useState(false);

  useEffect(() => {
    try {
      const saved = localStorage.getItem("grtzky_teamTheme");
      if (saved) setThemeState(JSON.parse(saved));
      setCortexPinnedState(localStorage.getItem("grtzky_cortexPinned") === "1");
    } catch {}
  }, []);

  const setTeamTheme = useCallback((newTheme: TeamTheme) => {
    localStorage.setItem("grtzky_teamTheme", JSON.stringify(newTheme));
    setThemeState(newTheme);
    setIsShowingLoader(true);
    const t = setTimeout(() => setIsShowingLoader(false), 6000);
    return () => clearTimeout(t);
  }, []);

  const clearTheme = useCallback(() => {
    localStorage.removeItem("grtzky_teamTheme");
    setThemeState(null);
  }, []);

  const setCortexPinned = useCallback((v: boolean) => {
    if (v) localStorage.setItem("grtzky_cortexPinned", "1");
    else localStorage.removeItem("grtzky_cortexPinned");
    setCortexPinnedState(v);
  }, []);

  return (
    <ThemeContext.Provider value={{ theme, setTeamTheme, clearTheme, isShowingLoader, cortexPinned, setCortexPinned }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  return useContext(ThemeContext);
}
