"use client";

import { createContext, useContext, useState, useEffect, useCallback } from "react";

export interface TeamTheme {
  abbrev: string;
  primaryColor: string; // hex e.g. "#6CACE4"
  logoUrl: string;
}

interface ThemeContextValue {
  theme: TeamTheme | null;
  setTeamTheme: (theme: TeamTheme) => void;
  clearTheme: () => void;
  isShowingLoader: boolean;
}

export const ThemeContext = createContext<ThemeContextValue>({
  theme: null,
  setTeamTheme: () => {},
  clearTheme: () => {},
  isShowingLoader: false,
});

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<TeamTheme | null>(null);
  const [isShowingLoader, setIsShowingLoader] = useState(false);

  // Rehydrate from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem("grtzky_teamTheme");
      if (saved) {
        setThemeState(JSON.parse(saved));
      }
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

  return (
    <ThemeContext.Provider value={{ theme, setTeamTheme, clearTheme, isShowingLoader }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  return useContext(ThemeContext);
}
