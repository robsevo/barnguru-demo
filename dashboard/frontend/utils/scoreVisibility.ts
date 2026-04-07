"use client";

import { useCallback, useEffect, useState } from "react";

const EVENT = "gretzky:score-visibility";
const KEY_GAMES = "gretzky_hidden_games";
const KEY_ALL   = "gretzky_hide_all_scores";

function readHiddenGames(): Set<number> {
  if (typeof window === "undefined") return new Set();
  const s = localStorage.getItem(KEY_GAMES) ?? "";
  return new Set(s.split(",").map(Number).filter(Boolean));
}

function readHideAll(): boolean {
  if (typeof window === "undefined") return false;
  return localStorage.getItem(KEY_ALL) === "1";
}

function broadcast() {
  window.dispatchEvent(new Event(EVENT));
}

/** Returns true if scores for this game should be hidden. */
export function useScoreHidden(gameId: number): { hidden: boolean; toggle: () => void } {
  const [hideAll, setHideAll]     = useState(readHideAll);
  const [hiddenSet, setHiddenSet] = useState(readHiddenGames);

  const sync = useCallback(() => {
    setHideAll(readHideAll());
    setHiddenSet(readHiddenGames());
  }, []);

  useEffect(() => {
    window.addEventListener(EVENT, sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener(EVENT, sync);
      window.removeEventListener("storage", sync);
    };
  }, [sync]);

  const toggle = useCallback(() => {
    const set = readHiddenGames();
    if (set.has(gameId)) set.delete(gameId); else set.add(gameId);
    localStorage.setItem(KEY_GAMES, [...set].join(","));
    broadcast();
  }, [gameId]);

  return { hidden: hideAll || hiddenSet.has(gameId), toggle };
}

/** Global hide-all toggle for the scoreboard bar. */
export function useHideAllScores(): { hideAll: boolean; toggleAll: () => void } {
  const [hideAll, setHideAll] = useState(readHideAll);

  const sync = useCallback(() => setHideAll(readHideAll()), []);

  useEffect(() => {
    window.addEventListener(EVENT, sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener(EVENT, sync);
      window.removeEventListener("storage", sync);
    };
  }, [sync]);

  const toggleAll = useCallback(() => {
    const next = !readHideAll();
    localStorage.setItem(KEY_ALL, next ? "1" : "0");
    if (next) { localStorage.removeItem(KEY_GAMES); }
    broadcast();
  }, []);

  return { hideAll, toggleAll };
}
