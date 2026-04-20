"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const router = useRouter();
  const [open,        setOpen]       = useState(false);
  const [username,    setUsername]   = useState("");
  const [password,    setPassword]   = useState("");
  const [secretWord,  setSecretWord] = useState("");
  const [error,       setError]      = useState("");
  const [loading,     setLoading]    = useState(false);
  const [showPw,      setShowPw]     = useState(false);
  const [showSecret,  setShowSecret] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const res = await fetch("/api/auth/login", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ username, password, secret_word: secretWord }),
      });
      if (res.ok) {
        router.push("/");
        router.refresh();
      } else {
        setError("Invalid credentials. Try again.");
      }
    } catch {
      setError("Connection error. Try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="relative min-h-dvh flex items-center justify-center overflow-y-auto py-6 sm:pb-[25vh]">
      {/* Background — fixed so it doesn't scroll with content */}
      <div className="fixed inset-0 -z-10"
        style={{
          backgroundImage:    "url('/gretz.jpg')",
          backgroundSize:     "cover",
          backgroundPosition: "center 60%",
          backgroundRepeat:   "no-repeat",
        }}
      />
      {/* Overlays — also fixed */}
      <div className="fixed inset-0 -z-10 bg-[#07080a]/75 backdrop-blur-[2px]" />
      <div className="fixed inset-0 -z-10" style={{ background: "radial-gradient(ellipse 70% 50% at 50% 0%, rgba(210,215,225,0.06) 0%, transparent 65%)" }} />
      <div className="fixed inset-0 -z-10" style={{ background: "radial-gradient(ellipse 60% 40% at 50% 100%, rgba(0,0,0,0.5) 0%, transparent 70%)" }} />

      {/* Card */}
      <div className="relative z-10 w-full max-w-sm mx-4">
        <div className="rounded-2xl border border-[#C9A84C]/[0.18] bg-gradient-to-b from-[#0d0f13]/95 via-[#0a0b0e]/90 to-[#080909]/95 backdrop-blur-2xl overflow-hidden
          shadow-[0_40px_100px_rgba(0,0,0,0.90),0_8px_32px_rgba(0,0,0,0.70),inset_0_1px_0_rgba(220,225,235,0.10),inset_0_-1px_0_rgba(0,0,0,0.40)]">

          {/* Branding — always visible */}
          <button
            onClick={() => setOpen(v => !v)}
            className="w-full px-7 pt-6 pb-5 flex flex-col gap-4 group transition-all duration-200 hover:bg-white/[0.02]"
          >
            <div className="flex items-center gap-3">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src="/logo-circle.png" alt="GRTZKY" className="h-14 w-auto" />
              <span className="text-white/15 font-thin text-lg">|</span>
              <div className="text-left">
                <div className="text-[22px] font-semibold tracking-[0.14em] uppercase bg-gradient-to-r from-white via-[#e8eaec] to-[#b0b8c2] bg-clip-text text-transparent leading-none">
                  GRTZKY
                </div>
                <div className="text-[8px] font-semibold uppercase tracking-[0.22em] text-white/20 mt-1">
                  Bayesian Analytics and Rating Network
                </div>
              </div>
            </div>
            <div className={`w-full flex items-center justify-center gap-2 px-5 py-3.5 rounded-xl border transition-all duration-200 ${
              open
                ? "border-[#C9A84C]/[0.20] bg-[#C9A84C]/[0.06] text-[#C9A84C]/50"
                : "border-[#C9A84C]/[0.35] bg-gradient-to-b from-[#C9A84C]/[0.12] to-[#C9A84C]/[0.05] text-[#C9A84C] shadow-[0_2px_12px_rgba(201,168,76,0.12),inset_0_1px_0_rgba(201,168,76,0.18)] group-hover:border-[#C9A84C]/[0.55] group-hover:from-[#C9A84C]/[0.18]"
            }`}>
              <span className="text-[13px] font-black uppercase tracking-[0.22em]">
                {open ? "Close" : "Sign In"}
              </span>
              <span className={`text-[10px] transition-all duration-300 ${open ? "rotate-180" : ""}`}>▼</span>
            </div>
          </button>

          {/* Collapsible form */}
          <div className={`overflow-hidden transition-all duration-300 ease-in-out ${open ? "max-h-[900px] opacity-100" : "max-h-0 opacity-0"}`}>
            <div className="h-px w-full bg-gradient-to-r from-transparent via-white/[0.04] to-transparent" />

            <form onSubmit={handleSubmit} className="px-6 py-5 space-y-3.5">
              <div className="space-y-1">
                <label className="text-[8px] font-black uppercase tracking-[0.25em] text-white/25">Username</label>
                <input
                  type="text"
                  value={username}
                  onChange={e => setUsername(e.target.value)}
                  autoComplete="username"
                  required
                  className="w-full bg-white/[0.04] border border-white/[0.07] rounded-xl px-4 py-2.5 text-base sm:text-[13px] text-white/90 placeholder-white/15 outline-none focus:border-white/[0.22] focus:bg-white/[0.07] transition-all duration-150 shadow-[inset_0_1px_0_rgba(0,0,0,0.2)]"
                  placeholder="username"
                />
              </div>

              <div className="space-y-1">
                <label className="text-[8px] font-black uppercase tracking-[0.25em] text-white/25">Password</label>
                <div className="relative">
                  <input
                    type={showPw ? "text" : "password"}
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    autoComplete="current-password"
                    required
                    className="w-full bg-white/[0.04] border border-white/[0.07] rounded-xl px-4 py-2.5 pr-14 text-base sm:text-[13px] text-white/90 placeholder-white/15 outline-none focus:border-white/[0.22] focus:bg-white/[0.07] transition-all duration-150 shadow-[inset_0_1px_0_rgba(0,0,0,0.2)]"
                    placeholder="••••••••"
                  />
                  <button type="button" onClick={() => setShowPw(v => !v)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-[8px] font-black uppercase tracking-wider text-white/20 hover:text-white/50 transition-colors">
                    {showPw ? "Hide" : "Show"}
                  </button>
                </div>
              </div>

              <div className="space-y-1">
                <label className="text-[8px] font-black uppercase tracking-[0.25em] text-white/25">Secret Word</label>
                <div className="relative">
                  <input
                    type={showSecret ? "text" : "password"}
                    value={secretWord}
                    onChange={e => setSecretWord(e.target.value)}
                    required
                    className="w-full bg-white/[0.04] border border-white/[0.07] rounded-xl px-4 py-2.5 pr-14 text-base sm:text-[13px] text-white/90 placeholder-white/15 outline-none focus:border-white/[0.22] focus:bg-white/[0.07] transition-all duration-150 shadow-[inset_0_1px_0_rgba(0,0,0,0.2)]"
                    placeholder="••••••••"
                  />
                  <button type="button" onClick={() => setShowSecret(v => !v)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-[8px] font-black uppercase tracking-wider text-white/20 hover:text-white/50 transition-colors">
                    {showSecret ? "Hide" : "Show"}
                  </button>
                </div>
              </div>

              {error && (
                <p className="text-[11px] text-[#f87171]/80 font-semibold text-center">{error}</p>
              )}

              <button
                type="submit"
                disabled={loading}
                className="w-full py-2.5 rounded-xl border border-white/[0.10] bg-gradient-to-b from-white/[0.08] to-white/[0.04] text-[10px] font-black uppercase tracking-[0.22em] text-white/60 hover:border-white/[0.22] hover:text-white/85 hover:from-white/[0.12] hover:to-white/[0.06] active:scale-[0.98] transition-all duration-150 disabled:opacity-40 disabled:cursor-not-allowed shadow-[0_4px_16px_rgba(0,0,0,0.5),inset_0_1px_0_rgba(255,255,255,0.08),inset_0_-1px_0_rgba(0,0,0,0.35)]"
              >
                {loading ? "Authenticating…" : "Enter"}
              </button>

              {/* Home screen tip */}
              <div className="rounded-xl border border-white/[0.05] bg-white/[0.02] px-3.5 py-3">
                <p className="text-[8px] font-black uppercase tracking-[0.2em] text-white/20 mb-1.5">Pro tip</p>
                <p className="text-[10px] text-white/25 leading-relaxed">
                  Open GRTZKY in your browser, tap <span className="text-white/40 font-semibold">Share → Add to Home Screen</span> and you get the app icon, full-screen mode, and instant access — no App Store needed.
                </p>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
