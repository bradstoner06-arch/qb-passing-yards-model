"""
qb_gui.py
==================================
Desktop GUI for the QB passing yards model. Pulls live current-season
data from nflverse (see data_loader.py), projects passing yards two ways
(a transparent adjustment model, or a global regression trained across
all QBs), applies weather/Vegas context, runs a Monte Carlo simulation,
and reports edge + Kelly stake against the sportsbook line and odds you
enter.

Run:
    python qb_gui.py

Requires:
    pip install nfl_data_py numpy pandas
"""

import threading
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

from data_loader import (
    load_weekly_player_stats,
    load_team_defense_epa,
    get_qb_pregame_features,
)
from model import (
    project_mean_yards,
    simulate_distribution,
    implied_probability_from_american_odds,
    kelly_fraction,
    build_training_table,
    train_global_regression,
    predict_global_regression,
    apply_weather,
    cap_std,
)

CURRENT_SEASON_DEFAULT = 2026  # change at season rollover, or just type a different year in the GUI


# =========================
# GUI
# =========================
class QBGuiApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("QB Passing Yards Model — Projection & Edge")
        self.geometry("820x880")

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        self.weekly_df = None
        self.def_epa_df = None
        self.global_model = None  # (beta, mu, sigma, resid_std) once trained

        self._build_ui()
        self._load_data_async(initial=True)

    # ---------- UI construction ----------

    def _build_ui(self):
        outer = ttk.Frame(self, padding=14)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer, text="QB Passing Yards — Projection & Edge",
            font=("Segoe UI", 16, "bold")
        ).pack(pady=(0, 4))

        self.status_label = ttk.Label(outer, text="Loading season data...", foreground="#555")
        self.status_label.pack(pady=(0, 8))

        # ---- Data / season controls ----
        data_frame = ttk.LabelFrame(outer, text="Data", padding=10)
        data_frame.pack(fill="x", pady=4)

        ttk.Label(data_frame, text="Season").grid(row=0, column=0, sticky="w")
        self.season_entry = ttk.Entry(data_frame, width=8)
        self.season_entry.insert(0, str(CURRENT_SEASON_DEFAULT))
        self.season_entry.grid(row=0, column=1, padx=4)

        self.reload_btn = ttk.Button(data_frame, text="Reload Data", command=self._load_data_async)
        self.reload_btn.grid(row=0, column=2, padx=10)

        self.train_btn = ttk.Button(data_frame, text="Train Global Model", command=self._train_global_async)
        self.train_btn.grid(row=0, column=3, padx=4)

        self.model_status = ttk.Label(data_frame, text="Global model: not trained", foreground="#555")
        self.model_status.grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))

        # ---- Matchup inputs ----
        inputs = ttk.LabelFrame(outer, text="Matchup", padding=10)
        inputs.pack(fill="x", pady=8)

        ttk.Label(inputs, text="QB full name").grid(row=0, column=0, sticky="w")
        self.qb_entry = ttk.Entry(inputs, width=24)
        self.qb_entry.insert(0, "Patrick Mahomes")
        self.qb_entry.grid(row=0, column=1, padx=4, pady=2)

        ttk.Label(inputs, text="Opponent (team code)").grid(row=0, column=2, sticky="w")
        self.opp_entry = ttk.Entry(inputs, width=8)
        self.opp_entry.insert(0, "BUF")
        self.opp_entry.grid(row=0, column=3, padx=4)

        self.home_var = tk.IntVar(value=1)
        ttk.Checkbutton(inputs, text="QB's team is HOME", variable=self.home_var).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=2
        )

        proj_fields = [
            ("Vegas Total", "47.5"),
            ("Spread (QB's team)", "-3.0"),
        ]
        self.proj_entries = {}
        for i, (label, default) in enumerate(proj_fields, start=2):
            ttk.Label(inputs, text=label).grid(row=i, column=0, sticky="w")
            e = ttk.Entry(inputs, width=10)
            e.insert(0, default)
            e.grid(row=i, column=1, sticky="w", pady=2)
            self.proj_entries[label] = e

        # Model mode
        ttk.Label(inputs, text="Projection method").grid(row=2, column=2, sticky="w")
        self.method_var = tk.StringVar(value="adjustment")
        method_box = ttk.Combobox(
            inputs, textvariable=self.method_var, width=22, state="readonly",
            values=["adjustment", "global_regression", "blend (avg of both)"]
        )
        method_box.grid(row=2, column=3, padx=4, sticky="w")

        # ---- Weather ----
        weather = ttk.LabelFrame(outer, text="Weather (skip if indoor)", padding=10)
        weather.pack(fill="x", pady=8)

        weather_fields = [
            ("Wind (mph)", "0"),
            ("Temp (F)", "70"),
        ]
        self.weather_entries = {}
        for i, (label, default) in enumerate(weather_fields):
            ttk.Label(weather, text=label).grid(row=0, column=i * 2, sticky="w")
            e = ttk.Entry(weather, width=8)
            e.insert(0, default)
            e.grid(row=0, column=i * 2 + 1, padx=4)
            self.weather_entries[label] = e

        self.rain_var = tk.IntVar(value=0)
        self.snow_var = tk.IntVar(value=0)
        self.indoor_var = tk.IntVar(value=1)
        ttk.Checkbutton(weather, text="Rain", variable=self.rain_var).grid(row=0, column=4, padx=6)
        ttk.Checkbutton(weather, text="Snow", variable=self.snow_var).grid(row=0, column=5, padx=6)
        ttk.Checkbutton(weather, text="Indoor / dome", variable=self.indoor_var).grid(row=0, column=6, padx=6)

        # ---- Sportsbook / Kelly ----
        book = ttk.LabelFrame(outer, text="Sportsbook line & Kelly staking", padding=10)
        book.pack(fill="x", pady=8)

        book_fields = [
            ("Passing yards line", "265.5"),
            ("Over odds (American)", "-115"),
            ("Under odds (American)", "-105"),
            ("Kelly fraction (0.25=quarter)", "0.25"),
        ]
        self.book_entries = {}
        for i, (label, default) in enumerate(book_fields):
            ttk.Label(book, text=label).grid(row=i // 2, column=(i % 2) * 2, sticky="w", pady=2)
            e = ttk.Entry(book, width=10)
            e.insert(0, default)
            e.grid(row=i // 2, column=(i % 2) * 2 + 1, padx=4, pady=2, sticky="w")
            self.book_entries[label] = e

        ttk.Button(outer, text="Run Projection", command=self.run).pack(pady=10)

        self.output = tk.Text(outer, height=20, font=("Consolas", 10))
        self.output.pack(fill="both", expand=True)

    # ---------- Data loading (threaded so the GUI doesn't freeze) ----------

    def _load_data_async(self, initial=False):
        self.reload_btn.config(state="disabled")
        self.status_label.config(text="Loading season data from nflverse... this can take a minute.")

        def worker():
            try:
                season = int(self.season_entry.get())
                weekly = load_weekly_player_stats(season)
                def_epa = load_team_defense_epa(season)
                self.weekly_df = weekly
                self.def_epa_df = def_epa
                self.after(0, lambda: self.status_label.config(
                    text=f"Loaded {season} data: {weekly['player_display_name'].nunique()} QBs, "
                         f"{len(def_epa)} teams with defense EPA.",
                    foreground="#555"
                ))
            except Exception as e:
                msg = str(e)
                self.after(0, lambda: self.status_label.config(text=f"Data load FAILED: {msg}", foreground="#b00"))
                if not initial:
                    self.after(0, lambda: messagebox.showerror("Data load error", msg))
            finally:
                self.after(0, lambda: self.reload_btn.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    def _train_global_async(self):
        if self.weekly_df is None:
            messagebox.showwarning("No data", "Load season data first.")
            return

        self.train_btn.config(state="disabled")
        self.model_status.config(text="Training global model...")

        def worker():
            try:
                X, y, names = build_training_table(self.weekly_df)
                if len(X) < 50:
                    raise ValueError(
                        f"Only {len(X)} training rows after filtering -- too few to fit reliably. "
                        "Try a season with more completed weeks, or lower min_prior_games."
                    )
                beta, mu, sigma, resid_std = train_global_regression(X, y)
                self.global_model = (beta, mu, sigma, resid_std)
                self.after(0, lambda: self.model_status.config(
                    text=f"Global model: trained on {len(X)} QB-weeks across "
                         f"{self.weekly_df['player_display_name'].nunique()} QBs "
                         f"(residual std: {resid_std:.1f} yds)",
                    foreground="#070"
                ))
            except Exception as e:
                msg = str(e)
                self.after(0, lambda: self.model_status.config(text=f"Training FAILED: {msg}", foreground="#b00"))
                self.after(0, lambda: messagebox.showerror("Training error", msg))
            finally:
                self.after(0, lambda: self.train_btn.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    # ---------- Run projection ----------

    def run(self):
        if self.weekly_df is None or self.def_epa_df is None:
            messagebox.showwarning("No data", "Season data hasn't loaded yet. Wait for it or click Reload Data.")
            return

        method = self.method_var.get()
        if method in ("global_regression", "blend (avg of both)") and self.global_model is None:
            messagebox.showwarning(
                "Model not trained",
                "You selected a method that needs the global regression. Click 'Train Global Model' first."
            )
            return

        try:
            qb_name = self.qb_entry.get().strip()
            opponent = self.opp_entry.get().strip().upper()
            is_home = bool(self.home_var.get())

            vegas_total = float(self.proj_entries["Vegas Total"].get())
            vegas_spread = float(self.proj_entries["Spread (QB's team)"].get())

            wind = float(self.weather_entries["Wind (mph)"].get())
            temp = float(self.weather_entries["Temp (F)"].get())
            rain = bool(self.rain_var.get())
            snow = bool(self.snow_var.get())
            indoor = bool(self.indoor_var.get())

            line = float(self.book_entries["Passing yards line"].get())
            over_odds = float(self.book_entries["Over odds (American)"].get())
            under_odds = float(self.book_entries["Under odds (American)"].get())
            kelly_mult = float(self.book_entries["Kelly fraction (0.25=quarter)"].get())

            features, yard_history = get_qb_pregame_features(self.weekly_df, qb_name)

            opp_row = self.def_epa_df[self.def_epa_df["team"] == opponent]
            opp_def_epa = float(opp_row["def_pass_epa"].iloc[0]) if not opp_row.empty else 0.0
            opp_note = "" if not opp_row.empty else "  (WARNING: opponent not found, used league avg 0.0)"

            adj_mean = project_mean_yards(
                qb_features=features, opp_def_epa=opp_def_epa,
                vegas_total=vegas_total, vegas_spread=vegas_spread,
                home_field=1 if is_home else 0,
            )

            reg_mean = None
            if self.global_model is not None:
                beta, mu, sigma, resid_std = self.global_model
                reg_features = dict(features)
                reg_mean = predict_global_regression(beta, mu, sigma, reg_features)
                # regression doesn't see Vegas/opponent directly, so apply
                # the same external adjustments on top of its baseline
                reg_mean += (vegas_total - 44.0) * 1.1
                reg_mean += vegas_spread * -2.5
                reg_mean += opp_def_epa * 60.0
                reg_mean += 4.0 if is_home else -4.0

            if method == "adjustment":
                mean = adj_mean
                std = features["std_yards"]
            elif method == "global_regression":
                mean = reg_mean
                std = self.global_model[3]  # residual std from training
            else:  # blend
                mean = (adj_mean + reg_mean) / 2.0
                std = (features["std_yards"] + self.global_model[3]) / 2.0

            std = cap_std(std)
            mean, std = apply_weather(mean, std, wind, temp, rain, snow, indoor)

            sims = simulate_distribution(mean, std, n=20000)
            sims = np.clip(sims, 0, None)

            over_prob = float((sims > line).mean())
            under_prob = 1 - over_prob

            over_kelly, over_edge = kelly_fraction(over_prob, over_odds, kelly_mult)
            under_kelly, under_edge = kelly_fraction(under_prob, under_odds, kelly_mult)

            self._render_report(
                qb_name, opponent, opp_note, opp_def_epa, features, method,
                adj_mean, reg_mean, mean, std, sims, line,
                over_prob, under_prob, over_odds, under_odds,
                over_kelly, over_edge, under_kelly, under_edge,
            )

        except ValueError as e:
            messagebox.showerror("Input or data error", str(e))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _render_report(self, qb_name, opponent, opp_note, opp_def_epa, features, method,
                        adj_mean, reg_mean, mean, std, sims, line,
                        over_prob, under_prob, over_odds, under_odds,
                        over_kelly, over_edge, under_kelly, under_edge):

        lines = []
        lines.append(f"QB: {qb_name}   vs   {opponent}{opp_note}")
        lines.append(f"Games used: {features['games_played']}  |  Season avg yards/gm: "
                      f"{features['avg_yards']:.1f}  |  Opp def pass EPA/play: {opp_def_epa:+.3f}")
        lines.append("")
        lines.append(f"Method used: {method}")
        lines.append(f"  Adjustment-model mean: {adj_mean:.1f}" +
                      (f"   |   Global regression mean: {reg_mean:.1f}" if reg_mean is not None else ""))
        lines.append(f"  FINAL mean (after weather): {mean:.1f}   |   std: {std:.1f}")
        lines.append("")
        lines.append(f"Median projection: {float(np.median(sims)):.1f} yards")
        lines.append(f"Line: {line}")
        lines.append(f"P(Over):  {over_prob*100:.1f}%   (book implied: "
                      f"{implied_probability_from_american_odds(over_odds)*100:.1f}%)")
        lines.append(f"P(Under): {under_prob*100:.1f}%   (book implied: "
                      f"{implied_probability_from_american_odds(under_odds)*100:.1f}%)")
        lines.append("")
        lines.append("---- Edge & Kelly ----")
        if over_kelly > 0:
            lines.append(f"OVER  @ {over_odds:+.0f}:  edge {over_edge*100:+.1f} pts  ->  "
                          f"stake {over_kelly*100:.2f}% of bankroll")
        else:
            lines.append(f"OVER  @ {over_odds:+.0f}:  edge {over_edge*100:+.1f} pts  ->  NO BET")
        if under_kelly > 0:
            lines.append(f"UNDER @ {under_odds:+.0f}:  edge {under_edge*100:+.1f} pts  ->  "
                          f"stake {under_kelly*100:.2f}% of bankroll")
        else:
            lines.append(f"UNDER @ {under_odds:+.0f}:  edge {under_edge*100:+.1f} pts  ->  NO BET")

        lines.append("")
        if over_kelly <= 0 and under_kelly <= 0:
            lines.append(">>> BEST PLAY: No bet -- no edge found on either side.")
        elif over_kelly >= under_kelly:
            lines.append(f">>> BEST PLAY: OVER {line} @ {over_odds:+.0f}  "
                          f"-- stake {over_kelly*100:.2f}% of bankroll")
        else:
            lines.append(f">>> BEST PLAY: UNDER {line} @ {under_odds:+.0f}  "
                          f"-- stake {under_kelly*100:.2f}% of bankroll")

        lines.append("")
        lines.append("Reminder: model opinion, not a guaranteed edge. Sportsbook lines")
        lines.append("already price in most public information.")

        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, "\n".join(lines))


if __name__ == "__main__":
    QBGuiApp().mainloop()
