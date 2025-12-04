# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# DATA_PATH = "data/condor_real_data.csv"
# FIG_DIR = "figs"


# def ensure_fig_dir():
#     os.makedirs(FIG_DIR, exist_ok=True)


# def load_data(path=DATA_PATH):
#     """
#     Load real California condor data.

#     Expected columns:
#         year, N_total, N_wild, N_captive, releases, deaths_wild
#     """
#     df = pd.read_csv(path)
#     # 保证按年份排序
#     df = df.sort_values("year").reset_index(drop=True)
#     return df


# # ---------------------------------------------------------------------
# # 1. 时间序列可视化
# # ---------------------------------------------------------------------
# def plot_time_series(df):
#     """
#     Plot N_total, N_wild, N_captive over time.
#     """
#     ensure_fig_dir()
#     years = df["year"]

#     plt.figure(figsize=(10, 6))
#     plt.plot(years, df["N_total"], label="Total", linewidth=2)
#     plt.plot(years, df["N_wild"], label="Wild", linewidth=2)
#     plt.plot(years, df["N_captive"], label="Captive", linewidth=2)

#     plt.xlabel("Year")
#     plt.ylabel("Number of condors")
#     plt.title("California Condor Population (Real Data)")
#     plt.legend()
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     out_path = os.path.join(FIG_DIR, "real_time_series_total_wild_captive.png")
#     plt.savefig(out_path, dpi=200)
#     plt.close()
#     print(f"[SAVE] {out_path}")


# def plot_releases_vs_wild(df):
#     """
#     Plot releases vs N_wild (scatter + time annotation).
#     """
#     ensure_fig_dir()

#     plt.figure(figsize=(8, 6))
#     plt.scatter(df["N_wild"], df["releases"], s=60)

#     for _, row in df.iterrows():
#         plt.text(row["N_wild"] + 2, row["releases"] + 0.2,
#                  str(row["year"]), fontsize=8, alpha=0.7)

#     plt.xlabel("Wild population (N_wild)")
#     plt.ylabel("Annual releases")
#     plt.title("Releases vs Wild Population (Real Data)")
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     out_path = os.path.join(FIG_DIR, "real_releases_vs_wild.png")
#     plt.savefig(out_path, dpi=200)
#     plt.close()
#     print(f"[SAVE] {out_path}")


# def plot_deaths_vs_wild(df):
#     """
#     Plot deaths_wild vs N_wild (scatter).
#     """
#     ensure_fig_dir()

#     plt.figure(figsize=(8, 6))
#     plt.scatter(df["N_wild"], df["deaths_wild"], s=60)

#     for _, row in df.iterrows():
#         plt.text(row["N_wild"] + 2, row["deaths_wild"] + 0.2,
#                  str(row["year"]), fontsize=8, alpha=0.7)

#     plt.xlabel("Wild population (N_wild)")
#     plt.ylabel("Wild deaths")
#     plt.title("Wild Deaths vs Wild Population (Real Data)")
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     out_path = os.path.join(FIG_DIR, "real_deaths_vs_wild.png")
#     plt.savefig(out_path, dpi=200)
#     plt.close()
#     print(f"[SAVE] {out_path}")


# # ---------------------------------------------------------------------
# # 2. 增长率估计（年增长率 + rolling + log-linear）
# # ---------------------------------------------------------------------
# def compute_growth_rates(df):
#     """
#     Compute annual growth rates based on N_total and N_wild.

#     r_t_total  = N_total[t+1] / N_total[t]
#     r_t_wild   = N_wild[t+1] / N_wild[t]
#     """
#     df = df.copy()
#     df["r_total"] = df["N_total"].shift(-1) / df["N_total"]
#     df["r_wild"] = df["N_wild"].shift(-1) / df["N_wild"]

#     # 去掉最后一年（没有 t+1）
#     df_growth = df.iloc[:-1].reset_index(drop=True)
#     return df_growth


# def plot_growth_rates(df_growth, window=3):
#     """
#     Plot annual growth rates and rolling mean.
#     """
#     ensure_fig_dir()
#     years = df_growth["year"]

#     # 滚动平均
#     r_total = df_growth["r_total"]
#     r_wild = df_growth["r_wild"]
#     r_total_rolling = r_total.rolling(window=window, center=True).mean()
#     r_wild_rolling = r_wild.rolling(window=window, center=True).mean()

#     plt.figure(figsize=(10, 6))
#     plt.plot(years, r_total, "o-", label="Annual growth (Total)", alpha=0.6)
#     plt.plot(years, r_wild, "o-", label="Annual growth (Wild)", alpha=0.6)
#     plt.plot(years, r_total_rolling, "-", label=f"Rolling mean {window} (Total)", linewidth=2)
#     plt.plot(years, r_wild_rolling, "-", label=f"Rolling mean {window} (Wild)", linewidth=2)

#     plt.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
#     plt.xlabel("Year")
#     plt.ylabel("Growth factor r = N_{t+1} / N_t")
#     plt.title("Annual Growth Rates (Real Data)")
#     plt.legend()
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     out_path = os.path.join(FIG_DIR, "real_growth_rates.png")
#     plt.savefig(out_path, dpi=200)
#     plt.close()
#     print(f"[SAVE] {out_path}")


# def fit_log_linear_growth(df):
#     """
#     Fit a log-linear growth model:
#         log(N_total) = a + b * year
#     Then annual growth factor ~ exp(b)
#     """
#     df = df.copy()
#     # 只用 N_total，也可以改成 N_wild
#     y = np.log(df["N_total"].values)
#     t = df["year"].values.astype(float)

#     # 简单线性回归
#     A = np.vstack([t, np.ones_like(t)]).T
#     # 最小二乘
#     b, a = np.linalg.lstsq(A, y, rcond=None)[0]  # y ≈ a + b*t
#     # b 是 log-scale slope
#     r_est = np.exp(b)

#     return a, b, r_est


# def plot_log_linear_fit(df, a, b, r_est):
#     """
#     Plot log(N_total) and fitted line.
#     """
#     ensure_fig_dir()
#     t = df["year"].values.astype(float)
#     y = np.log(df["N_total"].values)
#     y_fit = a + b * t

#     plt.figure(figsize=(10, 6))
#     plt.plot(t, y, "o-", label="log(N_total)")
#     plt.plot(t, y_fit, "-", label=f"Fit: log(N) = a + b*t, r≈{r_est:.3f}")
#     plt.xlabel("Year")
#     plt.ylabel("log(N_total)")
#     plt.title("Log-linear Growth Fit (Real Data)")
#     plt.legend()
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     out_path = os.path.join(FIG_DIR, "real_log_linear_fit.png")
#     plt.savefig(out_path, dpi=200)
#     plt.close()
#     print(f"[SAVE] {out_path}")


# # ---------------------------------------------------------------------
# # 3. 粗略参数估计：生存率、繁殖率、圈养增长率
# # ---------------------------------------------------------------------
# def estimate_survival_and_fecundity(df):
#     """
#     Very rough estimates, based on available aggregated data.

#     Assumptions (必须在报告里写清楚)：
#       - 无迁徙（没有移入/移出）。
#       - 捕获放归数据 releases 只算作增加野外 N_wild。
#       - births_t ≈ (N_wild[t+1] - N_wild[t]) + deaths_wild[t+1] - releases[t+1]
#       - survival_t ≈ 1 - deaths_wild[t] / N_wild[t]

#     这是一个近似，用于给 Age-3 模型提供一个“初始猜测”。
#     """
#     df = df.sort_values("year").reset_index(drop=True)
#     df_est = df.copy()

#     # 生存率估计（基于当年死亡）
#     df_est["survival_approx"] = 1.0 - df_est["deaths_wild"] / df_est["N_wild"]

#     # 繁殖率估计：使用下一年的数据
#     births = []
#     for i in range(len(df_est) - 1):
#         Nw_t = df_est.loc[i, "N_wild"]
#         Nw_next = df_est.loc[i + 1, "N_wild"]
#         deaths_next = df_est.loc[i + 1, "deaths_wild"]
#         releases_next = df_est.loc[i + 1, "releases"]

#         # ΔN = N_{t+1} - N_t
#         delta = Nw_next - Nw_t
#         # 简化的 births 估计
#         b_approx = delta + deaths_next - releases_next
#         births.append(b_approx)

#     births.append(np.nan)  # 最后一年的 births 无法计算
#     df_est["births_approx"] = births

#     # 粗略 fecundity ≈ births / (N_adults_estimated)
#     # 这里没有年龄结构，只能用 N_wild/2 近似当作 breeding adults 数量
#     df_est["fecundity_approx"] = df_est["births_approx"] / (0.5 * df_est["N_wild"])
#     return df_est


# def estimate_captive_growth(df):
#     """
#     Estimate captive reproduction rate:

#     ΔC_t = C_{t+1} - C_t + releases_{t+1}

#     因为放飞减少了 C，但 releases 实际是来自圈养个体，
#     所以大致有：
#         C_{t+1} ≈ C_t + births_captive - deaths_captive - releases_{t+1}
#     故:
#         births_captive - deaths_captive ≈ ΔC_t + releases_{t+1}
#     我们将此视为 'net_captive_growth_t'.
#     """
#     df = df.sort_values("year").reset_index(drop=True)
#     net_growth = []
#     rate = []

#     for i in range(len(df) - 1):
#         C_t = df.loc[i, "N_captive"]
#         C_next = df.loc[i + 1, "N_captive"]
#         rel_next = df.loc[i + 1, "releases"]

#         delta = C_next - C_t
#         net = delta + rel_next   # net births - deaths （近似）
#         net_growth.append(net)
#         rate.append(net / C_t if C_t > 0 else np.nan)

#     net_growth.append(np.nan)
#     rate.append(np.nan)
#     df_est = df.copy()
#     df_est["captive_net_growth"] = net_growth
#     df_est["captive_net_rate"] = rate
#     return df_est


# def main():
#     print("=== Real Data Analysis (Phase 1) ===")
#     df = load_data()

#     print("\n[INFO] Head of data:")
#     print(df.head())

#     # 1) 时间序列可视化
#     plot_time_series(df)
#     plot_releases_vs_wild(df)
#     plot_deaths_vs_wild(df)

#     # 2) 增长率估计
#     df_growth = compute_growth_rates(df)
#     plot_growth_rates(df_growth, window=3)

#     a, b, r_est = fit_log_linear_growth(df)
#     print(f"\n[LOG-LINEAR] Fitted growth factor r ≈ {r_est:.4f}")
#     plot_log_linear_fit(df, a, b, r_est)

#     # 3) 参数近似估计
#     df_surv_fec = estimate_survival_and_fecundity(df)
#     df_captive = estimate_captive_growth(df)

#     # 合并关键信息，方便导出
#     df_params = df_surv_fec[[
#         "year", "N_total", "N_wild", "N_captive",
#         "survival_approx", "births_approx", "fecundity_approx"
#     ]].merge(
#         df_captive[["year", "captive_net_rate"]],
#         on="year",
#         how="left"
#     )

#     ensure_fig_dir()
#     out_csv = os.path.join(FIG_DIR, "real_data_parameter_estimates.csv")
#     df_params.to_csv(out_csv, index=False)
#     print(f"\n[SAVE] Parameter estimates (rough) → {out_csv}")

#     print("\n[INFO] Summary of approx parameters:")
#     print(df_params.describe().T)


# if __name__ == "__main__":
#     main()

"""
Phase 1: Real Data Analysis

可视化:
- N_total, N_wild, N_captive 随时间变化
- releases vs N_wild
- deaths_wild vs N_wild
- 年增长率 r_t = N_{t+1} / N_t
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figs", exist_ok=True)


def main():
    df = pd.read_csv("data/condor_real_data.csv")

    # 简单排序 simple sort by Year
    df = df.sort_values("Year").reset_index(drop=True)

    years = df["Year"].values
    N_total = df["Total"].values
    N_wild = df["Wild"].values
    N_captive = df["Captive"].values
    releases = df["Released_to_Wild"].values
    deaths_wild = df["Wild_Deaths"].values

    # 1) time series
    plt.figure(figsize=(10, 6))
    plt.plot(years, N_total, label="N_total")
    plt.plot(years, N_wild, label="N_wild")
    plt.plot(years, N_captive, label="N_captive")
    plt.xlabel("Year")
    plt.ylabel("Population size")
    plt.title("Real Data: Total, Wild, Captive")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figs/real_time_series_total_wild_captive.png", dpi=200)

    # 2) releases vs N_wild
    plt.figure(figsize=(6, 5))
    plt.scatter(N_wild, releases)
    plt.xlabel("N_wild")
    plt.ylabel("releases")
    plt.title("Real Data: releases vs N_wild")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figs/real_releases_vs_wild.png", dpi=200)

    # 3) deaths vs N_wild
    plt.figure(figsize=(6, 5))
    plt.scatter(N_wild, deaths_wild)
    plt.xlabel("N_wild")
    plt.ylabel("deaths_wild")
    plt.title("Real Data: deaths_wild vs N_wild")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figs/real_deaths_vs_wild.png", dpi=200)

    # 4) growth rates
    r = N_total[1:] / N_total[:-1]
    years_mid = years[1:]

    plt.figure(figsize=(8, 5))
    plt.plot(years_mid, r, marker="o")
    plt.axhline(1.0, color="gray", linestyle="--")
    plt.xlabel("Year")
    plt.ylabel("Annual growth ratio r_t")
    plt.title("Real Data: Annual Growth Rates N_total(t+1)/N_total(t)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figs/real_growth_rates.png", dpi=200)

    print("Phase 1 real-data analysis done. Figures saved in figs/")

if __name__ == "__main__":
    main()
