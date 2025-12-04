import numpy as np
import yaml


class CondorEnvAge3:
    """
    Unified Condor Environment (Age-3 structured model)
    ---------------------------------------------------
    - 使用统一的参数结构 env / reward / sim / train
    - 所有 world（mechanistic、calibrated、optimistic）都通过参数文件切换
    - 环境动力学公式完全固定：survival / reproduction / lead / release / habitat / disaster
    """

    def __init__(self, **kwargs):
        """
        所有环境参数通过 kwargs 传入。
        建议从 .from_config() 或 from_dict() 来构造。
        """

        # --------------- Environment Core Parameters ---------------
        # Carrying capacity
        self.K = kwargs.get("K", 800)

        # Initial state
        self.Nj0 = kwargs.get("Nj0", 74)
        self.Ns0 = kwargs.get("Ns0", 74)
        self.Na0 = kwargs.get("Na0", 221)
        self.C0  = kwargs.get("C0", 197)
        self.H0  = kwargs.get("H0", 1.0)

        # Survival
        self.sj_base = kwargs.get("sj_base", 0.70)
        self.ss_base = kwargs.get("ss_base", 0.85)
        self.sa_base = kwargs.get("sa_base", 0.92)

        # Reproduction
        self.fecundity_per_adult = kwargs.get("fecundity_per_adult", 0.40)
        self.habitat_repro_sensitivity = kwargs.get("habitat_repro_sensitivity", 0.30)

        # Lead
        self.lead_j_base = kwargs.get("lead_j_base", 0.05)
        self.lead_s_base = kwargs.get("lead_s_base", 0.12)
        self.lead_a_base = kwargs.get("lead_a_base", 0.18)
        self.lead_mitigation_effect = kwargs.get("lead_mitigation_effect", 0.65)
        self.habitat_lead_sensitivity = kwargs.get("habitat_lead_sensitivity", 0.6)

        # Disaster
        self.disaster_base_prob = kwargs.get("disaster_base_prob", 0.03)
        self.disaster_mortality_frac = kwargs.get("disaster_mortality_frac", 0.25)
        self.disaster_habitat_sensitivity = kwargs.get("disaster_habitat_sensitivity", 1.0)

        # Release
        self.release_base_survival = kwargs.get("release_base_survival", 0.82)
        self.release_density_sensitivity = kwargs.get("release_density_sensitivity", 1.0)

        # Captive population
        self.captive_recruitment_rate = kwargs.get("captive_recruitment_rate", 0.05)
        self.C_cap = kwargs.get("C_cap", 400)

        # Habitat dynamics
        self.habitat_baseline = kwargs.get("habitat_baseline", 1.0)
        self.habitat_recovery_rate = kwargs.get("habitat_recovery_rate", 0.03)
        self.habitat_mitigation_effect = kwargs.get("habitat_mitigation_effect", 0.05)
        self.habitat_min = kwargs.get("habitat_min", 0.3)
        self.habitat_max = kwargs.get("habitat_max", 1.3)

        # ---------------- Reward parameters ----------------
        self.target_low = kwargs.get("target_low", 650)
        self.target_high = kwargs.get("target_high", 700)
        self.w_low = kwargs.get("w_low", 20.0)
        self.w_high = kwargs.get("w_high", 1.0)
        self.lambda_release = kwargs.get("lambda_release", 0.2)
        self.lambda_mitigation = kwargs.get("lambda_mitigation", 0.1)
        self.lambda_disaster = kwargs.get("lambda_disaster", 20.0)

        # RNG
        self.rng = np.random.RandomState(kwargs.get("seed", 0))

        # Initialize state
        self.reset()

    # ======================================================================
    # Unified config loader
    # ======================================================================
    @classmethod
    def from_config(cls, yaml_path):
        """
        可以加载统一结构的配置文件：
        - env: 环境动力学参数
        - reward: 奖励参数
        - sim: 仿真参数（不会传入环境内部）
        - train: 训练参数（不会传入环境内部）
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        env_cfg = cfg.get("env", {})
        reward_cfg = cfg.get("reward", {})

        merged = {**env_cfg, **reward_cfg}
        return cls(**merged)

    # ======================================================================
    # Reset
    # ======================================================================
    def reset(self):
        self.Nj = float(self.Nj0)
        self.Ns = float(self.Ns0)
        self.Na = float(self.Na0)
        self.C  = float(self.C0)
        self.H  = float(self.H0)
        return np.array([self.Nj, self.Ns, self.Na, self.C, self.H], dtype=float)

    # ======================================================================
    # Step
    # ======================================================================
    def step(self, action):
        release, mitigation = action
        release = float(max(0.0, min(self.C, round(release))))
        mitigation = float(np.clip(mitigation, 0.0, 1.0))

        Nj, Ns, Na, C, H = self.Nj, self.Ns, self.Na, self.C, self.H
        N_total = Nj + Ns + Na

        # ---------------- Disaster ----------------
        disaster_prob = self.disaster_base_prob * np.exp(
            self.disaster_habitat_sensitivity * (1.0 - H)
        )
        disaster_flag = self.rng.rand() < np.clip(disaster_prob, 0, 1)
        disaster_factor = (1 - self.disaster_mortality_frac) if disaster_flag else 1

        # ---------------- Lead survival ----------------
        def age_lead_survival(base):
            risk = base * (N_total / (N_total + 50.0))
            risk *= (1 - self.lead_mitigation_effect * mitigation)
            risk *= np.exp(self.habitat_lead_sensitivity * (1.0 - H))
            return 1 - np.clip(risk, 0, 1)

        sj = np.clip(self.sj_base * age_lead_survival(self.lead_j_base) * disaster_factor, 0, 1)
        ss = np.clip(self.ss_base * age_lead_survival(self.lead_s_base) * disaster_factor, 0, 1)
        sa = np.clip(self.sa_base * age_lead_survival(self.lead_a_base) * disaster_factor, 0, 1)

        # ---------------- Age transition ----------------
        Nj_surv, Ns_surv, Na_surv = sj * Nj, ss * Ns, sa * Na

        # Reproduction
        repro_factor = np.exp(self.habitat_repro_sensitivity * (H - 1.0))
        births = max(0, self.fecundity_per_adult * Na * repro_factor)

        Nj_next = births
        Ns_next = Nj_surv
        Na_next = Ns_surv + Na_surv

        # ---------------- Release ----------------
        rel_surv = self.release_base_survival * np.exp(
            -self.release_density_sensitivity * (N_total / (self.K + 1e-8))
        )
        rel_surv = np.clip(rel_surv * np.clip(H, 0.5, 1.5), 0, 1)

        Ns_next += rel_surv * release

        # ---------------- Captive C ----------------
        C_next = C + self.captive_recruitment_rate * C * (1 - C / self.C_cap) - release
        C_next = max(0.0, C_next)

        # ---------------- Habitat H ----------------
        H_next = H + self.habitat_recovery_rate * (self.habitat_baseline - H) \
                    + self.habitat_mitigation_effect * mitigation
        H_next = np.clip(H_next, self.habitat_min, self.habitat_max)

        # ---------------- Write state ----------------
        self.Nj, self.Ns, self.Na, self.C, self.H = Nj_next, Ns_next, Na_next, C_next, H_next
        N_total_next = Nj_next + Ns_next + Na_next

        # ---------------- Reward ----------------
        if N_total_next < self.target_low:
            penalty_pop = self.w_low * (self.target_low - N_total_next) ** 2
        elif N_total_next > self.target_high:
            penalty_pop = self.w_high * (N_total_next - self.target_high) ** 2
        else:
            penalty_pop = 0

        reward = -(penalty_pop +
                   self.lambda_release * release +
                   self.lambda_mitigation * mitigation +
                   self.lambda_disaster * (1 if disaster_flag else 0))

        return (
            np.array([Nj_next, Ns_next, Na_next, C_next, H_next], dtype=float),
            reward,
            False,
            {"N_total": N_total_next, "births": births, "disaster": disaster_flag}
        )
