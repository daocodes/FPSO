import numpy as np
import pandas as pd

from algorithm.portfolio_model import evaluate_fitness
from algorithm.utils import generate_naive_equal_weight


class Particle:
    """State container for one candidate portfolio (firefly)."""

    def __init__(self, num_assets):
        # Start from a random valid long-only portfolio that sums to 1.
        self.weights = np.random.dirichlet(np.ones(num_assets))
        # Velocity is the "movement direction" used by PSO-style updates.
        self.velocity = np.zeros(num_assets, dtype=float)
        # Current objective score.
        self.fitness = -np.inf
        # Firefly brightness derived from fitness each iteration.
        self.brightness = 0.0
        # Personal best memory for PSO cognitive term.
        self.best_weights = self.weights.copy()
        self.best_fitness = -np.inf


class FPSO:
    """Paper-aligned FPSO optimizer for portfolio weights."""

    def __init__(
        self, num_particles, num_assets, max_iter, beta_0, gamma, alpha, delta
    ):
        # Core optimization dimensions/hyperparameters.
        self.num_particles = int(num_particles)
        self.num_assets = int(num_assets)
        self.max_iter = int(max_iter)
        self.beta_0 = float(beta_0)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.delta = float(delta)

        # Runtime optimization state.
        self.swarm = []
        self.global_best_weights = None
        self.global_best_fitness = -np.inf
        self.history = []
        self._current_iter = 0

        # These defaults can be overridden after initialization.
        self.lambda_v = 1.0
        self.lambda_t = 0.01
        self.rho_1 = 100.0
        self.rho_2 = 100.0
        self.rho_3 = 100.0
        self.K = self.num_assets
        self.u = 1.0

    def generate_initial_swarm(self):
        # Create the full population of candidate portfolios.
        self.swarm = [Particle(self.num_assets) for _ in range(self.num_particles)]

    def update_swarm_brightness(self, phi_min, epsilon):
        # Shift fitness values so brightness is nonnegative and nonzero.
        # Paper uses a small positive random epsilon.
        for particle in self.swarm:
            eps_i = np.random.uniform(1e-12, max(float(epsilon), 1e-12))
            particle.brightness = max(particle.fitness - phi_min, 0.0) + eps_i

    def identify_brighter_peers(self, target_particle):
        # Firefly logic: only pay attention to particles brighter than target.
        return [
            particle
            for particle in self.swarm
            if particle is not target_particle
            and particle.brightness > target_particle.brightness
        ]

    def calculate_weighted_centroid(self, target_particle, brighter_peers):
        # If no better peers exist, keep moving relative to self.
        if not brighter_peers:
            return target_particle.weights.copy()

        # Paper-aligned centroid weighting:
        # weight_ij = beta_0 * exp(-gamma * r_ij^2) * I_j
        # a_i = sum_j(weight_ij * x_j) / (sum_j(weight_ij) + delta)
        weighted_sum = np.zeros(self.num_assets, dtype=float)
        weight_total = 0.0

        for peer in brighter_peers:
            r_ij = float(np.linalg.norm(target_particle.weights - peer.weights))
            weight_ij = self.beta_0 * np.exp(-self.gamma * (r_ij**2)) * peer.brightness
            weighted_sum += weight_ij * peer.weights
            weight_total += weight_ij

        denom = weight_total + max(float(self.delta), 1e-12)
        if denom <= 0:
            return target_particle.weights.copy()
        return weighted_sum / denom

    def apply_hybrid_update(self, target_particle, centroid):
        # Paper-aligned update:
        # x_{t+1} = x_t + beta * (a_i - x_t) + alpha * epsilon_i
        distance = float(np.linalg.norm(target_particle.weights - centroid))
        beta = self.beta_0 * np.exp(-self.gamma * (distance**2))

        attraction_step = beta * (centroid - target_particle.weights)
        random_walk = self.alpha * (np.random.rand(self.num_assets) - 0.5)

        raw_weights = target_particle.weights + attraction_step + random_walk
        # Repair constraints so portfolio remains feasible.
        new_weights = self.repair_portfolio_constraints(raw_weights)

        # Store velocity as realized position change after repair.
        target_particle.velocity = new_weights - target_particle.weights
        target_particle.weights = new_weights

    def repair_portfolio_constraints(self, raw_weights):
        # Convert to flat float vector and validate dimension.
        w = np.asarray(raw_weights, dtype=float).reshape(-1)
        if w.shape[0] != self.num_assets:
            raise ValueError("Raw weight vector has wrong length.")

        # Clean numerically bad values then enforce long-only lower bound.
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.maximum(w, 0.0)

        # Cardinality projection: keep only top-K positions.
        k = int(np.clip(self.K, 1, self.num_assets))
        if k < self.num_assets:
            keep_idx = np.argpartition(w, -k)[-k:]
            mask = np.zeros(self.num_assets, dtype=bool)
            mask[keep_idx] = True
            w = np.where(mask, w, 0.0)

        # Normalize/validate upper bounds (scalar or per-asset vector).
        upper = np.asarray(self.u, dtype=float)
        if upper.ndim == 0:
            upper = np.full(self.num_assets, float(upper))
        else:
            upper = upper.reshape(-1)
        if upper.shape[0] != self.num_assets:
            raise ValueError("Upper bounds u must be a scalar or match num_assets.")

        upper = np.maximum(upper, 0.0)
        if np.sum(upper) <= 0:
            raise ValueError("Upper bounds are infeasible; all upper limits are zero.")

        if np.sum(upper) < 1.0:
            # If full investment is impossible under upper bounds, normalize to max feasible sum.
            projected = upper.copy()
            return projected / np.sum(projected)

        # If all weights are zero after clipping, restart from equal weights.
        if np.sum(w) <= 0:
            w = np.ones(self.num_assets, dtype=float)

        # Project onto simplex with upper bounds.
        p = w / np.sum(w)
        projected = np.zeros(self.num_assets, dtype=float)
        active = np.ones(self.num_assets, dtype=bool)
        remaining = 1.0

        # Iteratively cap overweight assets and redistribute leftover mass.
        while np.any(active):
            active_idx = np.where(active)[0]
            if active_idx.size == 0:
                break
            scaled = remaining * p[active_idx] / np.sum(p[active_idx])
            over = scaled > upper[active_idx]

            # If nothing violates upper bounds, finalize and stop.
            if not np.any(over):
                projected[active_idx] = scaled
                remaining = 0.0
                break

            # Clamp violated assets to upper bound and continue redistribution.
            hit_idx = active_idx[over]
            projected[hit_idx] = upper[hit_idx]
            active[hit_idx] = False
            remaining = 1.0 - np.sum(projected)
            if remaining <= 1e-12:
                break

        # Final safety normalization.
        projected = np.maximum(projected, 0.0)
        total = np.sum(projected)
        if total <= 0:
            return np.ones(self.num_assets, dtype=float) / self.num_assets
        return projected / total

    def optimize(self, expected_returns, covariance, weights_prev):
        # Lazily initialize swarm if user did not call generator manually.
        if not self.swarm:
            self.generate_initial_swarm()

        # Reset run-level global state.
        self.global_best_weights = None
        self.global_best_fitness = -np.inf
        self.history = []

        # ----- Initial evaluation pass -----
        for particle in self.swarm:
            # Ensure each candidate is feasible before scoring.
            particle.weights = self.repair_portfolio_constraints(particle.weights)
            # Evaluate paper objective + penalties.
            particle.fitness = evaluate_fitness(
                particle.weights,
                expected_returns,
                covariance,
                weights_prev,
                self.lambda_v,
                self.lambda_t,
                self.rho_1,
                self.rho_2,
                self.rho_3,
                self.K,
                self.u,
            )
            # Initialize personal-best memory.
            particle.best_fitness = particle.fitness
            particle.best_weights = particle.weights.copy()

            # Track initial global best.
            if particle.fitness > self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_weights = particle.weights.copy()

        # ----- Main iterative loop -----
        for iteration in range(self.max_iter):
            self._current_iter = iteration
            # Normalize brightness relative to the current worst particle.
            phi_min = min(particle.fitness for particle in self.swarm)
            self.update_swarm_brightness(phi_min=phi_min, epsilon=1e-6)

            # Move each particle once according to hybrid update rule.
            for particle in self.swarm:
                peers = self.identify_brighter_peers(particle)
                centroid = self.calculate_weighted_centroid(particle, peers)
                self.apply_hybrid_update(particle, centroid)

            # Re-score all particles after movement.
            for particle in self.swarm:
                particle.fitness = evaluate_fitness(
                    particle.weights,
                    expected_returns,
                    covariance,
                    weights_prev,
                    self.lambda_v,
                    self.lambda_t,
                    self.rho_1,
                    self.rho_2,
                    self.rho_3,
                    self.K,
                    self.u,
                )

                # Update personal-best memory.
                if particle.fitness > particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_weights = particle.weights.copy()

                # Update global-best memory.
                if particle.fitness > self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_weights = particle.weights.copy()

            # Record optimization progress for diagnostics/plotting.
            self.history.append(self.global_best_fitness)

        # Return best found portfolio and objective trajectory.
        return self.global_best_weights, np.array(self.history, dtype=float)


def firefly(data, epsilon=0.01, timesteps=100, asset_cap=30):
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["date", "permno"]).dropna(subset=["ret"])

    dates = pd.Index(frame["date"].drop_duplicates().sort_values())
    if len(dates) < 2:
        return np.array([], dtype=float)

    portfolio_returns: list[float] = []
    weights_prev = None

    for end_index in range(timesteps, len(dates), timesteps):
        train_dates = dates[max(0, end_index - timesteps) : end_index]
        hold_dates = dates[end_index : end_index + timesteps]
        if len(train_dates) == 0 or len(hold_dates) == 0:
            break

        universe = frame.loc[frame["date"] == train_dates[-1], "permno"].dropna().unique()
        if len(universe) == 0:
            continue

        chosen_assets = np.asarray(universe[: min(asset_cap, len(universe))])
        train_window = frame.loc[
            frame["date"].isin(train_dates) & frame["permno"].isin(chosen_assets),
            ["date", "permno", "ret"],
        ]
        hold_window = frame.loc[
            frame["date"].isin(hold_dates) & frame["permno"].isin(chosen_assets),
            ["date", "permno", "ret"],
        ]
        if train_window.empty or hold_window.empty:
            continue

        train_matrix = (
            train_window.pivot_table(index="date", columns="permno", values="ret", aggfunc="last")
            .reindex(columns=chosen_assets)
            .sort_index()
            .ffill()
            .fillna(0.0)
        )
        hold_matrix = (
            hold_window.pivot_table(index="date", columns="permno", values="ret", aggfunc="last")
            .reindex(index=hold_dates, columns=chosen_assets)
            .sort_index()
            .ffill()
            .fillna(0.0)
        )

        expected_returns = train_matrix.mean().to_numpy(dtype=float) * 252.0
        covariance = train_matrix.cov().fillna(0.0).to_numpy(dtype=float) * 252.0
        optimizer = FPSO(
            num_particles=30,
            num_assets=len(chosen_assets),
            max_iter=max(20, int(50 * max(float(epsilon), 1e-3))),
            beta_0=1.0,
            gamma=1.0,
            alpha=max(float(epsilon), 1e-3),
            delta=1e-6,
        )
        optimizer.K = len(chosen_assets)
        optimizer.u = 1.0
        if weights_prev is None or len(weights_prev) != len(chosen_assets):
            weights_prev = generate_naive_equal_weight(len(chosen_assets))

        best_weights, _ = optimizer.optimize(expected_returns, covariance, weights_prev)
        weights_prev = best_weights
        period_returns = hold_matrix.to_numpy(dtype=float) @ best_weights
        portfolio_returns.extend(period_returns.tolist())

    return np.asarray(portfolio_returns, dtype=float)
