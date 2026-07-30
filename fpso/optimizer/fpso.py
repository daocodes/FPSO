"""FPSO: firefly centroid attraction blended with PSO best-refinement.

The published description of FPSO combines "FA's diversity-preserving centroid
attraction with PSO's personal- and global-best refinement". The pre-refactor
implementation carried `velocity` and `best_weights` on each particle but never
read them, so the search was pure firefly. Both halves are implemented here and
the blend is a configurable, *non-learned* constant (`FPSOParams.pso_blend`);
setting it to 0 recovers the old pure-firefly behaviour, which the paper reports
as an ablation. Learning that blend is deliberately left as future work.

Per iteration, for each particle i:

    firefly step   d_FA = beta(r) * (a_i - x_i) + alpha * eps,   eps ~ U(-.5,.5)^n
    PSO velocity   v_i  = w * v_i + c1 * u1 (p_i - x_i) + c2 * u2 (g - x_i)
    blended move   x_i <- repair(x_i + (1 - phi) * d_FA + phi * v_i)

where a_i is the brightness-weighted centroid of the fireflies brighter than i,
beta(r) = beta_0 * exp(-gamma r^2), p_i is i's personal best and g the swarm best.
"""

from __future__ import annotations

import time

import numpy as np

from fpso.config.schema import FPSOParams
from fpso.data.moments import Moments
from fpso.optimizer.base import OptimizationResult, PortfolioOptimizer
from fpso.optimizer.constraints import ConstraintSet, SimplexBoxCardinalityRepair
from fpso.optimizer.objective import MeanVarianceTurnover, PenalizedObjective


class Particle:
    """One firefly: a candidate portfolio and the memory the PSO terms need."""

    __slots__ = ("weights", "velocity", "fitness", "brightness", "best_weights",
                 "best_fitness")

    def __init__(self, weights: np.ndarray):
        self.weights = weights
        self.velocity = np.zeros_like(weights)
        self.fitness = -np.inf
        self.brightness = 0.0
        self.best_weights = weights.copy()
        self.best_fitness = -np.inf

    def record_fitness(self, fitness: float) -> None:
        """Store the current score and update the personal-best memory."""
        self.fitness = fitness
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_weights = self.weights.copy()


class Swarm:
    """The population, its brightness bookkeeping, and the global best.

    Kept separate from :class:`FPSOOptimizer` so that the update rule reads as
    "what one particle does" and the population-level concerns (initialisation,
    brightness normalisation, best tracking) live in one place.
    """

    BRIGHTNESS_EPSILON = 1e-6
    """Floor added to shifted fitness so the worst particle still attracts weakly."""

    def __init__(self, particles: list[Particle]):
        self.particles = particles
        self.best_weights: np.ndarray | None = None
        self.best_fitness = -np.inf

    @classmethod
    def initialize(
        cls,
        n_particles: int,
        n_assets: int,
        repair: SimplexBoxCardinalityRepair,
        weights_prev: np.ndarray,
        rng: np.random.Generator,
    ) -> Swarm:
        """Seed the swarm with the incumbent portfolio plus Dirichlet draws.

        Including `weights_prev` as particle 0 warm-starts the search from the
        portfolio actually held. Without it the optimizer would have to rediscover
        a good allocation from scratch every month and the turnover term would be
        fighting an arbitrary starting point.
        """
        particles = [Particle(repair.repair(weights_prev))]
        for _ in range(n_particles - 1):
            particles.append(Particle(repair.repair(rng.dirichlet(np.ones(n_assets)))))
        return cls(particles)

    def update_brightness(self, rng: np.random.Generator) -> None:
        """Shift fitness to be non-negative so it can act as a light intensity.

        Fitness can be negative (the objective nets out risk and turnover), but
        brightness weights a convex combination and must be positive. Subtracting
        the worst fitness in the swarm and adding a small jitter achieves that
        while preserving the fitness ordering.
        """
        worst = min(particle.fitness for particle in self.particles)
        for particle in self.particles:
            jitter = rng.uniform(1e-12, self.BRIGHTNESS_EPSILON)
            particle.brightness = max(particle.fitness - worst, 0.0) + jitter

    def brighter_than(self, target: Particle) -> list[Particle]:
        """Peers that outshine `target` — the only ones it is attracted to."""
        return [p for p in self.particles if p is not target and p.brightness > target.brightness]

    def track_best(self, particle: Particle) -> None:
        """Promote `particle` to swarm best if it improves on the incumbent."""
        if particle.fitness > self.best_fitness:
            self.best_fitness = particle.fitness
            self.best_weights = particle.weights.copy()


class FPSOOptimizer(PortfolioOptimizer):
    """Constraint-native hybrid firefly / particle-swarm portfolio optimizer."""

    name = "fpso"

    def __init__(self, params: FPSOParams):
        self.params = params

    def solve(
        self,
        moments: Moments,
        weights_prev: np.ndarray,
        rng: np.random.Generator,
    ) -> OptimizationResult:
        started = time.perf_counter()
        n_assets = len(moments.assets)

        constraints = ConstraintSet(
            n_assets=n_assets,
            max_assets=self.params.max_assets,
            max_weight=self.params.max_weight,
        )
        repair = SimplexBoxCardinalityRepair(constraints)
        objective = PenalizedObjective(
            base=MeanVarianceTurnover(
                mu=moments.mu,
                sigma=moments.sigma,
                weights_prev=np.asarray(weights_prev, dtype=float),
                lambda_v=self.params.lambda_v,
                lambda_t=self.params.lambda_t,
            ),
            constraints=constraints,
            params=self.params,
        )

        swarm = Swarm.initialize(
            n_particles=self.params.num_particles,
            n_assets=n_assets,
            repair=repair,
            weights_prev=weights_prev,
            rng=rng,
        )
        for particle in swarm.particles:
            particle.record_fitness(objective.score(particle.weights))
            swarm.track_best(particle)

        convergence = np.empty(self.params.max_iter, dtype=float)
        for iteration in range(self.params.max_iter):
            swarm.update_brightness(rng)
            for particle in swarm.particles:
                self._move(particle, swarm, repair, rng)
            for particle in swarm.particles:
                particle.record_fitness(objective.score(particle.weights))
                swarm.track_best(particle)
            convergence[iteration] = swarm.best_fitness

        return OptimizationResult(
            weights=swarm.best_weights,
            objective_value=swarm.best_fitness,
            convergence=convergence,
            solve_seconds=time.perf_counter() - started,
        )

    def _move(
        self,
        particle: Particle,
        swarm: Swarm,
        repair: SimplexBoxCardinalityRepair,
        rng: np.random.Generator,
    ) -> None:
        """Advance one particle by the blended firefly / PSO step."""
        firefly_step = self._firefly_step(particle, swarm, rng)
        pso_step = self._pso_step(particle, swarm, rng)

        blend = self.params.pso_blend
        proposed = particle.weights + (1.0 - blend) * firefly_step + blend * pso_step
        repaired = repair.repair(proposed)

        # Velocity is the *realised* displacement after projection, so the PSO
        # momentum term never accumulates movement the constraints rejected.
        particle.velocity = repaired - particle.weights
        particle.weights = repaired

    def _firefly_step(
        self, particle: Particle, swarm: Swarm, rng: np.random.Generator
    ) -> np.ndarray:
        """Attraction toward the brightness-weighted centroid, plus a random walk."""
        centroid = self._weighted_centroid(particle, swarm)
        distance = float(np.linalg.norm(particle.weights - centroid))
        beta = self.params.beta_0 * np.exp(-self.params.gamma * distance**2)
        random_walk = self.params.alpha * (rng.random(particle.weights.shape[0]) - 0.5)
        return beta * (centroid - particle.weights) + random_walk

    def _weighted_centroid(self, particle: Particle, swarm: Swarm) -> np.ndarray:
        """a_i = sum_j beta(r_ij) I_j x_j / (sum_j beta(r_ij) I_j + delta).

        Attracting to the centroid of all brighter peers at once, rather than
        pairwise, is what preserves swarm diversity: a particle is not dragged
        onto whichever single neighbour happens to be brightest.
        """
        brighter = swarm.brighter_than(particle)
        if not brighter:
            return particle.weights.copy()

        weighted_sum = np.zeros_like(particle.weights)
        weight_total = 0.0
        for peer in brighter:
            r = float(np.linalg.norm(particle.weights - peer.weights))
            attraction = (
                self.params.beta_0 * np.exp(-self.params.gamma * r**2) * peer.brightness
            )
            weighted_sum += attraction * peer.weights
            weight_total += attraction

        return weighted_sum / (weight_total + max(self.params.delta, 1e-12))

    def _pso_step(
        self, particle: Particle, swarm: Swarm, rng: np.random.Generator
    ) -> np.ndarray:
        """Inertia-weighted pull toward the personal and swarm bests."""
        n = particle.weights.shape[0]
        global_best = (
            swarm.best_weights if swarm.best_weights is not None else particle.weights
        )
        cognitive = (
            self.params.c_cognitive
            * rng.random(n)
            * (particle.best_weights - particle.weights)
        )
        social = self.params.c_social * rng.random(n) * (global_best - particle.weights)
        return self.params.inertia * particle.velocity + cognitive + social
