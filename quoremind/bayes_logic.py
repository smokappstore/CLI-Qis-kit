"""
Optimizador Bayesiano Cuántico Avanzado
Implementación integrada con metodologías bayesianas, Mahalanobis,
entropía de von Neumann, análisis de covarianzas y cosenos directores.
Basado en QuoreMind v1.0.0 — 2025
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from typing import Tuple, List, Dict, Union, Any, Optional, Callable, cast
from scipy.spatial.distance import mahalanobis
from scipy.linalg import logm, sqrtm, pinv
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from backend import (
    VonNeumannEntropy,
    QuantumNoiseCollapse,
    PoissonBrackets,
    MetriplecticStructure,
    lambda_doble_operator,
    H,
    S
)

from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =====================================================================
# ENUMS
# =====================================================================

class BayesLogic(Enum):
    CLASSIC = "classic"
    ENHANCED = "enhanced"
    QUANTUM_AWARE = "quantum_aware"
    FULLY_QUANTUM = "fully_quantum"


class OptimizationStrategy(Enum):
    BAYESIAN_ONLY = "bayesian_only"
    QUANTUM_ENHANCED = "quantum_enhanced"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"

# =====================================================================
# CONFIG
# =====================================================================

@dataclass
class OptimizerConfig:
    acquisition_function: str = "ucb"
    n_initial_points: int = 5
    n_iterations: int = 50
    alpha: float = 1e-6

    # Quantum settings
    n_qubits: int = 10
    quantum_noise_level: float = 0.01
    von_neumann_weight: float = 0.3

    # Covariance / Mahalanobis
    covariance_estimator: str = "empirical"
    mahalanobis_threshold: float = 2.0
    outlier_detection: bool = True

    # Directional
    use_directional_bias: bool = True
    directional_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    epsilon: float = 1e-8
    max_function_evaluations: int = 1000
    convergence_tolerance: float = 1e-6
    strategy: OptimizationStrategy = OptimizationStrategy.HYBRID

    adaptation_rate: float = 0.1
    memory_length: int = 10

# =====================================================================
# ANALYSIS: ENTROPÍA + COVARIANZA + MAHALANOBIS
# =====================================================================

class AdvancedStatisticalAnalysis:

    @staticmethod
    def von_neumann_entropy_robust(density_matrix: np.ndarray, base: str = "2") -> float:
        """Cálculo robusto de entropía de von Neumann."""
        try:
            # Simetrización
            if not np.allclose(density_matrix, density_matrix.conj().T, rtol=1e-10):
                density_matrix = (density_matrix + density_matrix.conj().T) / 2

            # Normalizar traza
            trace = np.trace(density_matrix).real
            if not np.isclose(trace, 1.0, rtol=1e-8):
                density_matrix /= trace

            eigenvalues = np.linalg.eigvalsh(density_matrix)
            eigenvalues = np.maximum(eigenvalues, 0)
            eigenvalues = eigenvalues[eigenvalues > 1e-12]

            if len(eigenvalues) == 0:
                return 0.0

            eigenvalues /= np.sum(eigenvalues)

            if base == "2":
                return float(-np.sum(eigenvalues * np.log2(eigenvalues)))
            else:
                return float(-np.sum(eigenvalues * np.log(eigenvalues)))

        except Exception as e:
            print(f"Error entropía von Neumann: {e}")
            return 0.0

    @staticmethod
    def compute_robust_covariance(data: np.ndarray, method: str = "ledoit_wolf"):
        if method == "ledoit_wolf":
            estimator = LedoitWolf()
        else:
            estimator = EmpiricalCovariance()

        estimator.fit(data)
        cov = estimator.covariance_

        try:
            precision = estimator.precision_
        except Exception:
            precision = cast(np.ndarray, pinv(cov))

        return cov, precision

    @staticmethod
    def mahalanobis_distance_batch(points, mean, precision):
        diff = points - mean
        return np.sqrt(np.sum(diff @ precision * diff, axis=1))

# =====================================================================
# DIRECTIONAL COSINES
# =====================================================================

class DirectionalAnalysis:

    @staticmethod
    def compute_directional_cosines(vector):
        mag = np.linalg.norm(vector)
        if mag == 0:
            return np.zeros_like(vector)
        return vector / mag

    @staticmethod
    def entropy_coherence_to_directional(entropy, coherence, prn_influence):
        theta = np.pi * entropy
        phi = 2 * np.pi * coherence
        r = prn_influence

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        return DirectionalAnalysis.compute_directional_cosines(
            np.array([x, y, z])
        )

    @staticmethod
    def apply_directional_bias(gradient, cosines, weight=0.1):
        if len(gradient) != len(cosines):
            if len(cosines) == 3 and len(gradient) > 3:
                ext = np.zeros_like(gradient)
                ext[:3] = cosines
                ext[3:] = np.mean(cosines)
                cosines = ext

        bias = weight * cosines * np.linalg.norm(gradient)
        return gradient + bias

# =====================================================================
# QUANTUM BAYESIAN OPTIMIZER
# =====================================================================

class QuantumBayesianOptimizer:

    def __init__(self, objective_function, bounds, config: Optional[OptimizerConfig] = None):
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.config = config or OptimizerConfig()
        self.dimension = len(bounds)

        self._initialize_gaussian_process()
        self._initialize_quantum_system()
        self._initialize_storage()

    # -------------------------- INIT --------------------------

    def _initialize_gaussian_process(self):
        kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=self.config.alpha)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.config.alpha,
            normalize_y=True,
            n_restarts_optimizer=5
        )

    def _initialize_quantum_system(self):
        self.n_qubits = self.config.n_qubits
        self.quantum_dim = 2 ** self.n_qubits

        self.quantum_state = np.zeros(self.quantum_dim, dtype=complex)
        self.quantum_state[0] = 1.0

        self.density_matrix = np.outer(self.quantum_state, self.quantum_state.conj())

    def _initialize_storage(self):
        self.X_observed = []
        self.y_observed = []
        self.entropy_history = deque(maxlen=self.config.memory_length)
        self.mahalanobis_history = deque(maxlen=self.config.memory_length)
        self.best_point = None
        self.best_value = np.inf
        self.iteration_count = 0

    # -------------------------- QUANTUM UPDATE --------------------------

    def _update_quantum_state(self, point, value):

        normalized_point = (point - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])

        # Entropía
        if len(self.y_observed) > 1:
            entropy = min(
                1.0,
                np.var(self.y_observed) / (np.mean(np.abs(self.y_observed)) + 1e-8)
            )
        else:
            entropy = 0.5

        # Coherencia
        if len(self.y_observed) > 2:
            improvements = np.diff(sorted(self.y_observed))
            coherence = 1.0 - min(
                1.0,
                np.std(improvements) / (np.mean(np.abs(improvements)) + 1e-8)
            )
        else:
            coherence = 0.5

        # PRN Influence
        if self.best_value != np.inf:
            prn_influence = max(
                0.0,
                min(1.0, (self.best_value - value) / (np.abs(self.best_value) + 1e-8))
            )
        else:
            prn_influence = 0.5

        eigs = np.array([
            0.5 + 0.3 * (1 - entropy),
            0.3 + 0.2 * coherence,
            0.2 * prn_influence
        ])

        if len(eigs) < self.quantum_dim:
            eigs = np.concatenate([
                eigs,
                np.zeros(self.quantum_dim - len(eigs))
            ])

        eigs /= np.sum(eigs) + 1e-12
        self.density_matrix = np.diag(eigs)

        eff_dim = min(4, self.quantum_dim)
        noise = self.config.quantum_noise_level * np.random.random((eff_dim, eff_dim))
        noise = (noise + noise.T) / 2

        noise_full = np.zeros_like(self.density_matrix)
        noise_full[:eff_dim, :eff_dim] = noise

        self.density_matrix += noise_full
        self.density_matrix /= np.trace(self.density_matrix)
        self.density_matrix = (self.density_matrix + self.density_matrix.conj().T) / 2

        vn = AdvancedStatisticalAnalysis.von_neumann_entropy_robust(self.density_matrix)
        self.entropy_history.append(vn)

    # -------------------------- ACQUISITION --------------------------

    def _normal_cdf(self, x):
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x * x / np.pi)))

    def _normal_pdf(self, x):
        return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)

    def _compute_acquisition_function(self, X):

        if len(self.X_observed) == 0:
            return np.random.random(len(X))

        mean, std = self.gp.predict(X, return_std=True)
        std = np.maximum(std, self.config.epsilon)

        # Base acquisition
        if self.config.acquisition_function == "ucb":
            acquisition = mean + 2.0 * std

        elif self.config.acquisition_function == "ei":
            imp = self.best_value - mean
            z = imp / std
            acquisition = imp * self._normal_cdf(z) + std * self._normal_pdf(z)

        else:  # PI
            imp = self.best_value - mean
            z = imp / std
            acquisition = self._normal_cdf(z)

        # Quantum enhancement
        if self.config.strategy in (
            OptimizationStrategy.QUANTUM_ENHANCED,
            OptimizationStrategy.HYBRID
        ):
            acquisition += self.config.von_neumann_weight * \
                self._compute_quantum_enhancement(X)

        # Directional bias
        if self.config.use_directional_bias:
            acquisition += 0.1 * self._compute_directional_enhancement(X)

        return acquisition

    # ---------- QUANTUM BOOST ----------
    def _compute_quantum_enhancement(self, X):

        if not self.entropy_history:
            return np.zeros(len(X))

        cur_entropy = self.entropy_history[-1]

        if len(self.X_observed) >= 2:
            X_obs = np.array(self.X_observed)

            cov, precision = AdvancedStatisticalAnalysis.compute_robust_covariance(
                X_obs, method=self.config.covariance_estimator
            )
            mean = np.mean(X_obs, axis=0)

            distances = AdvancedStatisticalAnalysis.mahalanobis_distance_batch(
                X, mean, precision
            )
        else:
            distances = np.ones(len(X))

        entropy_factor = cur_entropy / (1 + cur_entropy)
        distance_factor = 1.0 / (1 + distances)

        return entropy_factor * distance_factor

    # ---------- DIRECTIONAL BOOST ----------
    def _compute_directional_enhancement(self, X):

        if len(self.X_observed) < 2:
            return np.zeros(len(X))

        X_obs = np.array(self.X_observed)
        y_obs = np.array(self.y_observed)

        if len(X_obs) >= 3:
            recent_X = X_obs[-3:]
            recent_y = y_obs[-3:]
            grad = np.gradient(recent_y)[0] * (recent_X[-1] - recent_X[0])
        else:
            grad = np.array([y_obs[-1] - y_obs[0]])

        if len(grad) < self.dimension:
            ext = np.zeros(self.dimension)
            ext[:len(grad)] = grad
            grad = ext

        if self.entropy_history:
            entropy = self.entropy_history[-1]

            coherence = (
                float(min(
                    1.0,
                    1.0 / (1 + np.std(self.y_observed[-5:]))
                )) if len(self.y_observed) >= 5 else 1.0
            )

            prn = 0.5

            cosines = DirectionalAnalysis.entropy_coherence_to_directional(
                entropy, coherence, prn
            )

            biased = DirectionalAnalysis.apply_directional_bias(grad, cosines)

            proj = np.dot(X, biased[:self.dimension])
            proj = proj - np.min(proj)
            proj /= (np.max(proj) + 1e-8)

            return proj

        return np.zeros(len(X))

    # -------------------------- OUTLIERS --------------------------

    def _detect_outliers(self, new_point):

        if not self.config.outlier_detection or len(self.X_observed) < 3:
            return False

        X_obs = np.array(self.X_observed)

        try:
            cov, precision = AdvancedStatisticalAnalysis.compute_robust_covariance(
                X_obs, method=self.config.covariance_estimator
            )

            mean = np.mean(X_obs, axis=0)

            dist = AdvancedStatisticalAnalysis.mahalanobis_distance_batch(
                new_point.reshape(1, -1), mean, precision
            )[0]

            self.mahalanobis_history.append(dist)

            return dist > self.config.mahalanobis_threshold

        except Exception:
            return False

    # =====================================================================
    # OPTIMIZATION LOOP
    # =====================================================================

    def _generate_initial_points(self):
        n = self.config.n_initial_points
        d = self.dimension

        points = np.random.random((n, d))

        for i in range(d):
            idx = np.argsort(points[:, i])
            points[idx, i] = (np.arange(n) + 0.5) / n

        for i in range(d):
            low, high = self.bounds[i]
            points[:, i] = points[:, i] * (high - low) + low

        return points

    def optimize(self, verbose=True):

        print("Iniciando Optimización Bayesiana Cuántica...")
        print(f"Dimensiones: {self.dimension}")
        print(f"Estrategia: {self.config.strategy.value}")

        # Fase inicial
        init_pts = self._generate_initial_points()

        for p in init_pts:
            val = self.objective_function(p)
            self.X_observed.append(p)
            self.y_observed.append(val)

            if val < self.best_value:
                self.best_value = val
                self.best_point = p.copy()

            self._update_quantum_state(p, val)

            if verbose:
                print(f"Punto inicial → f = {val:.6f}")

        # Entrenamiento GP
        self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))

        # Ciclo iterativo
        for it in range(self.config.n_iterations):
            self.iteration_count = it

            candidates = np.random.uniform(
                self.bounds[:, 0],
                self.bounds[:, 1],
                (1000, self.dimension)
            )

            acq = self._compute_acquisition_function(candidates)
            idx = np.argmax(acq)
            next_point = candidates[idx]

            if self._detect_outliers(next_point):
                if verbose:
                    print(f"[Advertencia] Punto descartado por Mahalanobis")
                continue

            value = self.objective_function(next_point)
            self.X_observed.append(next_point)
            self.y_observed.append(value)

            if value < self.best_value:
                self.best_value = value
                self.best_point = next_point.copy()

            self._update_quantum_state(next_point, value)
            self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))

            if verbose:
                print(f"Iteración {it+1}/{self.config.n_iterations} → f = {value:.6f}")

        return {
            "best_point": self.best_point,
            "best_value": self.best_value,
            "iterations": self.config.n_iterations,
            "entropy_history": list(self.entropy_history),
            "mahalanobis_history": list(self.mahalanobis_history)
        }

# =====================================================================
# QUOREMIND — RUTINA CUÁNTICA DE DEMOSTRACIÓN
# =====================================================================

def run_quantum_step():
    # 1. Estado cuántico inicial
    state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    density = VonNeumannEntropy.density_matrix_from_state(state)

    quantum_states = np.array([
        [0.25, 0.80],
        [0.40, 0.60],
        [0.10, 0.90],
        [0.35, 0.55]
    ])

    entropy = 0.42
    coherence = 0.71
    prn = 0.63
    previous_action = 1

    M_matrix = np.eye(2) * 0.12

    # 2. Sistema de colapso cuántico
    collapse_sys = QuantumNoiseCollapse(prn_influence=prn)

    # 3. λ_doble
    lambda_val = lambda_doble_operator(
        state_vector=state,
        hamiltonian=H,
        qubits=np.array([0, 1]),
        golden_phase=1
    )

    # 4. Posterior con Mahalanobis
    posterior, projections = collapse_sys.calculate_quantum_posterior_with_mahalanobis(
        quantum_states, entropy, coherence
    )

    # 5. Acción Bayesiana
    results = collapse_sys.calculate_probabilities_and_select_action(
        entropy, coherence, prn, previous_action
    )

    # 6. Evolución metripléctica / Liouville
    liouville_val = PoissonBrackets.liouville_evolution(
        H,
        lambda q, p: q**2 + p**2,
        np.array([0.2]),
        np.array([0.3])
    )

    metriplectic_energy = MetriplecticStructure.metriplectic_evolution(
        H,
        S,
        lambda q, p: q * p,
        np.array([0.2]),
        np.array([0.3]),
        M_matrix
    )

    return {
        "λ_doble": lambda_val,
        "posterior": float(posterior),
        "proyecciones": projections.numpy(),
        "acción_bayesiana": results["action_to_take"],
        "probabilidad_condicional": results["conditional_action_given_b"],
        "liouville": liouville_val,
        "metriplectic": metriplectic_energy
    }

# Ejecutar demo
output = run_quantum_step()
for k, v in output.items():
    print(f"{k}: {v}")
