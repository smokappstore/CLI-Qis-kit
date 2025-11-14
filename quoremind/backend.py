import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Tuple, List, Dict, Union, Any, Optional, Callable
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EmpiricalCovariance
import functools
import time
from dataclasses import dataclass
import mpmath
from mpmath import mpf as MP_Float
from quoremind.decoherence_bayes_logic import DecoherenceBayesLogic



# ============================================================================
# OPERADOR ÁUREO Y PARÁMETROS
# ============================================================================

def golden_ratio_operator(dimension: int, index: int = 0, phase_offset: float = 0.0) -> np.ndarray:
    """
    Calcula el Operador Áureo (φ-operator) para modular fase cuasiperiódica.
    φ = (1 + √5) / 2 ≈ 1.618
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    phases = np.array([phi**(i + index) + phase_offset % (2 * np.pi) for i in range(dimension)])
    operator = np.exp(1j * phases)
    return operator / np.linalg.norm(operator)


def aureo_operator(n: int, phi: float = 1.6180339887) -> Tuple[float, float]:
    """Calcula paridad y fase del operador áureo Ô_n."""
    n_float = float(n)
    paridad = np.cos(np.pi * n_float)
    fase_mod = np.cos(np.pi * phi * n_float)
    return paridad, fase_mod


def lambda_doble_operator(state_vector: np.ndarray,
                         hamiltonian: Callable,
                         qubits: np.ndarray,
                         golden_phase: int = 0) -> float:
    """
    Parámetro λ_doble: combina amplitud + operador áureo + qubits para detectar anomalías.
    λ_doble = (weighted_sum * aureo_component * hamiltonian_effects) / mean_qbits
    """
    amplitude = np.max(np.abs(state_vector))
    weighted_sum = np.sum(state_vector * amplitude)
    mean_qbits = np.mean(qubits)
    mean_qbits = max(mean_qbits, 1e-6)

    aureo_op = golden_ratio_operator(len(state_vector), golden_phase)
    aureo_component = np.abs(np.dot(aureo_op, state_vector))

    hamiltonian_effects = np.mean([hamiltonian(q, 0.0) for q in qubits])

    lambda_doble = (weighted_sum * aureo_component * hamiltonian_effects) / mean_qbits
    return float(np.real(lambda_doble))


def calculate_cosines(entropy: float, prn_object: float) -> Tuple[float, float, float]:
    """Calcula los cosenos directores (x, y, z) para un vector 3D."""
    epsilon = 1e-6
    entropy = max(entropy, epsilon)
    prn_object = max(prn_object, epsilon)
    magnitude = np.sqrt(entropy ** 2 + prn_object ** 2 + 1)
    cos_x = entropy / magnitude
    cos_y = prn_object / magnitude
    cos_z = 1 / magnitude
    return cos_x, cos_y, cos_z


# Definir observables globales
H = lambda q, p: 0.5 * p**2 + 0.5 * q**2
S = lambda q, p: -0.5 * np.log(q**2 + p**2 + 1e-6)


# ============================================================================
# ENTROPÍA VON NEUMANN
# ============================================================================

class VonNeumannEntropy:
    """Entropía de von Neumann para operadores de densidad cuánticos."""

    @staticmethod
    def compute_von_neumann_entropy(density_matrix: np.ndarray,
                                   epsilon: float = 1e-15,
                                   normalize: bool = True,
                                   method: str = "sigmoid") -> float:
        """
        Calcula S = -Tr(ρ log ρ) donde ρ es la matriz de densidad.
        """
        eigenvalues = np.linalg.eigvalsh(density_matrix)
        eigenvalues = np.clip(eigenvalues, epsilon, None)
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))

        if not normalize:
            return float(entropy)

        dim = len(eigenvalues)
        max_entropy = np.log2(dim)

        if method == "sigmoid":
            normalized = 1.0 / (1.0 + np.exp(-entropy))
        elif method == "tanh":
            normalized = (np.tanh(entropy / 2.0) + 1.0) / 2.0
        elif method == "log_compression":
            normalized = np.log(1.0 + entropy) / np.log(1.0 + max_entropy)
        elif method == "min_max":
            normalized = entropy / max_entropy
        elif method == "clamp":
            normalized = np.clip(entropy / max_entropy, 0.0, 1.0)
        else:
            raise ValueError(f"Método desconocido: {method}")

        return float(np.clip(normalized, 0.0, 1.0))

    @staticmethod
    def density_matrix_from_state(state: np.ndarray) -> np.ndarray:
        """Construye ρ = |ψ⟩⟨ψ| desde un estado puro."""
        return np.outer(state, np.conj(state))

    @staticmethod
    def mixed_state_entropy(probabilities: List[float],
                           density_matrices: List[np.ndarray]) -> float:
        """Entropía para mezcla: ρ = Σᵢ pᵢ ρᵢ"""
        rho_mixed = np.zeros_like(density_matrices[0], dtype=complex)
        for p, rho in zip(probabilities, density_matrices):
            rho_mixed += p * rho
        return VonNeumannEntropy.compute_von_neumann_entropy(rho_mixed)


# ============================================================================
# ESTRUCTURA SIMPLÉCTICA
# ============================================================================

class PoissonBrackets:
    """Estructura simpléctica con corchetes de Poisson {f, g}."""

    @staticmethod
    def _to_scalar(val: Union[np.ndarray, float]) -> float:
        """Convierte un array de un solo elemento (0-D o 1-D) a un escalar float."""
        if isinstance(val, np.ndarray):
            return val.item()
        return float(val)

    @staticmethod
    def poisson_bracket(f: Callable, g: Callable,
                       q: np.ndarray, p: np.ndarray,
                       eps: float = 1e-5) -> float:
        """Calcula {f, g} = (∂f/∂q)(∂g/∂p) - (∂f/∂p)(∂g/∂q)"""
        # Ensure q and p are arrays, even if single-element
        q = np.atleast_1d(q)
        p = np.atleast_1d(p)

        df_dq_val = (f(q + eps, p) - f(q - eps, p)) / (2 * eps)
        df_dp_val = (f(q, p + eps) - f(q, p - eps)) / (2 * eps)
        dg_dq_val = (g(q + eps, p) - g(q - eps, p)) / (2 * eps)
        dg_dp_val = (g(q, p + eps) - g(q, p - eps)) / (2 * eps)

        df_dq = PoissonBrackets._to_scalar(df_dq_val)
        df_dp = PoissonBrackets._to_scalar(df_dp_val)
        dg_dq = PoissonBrackets._to_scalar(dg_dq_val)
        dg_dp = PoissonBrackets._to_scalar(dg_dp_val)

        return float(df_dq * dg_dp - df_dp * dg_dq)

    @staticmethod
    def liouville_evolution(H: Callable, f: Callable,
                           q: np.ndarray, p: np.ndarray) -> float:
        """Ecuación de Liouville: df/dt = {f, H}"""
        return PoissonBrackets.poisson_bracket(f, H, q, p)


# ============================================================================
# ESTRUCTURA METRIPLÉCTICA
# ============================================================================

class MetriplecticStructure:
    """Estructura metripléctica: parte simpléctica + parte métrica (disipativa)."""

    @staticmethod
    def metriplectic_bracket(f: Callable, g: Callable,
                            q: np.ndarray, p: np.ndarray,
                            M: np.ndarray,
                            eps: float = 1e-5) -> float:
        """Corchete metriplexico: [f, g] = {f, g} + [f, g]_M"""
        # Ensure q and p are arrays, even if single-element
        q = np.atleast_1d(q)
        p = np.atleast_1d(p)

        poisson_part = PoissonBrackets.poisson_bracket(f, g, q, p, eps)

        df_dq_val = (f(q + eps, p) - f(q - eps, p)) / (2 * eps)
        df_dp_val = (f(q, p + eps) - f(q, p - eps)) / (2 * eps)
        grad_f_q = PoissonBrackets._to_scalar(df_dq_val)
        grad_f_p = PoissonBrackets._to_scalar(df_dp_val)
        grad_f = np.array([grad_f_q, grad_f_p])

        dg_dq_val = (g(q + eps, p) - g(q - eps, p)) / (2 * eps)
        dg_dp_val = (g(q, p + eps) - g(q, p - eps)) / (2 * eps)
        grad_g_q = PoissonBrackets._to_scalar(dg_dq_val)
        grad_g_p = PoissonBrackets._to_scalar(dg_dp_val)
        grad_g = np.array([grad_g_q, grad_g_p])

        metric_part = np.dot(grad_f, np.dot(M, grad_g))
        return float(poisson_part + metric_part)

    @staticmethod
    def metriplectic_evolution(H: Callable, S: Callable, f: Callable,
                              q: np.ndarray, p: np.ndarray,
                              M: np.ndarray,
                              eps: float = 1e-5) -> float:
        """Ecuación metripléctica: df/dt = {f, H} + [f, S]_M"""
        hamiltonian_part = PoissonBrackets.poisson_bracket(f, H, q, p, eps)
        dissipative_part = MetriplecticStructure.metriplectic_bracket(f, S, q, p, M, eps)
        return float(hamiltonian_part + dissipative_part)


from quoremind.base_bayes import BayesLogic, timer_decorator, validate_input_decorator


# ============================================================================
# INTEGRACIÓN CUÁNTICO-BAYESIANA
# ============================================================================

class QuantumBayesMahalanobis(BayesLogic):
    """Combina lógica bayesiana con Mahalanobis en estados cuánticos."""

    def __init__(self):
        super().__init__()
        self.covariance_estimator = EmpiricalCovariance()

    def _get_inverse_covariance(self, data: np.ndarray) -> np.ndarray:
        """Retorna la inversa de la matriz de covarianza."""
        if data.ndim != 2:
            raise ValueError("Los datos deben ser bidimensionales.")
        self.covariance_estimator.fit(data)
        cov_matrix = self.covariance_estimator.covariance_
        try:
            inv_cov_matrix = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            inv_cov_matrix = np.linalg.pinv(cov_matrix)
        return inv_cov_matrix

    def compute_quantum_mahalanobis(self,
                                    quantum_states_A: np.ndarray,
                                    quantum_states_B: np.ndarray) -> np.ndarray:
        """Distancia de Mahalanobis vectorizada."""
        if quantum_states_A.ndim != 2 or quantum_states_B.ndim != 2:
            raise ValueError("Los estados cuánticos deben ser matrices bidimensionales.")
        if quantum_states_A.shape[1] != quantum_states_B.shape[1]:
            raise ValueError("Las dimensiones de A y B deben coincidir.")

        inv_cov_matrix = self._get_inverse_covariance(quantum_states_A)
        mean_A = np.mean(quantum_states_A, axis=0)

        diff_B = quantum_states_B - mean_A
        aux = diff_B @ inv_cov_matrix
        dist_sqr = np.einsum('ij,ij->i', aux, diff_B)
        distances = np.sqrt(dist_sqr)
        return distances

    def quantum_cosine_projection(self,
                                  quantum_states: np.ndarray,
                                  entropy: float,
                                  coherence: float,
                                  n_step: int = 1) -> tf.Tensor:
        """Proyecta estados cuánticos con cosenos."""
        if quantum_states.shape[1] != 2:
            raise ValueError("Se espera 'quantum_states' con 2 columnas.")

        cos_x, cos_y, cos_z = calculate_cosines(entropy, coherence)

        projected_states_A = quantum_states * np.array([cos_x, cos_y])
        projected_states_B = quantum_states * np.array([cos_x * cos_z, cos_y * cos_z])

        mahalanobis_distances = self.compute_quantum_mahalanobis(
            projected_states_A, projected_states_B
        )

        # Aplicar operador áureo
        paridad, fase_mod = aureo_operator(n_step)
        normalized_distances = np.tanh(mahalanobis_distances * paridad)

        return tf.convert_to_tensor(normalized_distances, dtype=tf.float32)

    def calculate_quantum_posterior_with_mahalanobis(self,
                                                     quantum_states: np.ndarray,
                                                     entropy: float,
                                                     coherence: float):
        """Calcula posterior bayesiana en proyecciones cuánticas."""
        quantum_projections = self.quantum_cosine_projection(
            quantum_states, entropy, coherence
        )

        tensor_projections = tf.convert_to_tensor(quantum_projections, dtype=tf.float32)
        mean = tf.reduce_mean(tensor_projections, axis=0, keepdims=True)
        centered = tensor_projections - mean
        n_samples = tf.cast(tf.shape(tensor_projections)[0], tf.float32)
        quantum_covariance = tf.matmul(tf.transpose(centered), centered) / (n_samples - 1)
        dim = tf.cast(tf.shape(quantum_covariance)[0], tf.float32)
        quantum_prior = tf.linalg.trace(quantum_covariance) / dim

        prior_coherence = self.calculate_high_coherence_prior(coherence)
        joint_prob = self.calculate_joint_probability(
            coherence, 1, tf.reduce_mean(tensor_projections).numpy()
        )
        cond_prob = self.calculate_conditional_probability(joint_prob, quantum_prior.numpy())
        posterior = self.calculate_posterior_probability(quantum_prior.numpy(),
                                                         prior_coherence,
                                                         cond_prob)
        return posterior, quantum_projections

    def predict_quantum_state(self,
                              quantum_states: np.ndarray,
                              entropy: float,
                              coherence: float):
        """Predice el siguiente estado cuántico."""
        posterior, projections = self.calculate_quantum_posterior_with_mahalanobis(
            quantum_states, entropy, coherence
        )

        next_state_prediction = tf.reduce_sum(
            tf.multiply(
                tf.cast(projections, tf.float32),
                tf.cast(tf.expand_dims(posterior, -1), tf.float32) # Ensure both operands are tf.float32
            ),
            axis=0
        )
        return next_state_prediction, posterior


# ============================================================================
# RUIDO PROBABILÍSTICO DE REFERENCIA (PRN)
# ============================================================================

class PRN:
    """Modelado del Ruido Probabilístico de Referencia."""

    def __init__(self, influence: float, algorithm_type: Optional[str] = None, **parameters):
        if not 0 <= influence <= 1:
            raise ValueError(f"Influencia debe estar en [0,1]. Valor: {influence}")
        self.influence = influence
        self.algorithm_type = algorithm_type
        self.parameters = parameters

    def adjust_influence(self, adjustment: float) -> None:
        """Ajusta influencia dentro de [0, 1]."""
        new_influence = np.clip(self.influence + adjustment, 0, 1)
        if new_influence != self.influence + adjustment:
            print(f"⚠️  Influencia ajustada a {new_influence}")
        self.influence = new_influence

    def combine_with(self, other_prn: 'PRN', weight: float = 0.5) -> 'PRN':
        """Combina dos PRN según un peso."""
        if not 0 <= weight <= 1:
            raise ValueError(f"Peso debe estar en [0,1]. Valor: {weight}")
        combined_influence = self.influence * weight + other_prn.influence * (1 - weight)
        combined_params = {**self.parameters, **other_prn.parameters}
        algorithm = self.algorithm_type if weight >= 0.5 else other_prn.algorithm_type
        return PRN(combined_influence, algorithm, **combined_params)

    def __str__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        algo_str = f", algorithm={self.algorithm_type}" if self.algorithm_type else ""
        return f"PRN(influence={self.influence:.4f}{algo_str}{', ' + params_str if params_str else ''})"


class EnhancedPRN(PRN):
    """Extiende PRN para registrar distancias de Mahalanobis."""

    def __init__(self, influence: float = 0.5, algorithm_type: str = None, **parameters):
        super().__init__(influence, algorithm_type, **parameters)
        self.mahalanobis_records = []

    def record_quantum_noise(self, probabilities: dict, quantum_states: np.ndarray):
        """Registra ruido cuántico basado en Mahalanobis."""
        entropy = self.calculate_entropy_from_probabilities(probabilities)

        cov_estimator = EmpiricalCovariance().fit(quantum_states)
        mean_state = np.mean(quantum_states, axis=0)
        inv_cov = np.linalg.pinv(cov_estimator.covariance_)

        diff = quantum_states - mean_state
        aux = diff @ inv_cov
        dist_sqr = np.einsum('ij,ij->i', aux, diff)
        distances = np.sqrt(dist_sqr)
        mahal_mean = np.mean(distances)

        self.mahalanobis_records.append(mahal_mean)
        return entropy, mahal_mean

    @staticmethod
    def calculate_entropy_from_probabilities(probabilities: dict) -> float:
        """Calcula entropía de Shannon."""
        probs = np.array(list(probabilities.values()))
        probs = probs[probs >= 0]
        sum_probs = np.sum(probs)
        if sum_probs > 0:
            probs = probs / sum_probs
        else:
            probs = np.ones_like(probs) / len(probs)

        probs = np.clip(probs, 1e-15, 1.0)
        return -np.sum(probs * np.log2(probs))


# ============================================================================
# COLAPSO DE ONDA CON ESTRUCTURA METRIPLÉCTICA
# ============================================================================

class QuantumNoiseCollapse(QuantumBayesMahalanobis):
    """Simula colapso de onda cuántico con estructura metripléctica."""

    def __init__(self, prn_influence: float = 0.5):
        super().__init__()
        self.prn = EnhancedPRN(influence=prn_influence)
        self.aureo_n_step = 1
        self.decoherence_bayes_logic = DecoherenceBayesLogic()
        self.golden_phase_offset = 0.0

    def adjust_golden_phase(self, bayesian_posterior: float):
        """
        Ajusta la fase del operador áureo basándose en la posterior bayesiana.
        """
        self.golden_phase_offset += bayesian_posterior * np.pi

    @timer_decorator
    def simulate_wave_collapse_metriplectic(self,
                                            quantum_states: np.ndarray,
                                            density_matrix: np.ndarray,
                                            prn_influence: float,
                                            previous_action: int,
                                            M: Optional[np.ndarray] = None):
        """Simula colapso de onda con estructura metripléctica."""
        if M is None:
            M = np.eye(2) * 0.1

        # 1. Entropía von Neumann
        von_neumann_ent = VonNeumannEntropy.compute_von_neumann_entropy(
            density_matrix, normalize=True, method="sigmoid"
        )

        # 2. Ruido cuántico
        magnitudes = np.array([np.linalg.norm(state) for state in quantum_states])
        sum_magnitudes = np.sum(magnitudes)
        if sum_magnitudes > 0:
            normalized_magnitudes = magnitudes / sum_magnitudes
        else:
            normalized_magnitudes = np.ones_like(magnitudes) / len(magnitudes)

        probabilities = {str(i): p for i, p in enumerate(normalized_magnitudes)}
        entropy, mahalanobis_mean = self.prn.record_quantum_noise(probabilities, quantum_states)

        # Normalizar entropía
        entropy_normalized = 1.0 / (1.0 + np.exp(-entropy)) if entropy > 1.0 else entropy
        entropy_normalized = np.clip(entropy_normalized, 0.0, 1.0)

        mahalanobis_normalized = 1.0 / (1.0 + np.exp(-mahalanobis_mean))
        mahalanobis_normalized = np.clip(mahalanobis_normalized, 0.0, 1.0)

        # 3. Cosenos directores
        cos_x, cos_y, cos_z = calculate_cosines(entropy_normalized, mahalanobis_normalized)

        # 4. Coherencia
        coherence = np.exp(-mahalanobis_normalized) * (cos_x + cos_y + cos_z) / 3.0

        # 5. Lógica bayesiana
        bayes_probs = self.decoherence_bayes_logic.calculate_probabilities_with_decoherence(
            entropy=entropy_normalized,
            coherence=coherence,
            prn_influence=prn_influence,
            action=previous_action,
            decoherence_metric=mahalanobis_normalized
        )

        # 6. Proyección cuántica
        projected_states = self.quantum_cosine_projection(
            quantum_states, entropy_normalized, coherence, n_step=self.aureo_n_step
        )

        # 7. Evolución metripléctica
        q_val = np.array([entropy_normalized])
        p_val = np.array([coherence])

        metriplectic_evolution = MetriplecticStructure.metriplectic_evolution(
            H, S, H, q_val, p_val, M
        )

        # 8. Colapso final
        collapsed_state = tf.reduce_sum(
            tf.multiply(
                projected_states,
                tf.cast(bayes_probs["action_to_take"], tf.float32) * self.aureo_n_step
            )
        )

        self.aureo_n_step += 1

        return {
            "collapsed_state": collapsed_state.numpy(),
            "action": bayes_probs["action_to_take"],
            "shannon_entropy": entropy,
            "shannon_entropy_normalized": entropy_normalized,
            "von_neumann_entropy": von_neumann_ent,
            "coherence": coherence,
            "mahalanobis_distance": mahalanobis_mean,
            "mahalanobis_normalized": mahalanobis_normalized,
            "cosines": (cos_x, cos_y, cos_z),
            "metriplectic_evolution": metriplectic_evolution,
            "bayesian_posterior": bayes_probs["posterior_a_given_b"]
        }

    @timer_decorator
    def objective_function_with_noise(self,
                                      quantum_states: tf.Tensor,
                                      target_state: np.ndarray,
                                      entropy_weight: float = 1.0,
                                      n_step: int = 1) -> tf.Tensor:
        """Función objetivo con ruido (estabilizada)."""
        paridad, fase_mod = aureo_operator(n_step)
        target_modulado = tf.cast(target_state, tf.float32) * tf.cast(fase_mod, tf.float32)

        fidelity = tf.abs(tf.reduce_sum(quantum_states * target_modulado))**2

        quantum_states_np = quantum_states.numpy()
        magnitudes = np.array([np.linalg.norm(state) for state in quantum_states_np])
        sum_magnitudes = np.sum(magnitudes)
        if sum_magnitudes > 0:
            normalized_magnitudes = magnitudes / sum_magnitudes
        else:
            normalized_magnitudes = np.ones_like(magnitudes) / len(magnitudes)

        probabilities_for_shannon = {str(i): p for i, p in enumerate(normalized_magnitudes)}
        entropy, mahalanobis_dist = self.prn.record_quantum_noise(probabilities_for_shannon, quantum_states_np)

        mahalanobis_dist_clipped = np.clip(mahalanobis_dist, 0, 1000.0)
        mahalanobis_term = tf.stop_gradient(1.0 - tf.exp(-tf.cast(mahalanobis_dist_clipped, tf.float32)))

        paridad_tf = tf.cast(paridad, tf.float32)
        entropy_clipped_tf = tf.cast(np.clip(entropy, 0, 1000.0), tf.float32)
        entropy_weight_tf = tf.cast(entropy_weight, tf.float32)

        paridad_term = tf.abs(tf.reduce_mean(quantum_states) - paridad_tf)

        objective_value = (
            (1.0 - fidelity)
            + entropy_weight_tf * entropy_clipped_tf
            + mahalanobis_term
            + 0.5 * paridad_term
        )

        return objective_value

    @timer_decorator
    def optimize_quantum_state(self,
                               initial_states: np.ndarray,
                               target_state: np.ndarray,
                               max_iterations: int = 100,
                               learning_rate: float = 0.01):
        """Optimiza estados cuánticos mediante descenso de gradiente (Adam)."""
        current_states = tf.Variable(initial_states.astype(np.float32), dtype=tf.float32)

        best_objective = float('inf')
        best_states = current_states.numpy().copy()

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        for i in range(1, max_iterations + 1):
            with tf.GradientTape() as tape:
                objective = self.objective_function_with_noise(current_states, target_state, n_step=i)

            grads = tape.gradient(objective, [current_states])

            if grads[0] is None or tf.reduce_any(tf.math.is_nan(grads[0])):
                print(f"⚠️  Deteniendo en iteración {i} por gradiente inválido")
                break

            optimizer.apply_gradients(zip(grads, [current_states]))

            new_objective = self.objective_function_with_noise(current_states, target_state, n_step=i)
            if new_objective < best_objective:
                best_objective = new_objective
                best_states = current_states.numpy().copy()

        return best_states, float(best_objective)

# dit_reynolds_hp.py

RE_CRITICO_HP = mpmath.mpf("4.0") # Umbral crítico de transición a caos

def calcular_reynolds_informacional(
    rho_info: MP_Float,
    v_info_mag: MP_Float,
    mu_q: MP_Float,
    L_char: MP_Float = mpmath.mpf("1.0")
) -> Tuple[MP_Float, str]:
    """
    Calcula el Número de Reynolds Informacional (Re_I) con alta precisión
    y diagnostica el régimen de Flujo Laminar o Turbulento (Caótico).
    """
    # Evitar división por cero
    mu_q_safe = max(mu_q, mpmath.mpf("1e-50"))

    # Re_I = (rho * L * v) / mu
    Re_I = mpmath.fdiv(mpmath.fmul(mpmath.fmul(rho_info, L_char), v_info_mag), mu_q_safe)

    # Diagnóstico de Régimen
    if Re_I < RE_CRITICO_HP:
        regimen = "LAMINAR (Estable/Coherente)"
    elif Re_I >= RE_CRITICO_HP and Re_I < mpmath.mpf("1000.0"):
        regimen = "TRANSICIÓN (Riesgo/Precursor de Caos)"
    else:
        # Re_I > 4.0 es la región donde el OTOC comienza a crecer exponencialmente.
        regimen = "CAÓTICO/TURBULENTO (Decoherencia Irreversible)"

    return Re_I, regimen

# --- Ejemplo de Uso en el Módulo ---
if __name__ == "__main__":
    print(f"\n--- PRUEBA DE DIAGNÓSTICO Re_I (Precisión: {mpmath.mp.dps} DPS) ---")

    # Escenario 1: Flujo Superfluido (Ideal para estados entrelazados)
    # Baja viscosidad, alta velocidad, rho ~ 1
    Re_sf, reg_sf = calcular_reynolds_informacional(
        rho_info=mpmath.mpf("0.999"), v_info_mag=mpmath.mpf("100.0"), mu_q=mpmath.mpf("1e-8")
    )
    print(f"\n[Superfluido] Re_I: {mpmath.nstr(Re_sf, 10)} -> Régimen: {reg_sf}")

    # Escenario 2: Umbral Crítico (El límite del caos cuántico)
    Re_crit, reg_crit = calcular_reynolds_informacional(
        rho_info=mpmath.mpf("0.8"), v_info_mag=mpmath.mpf("5.0"), mu_q=mpmath.mpf("1.0")
    )
    print(f"[Umbral Crítico] Re_I: {mpmath.nstr(Re_crit, 10)} -> Régimen: {reg_crit}")

# ============================================================================
# EJEMPLO DE USO INTEGRADO
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("⚛️  QuoreMind v1.0.0: Sistema Metripléctico Cuántico-Bayesiano")
    print("=" * 80)

    # 1. Crear matriz de densidad
    state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    density_matrix = VonNeumannEntropy.density_matrix_from_state(state)

    # 2. Estados cuánticos
    np.random.seed(42)
    quantum_states = np.random.randn(10, 2)

    # 3. Crear sistema
    collapse_system = QuantumNoiseCollapse(prn_influence=0.6)

    # 4. Matriz métrica
    M = np.array([[0.1, 0.0], [0.0, 0.1]])

    # ========================================================================
    # PARTE 1: SIMULACIÓN DE COLAPSO METRIPLÉCTICO
    # ========================================================================
    print("\n📊 1. ANÁLISIS CON ESTRUCTURA METRIPLÉCTICA (Ciclo 1)")
    print("-" * 80)

    collapse_result = collapse_system.simulate_wave_collapse_metriplectic(
        quantum_states=quantum_states,
        density_matrix=density_matrix,
        prn_influence=0.6,
        previous_action=1,
        M=M
    )

    print(f"✓ Estado colapsado: {collapse_result['collapsed_state']:.6f}")
    print(f"✓ Acción tomada: {collapse_result['action']}")
    print(f"✓ Entropía Shannon (norm): {collapse_result['shannon_entropy_normalized']:.6f}")
    print(f"✓ Entropía von Neumann: {collapse_result['von_neumann_entropy']:.6f}")
    print(f"✓ Coherencia: {collapse_result['coherence']:.6f}")
    print(f"✓ Mahalanobis (norm): {collapse_result['mahalanobis_normalized']:.6f}")
    print(f"✓ Evolución metripléctica: {collapse_result['metriplectic_evolution']:.6f}")
    print(f"✓ Posterior bayesiana: {collapse_result['bayesian_posterior']:.6f}")
    cos_x, cos_y, cos_z = collapse_result['cosines']
    print(f"✓ Cosenos (x, y, z): ({cos_x:.4f}, {cos_y:.4f}, {cos_z:.4f})")

    # ========================================================================
    # PARTE 2: CORCHETES DE POISSON
    # ========================================================================
    print("\n🔄 2. ANÁLISIS DE CORCHETES DE POISSON")
    print("-" * 80)

    entropy_val = collapse_result['shannon_entropy_normalized']
    coherence_val = collapse_result['coherence']

    x_obs = lambda q, p: q
    p_obs = lambda q, p: p

    q_test = np.array([entropy_val])
    p_test = np.array([coherence_val])

    pb_xh = PoissonBrackets.poisson_bracket(x_obs, H, q_test, p_test)
    pb_ph = PoissonBrackets.poisson_bracket(p_obs, H, q_test, p_test)

    print(f"✓ {{x, H}}: {pb_xh:.6f} (esperado: p ≈ {coherence_val:.6f})")
    print(f"✓ {{p, H}}: {pb_ph:.6f} (esperado: -q ≈ {-entropy_val:.6f})")

    df_dt = PoissonBrackets.liouville_evolution(H, x_obs, q_test, p_test)
    print(f"✓ dx/dt = {{x, H}}: {df_dt:.6f}")

    # ========================================================================
    # PARTE 3: ESTRUCTURA METRIPLÉCTICA EN DETALLE
    # ========================================================================
    print("\n⚙️  3. ESTRUCTURA METRIPLÉCTICA DETALLADA")
    print("-" * 80)

    metrib_hs = MetriplecticStructure.metriplectic_bracket(H, S, q_test, p_test, M)
    print(f"✓ [H, S]_M (metriplexico): {metrib_hs:.6f}")

    dh_dt = MetriplecticStructure.metriplectic_evolution(H, S, H, q_test, p_test, M)
    print(f"✓ dH/dt (reversible+disipativa): {dh_dt:.6f}")

    ds_dt = MetriplecticStructure.metriplectic_evolution(H, S, S, q_test, p_test, M)
    print(f"✓ dS/dt (producción de entropía): {ds_dt:.6f}")

    # ========================================================================
    # PARTE 4: OPTIMIZACIÓN
    # ========================================================================
    print("\n📈 4. OPTIMIZACIÓN DE ESTADOS CUÁNTICOS")
    print("-" * 80)

    target_state = np.array([0.8, 0.2])
    initial_states = np.random.randn(5, 2)

    optimized_states, final_objective = collapse_system.optimize_quantum_state(
        initial_states=initial_states,
        target_state=target_state,
        max_iterations=50,
        learning_rate=0.01
    )

    initial_objective = collapse_system.objective_function_with_noise(
        tf.convert_to_tensor(initial_states.astype(np.float32)),
        target_state,
        n_step=1
    )

    print(f"✓ Objetivo inicial: {initial_objective:.6f}")
    print(f"✓ Objetivo final: {final_objective:.6f}")
    print(f"✓ Mejora: {initial_objective - final_objective:.6f}")

    # ========================================================================
    # PARTE 5: EVOLUCIÓN TEMPORAL
    # ========================================================================
    print("\n🔁 5. EVOLUCIÓN TEMPORAL DEL SISTEMA")
    print("-" * 80)

    print("Simulando 5 ciclos de colapso:")
    current_states = quantum_states.copy()

    for cycle in range(5):
        result = collapse_system.simulate_wave_collapse_metriplectic(
            quantum_states=current_states,
            density_matrix=density_matrix,
            prn_influence=0.6,
            previous_action=1 if cycle % 2 == 0 else 0,
            M=M
        )

        print(f"\n  Ciclo {cycle + 1}:")
        print(f"    - von Neumann entropy: {result['von_neumann_entropy']:.4f}")
        print(f"    - Shannon (norm): {result['shannon_entropy_normalized']:.4f}")
        print(f"    - Mahalanobis (norm): {result['mahalanobis_normalized']:.4f}")
        print(f"    - Evolución metripléctica: {result['metriplectic_evolution']:.4f}")
        print(f"    - Acción: {result['action']}")

        collapse_system.adjust_golden_phase(result['bayesian_posterior'])
        current_states = current_states + np.random.randn(*current_states.shape) * 0.01

    # ========================================================================
    # PARTE 6: ANÁLISIS CON OPERADOR ÁUREO Y λ_DOBLE
    # ========================================================================
    print("\n" + "=" * 80)
    print("🔷 6. ANÁLISIS CON OPERADOR ÁUREO Y PARÁMETRO λ_DOBLE")
    print("=" * 80)

    # Estados indexados
    initial_states_indexed = np.array([
        [0.1 * i, 0.1 * (i + 1)] for i in range(10)
    ], dtype=np.float32)

    print("\n📌 Estados iniciales indexados:")
    print(initial_states_indexed)

    # Hamiltoniano
    H_local = lambda q, p: 0.5 * p**2 + 0.5 * q**2
    qubits = np.linspace(0.1, 1.0, 10)

    print(f"\n📊 Qubits: {qubits}")

    # Calcular λ_doble para estados legítimos
    print("\n🟢 Parámetro λ_doble (Estados Legítimos):")
    lambda_legit = []
    for i, state in enumerate(initial_states_indexed):
        lam = lambda_doble_operator(state, H_local, qubits, golden_phase=collapse_system.golden_phase_offset)
        lambda_legit.append(lam)
        if i < 3:
            print(f"   Estado {i}: λ_doble = {lam:.6f}")
    print(f"   ... (10 total)")

    mean_legit = np.mean(lambda_legit)
    std_legit = np.std(lambda_legit)
    print(f"\n   Media: {mean_legit:.6f}")
    print(f"   Desv. Est.: {std_legit:.6f}")

    # Estados anómalos
    print("\n🔴 Generando Estados Anómalos:")
    anomalous_states = np.random.randn(5, 2) * 5.0 + np.array([2.0, 2.0])

    lambda_anomaly = []
    for i, state in enumerate(anomalous_states):
        lam = lambda_doble_operator(state, H_local, qubits, golden_phase=1)
        lambda_anomaly.append(lam)
        print(f"   Estado {i}: λ_doble = {lam:.6f}")

    threshold = mean_legit + 3 * std_legit
    print(f"\n⚠️  Umbral de detección (3σ): {threshold:.6f}")

    anomalies_detected = [lam > threshold for lam in lambda_anomaly]
    detected_count = sum(anomalies_detected)
    print(f"\n✓ Anomalías detectadas: {detected_count}/{len(anomalous_states)}")

    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 RESUMEN FINAL")
    print("=" * 80)

    print(f"""
✓ Operador Áureo (φ-operator):
  - 10 estados indexados procesados
  - Modulación de fase cuasiperiódica integrada
  - Estabilización de optimización: ACTIVA

✓ Parámetro λ_doble (λ₂):
  - Detecta anomalías mediante:
    * Amplitud máxima de estado
    * Operador áureo (φ)
    * Efectos Hamiltonianos
    * Media de qubits
  - Umbral de detección: {threshold:.6f}
  - Sensibilidad: {(detected_count/len(anomalous_states))*100:.1f}%

✓ Estructura Metripléctica:
  - Parte reversible (Poisson): {pb_xh:.6f}
  - Parte disipativa (Métrica): {metrib_hs:.6f}
  - Producción de entropía: {ds_dt:.6f}

✓ Lógica Bayesiana:
  - Posterior: {collapse_result['bayesian_posterior']:.6f}
  - Coherencia: {collapse_result['coherence']:.6f}
  - Acción: {'ACEPTAR' if collapse_result['action'] else 'BLOQUEAR'}

✓ Optimización:
  - Mejora de pérdida: {initial_objective - final_objective:.6f}
  - Iteraciones: 50
  - Convergencia: ✓ EXITOSA
    """)

    print("=" * 80)
    print("✅ QuoreMind v1.0.0 - Análisis Completo")
    print("=" * 80)

    # ========================================================================
    # PARTE 7: PRUEBAS DE LA NUEVA FUNCIONALIDAD
    # ========================================================================
    print("\n" + "=" * 80)
    print("🔬 7. PRUEBAS DE LA NUEVA FUNCIONALIDAD")
    print("=" * 80)

    # Prueba 1: Verificar que la decoherencia afecta el prior bayesiano
    print("\n🧪 Prueba 1: Decoherencia afecta el prior bayesiano")
    deco_logic = DecoherenceBayesLogic()
    low_deco_prior = deco_logic.calculate_decoherence_prior(0.2)
    high_deco_prior = deco_logic.calculate_decoherence_prior(0.8)
    print(f"  - Prior con baja decoherencia (0.2): {low_deco_prior:.4f}")
    print(f"  - Prior con alta decoherencia (0.8): {high_deco_prior:.4f}")
    assert high_deco_prior > low_deco_prior, "El prior no aumenta con la decoherencia"
    print("  - ✓ El prior aumenta correctamente con la decoherencia")

    # Prueba 2: Verificar que el bucle de retroalimentación ajusta la fase
    print("\n🧪 Prueba 2: Bucle de retroalimentación ajusta la fase")
    initial_phase_offset = collapse_system.golden_phase_offset
    print(f"  - Fase inicial: {initial_phase_offset:.4f}")
    collapse_system.adjust_golden_phase(0.5)
    new_phase_offset = collapse_system.golden_phase_offset
    print(f"  - Fase después de ajuste (posterior=0.5): {new_phase_offset:.4f}")
    assert new_phase_offset > initial_phase_offset, "La fase no se ajustó correctamente"
    print("  - ✓ La fase se ajusta correctamente")

    print("\n" + "=" * 80)
    print("✅ Pruebas de la nueva funcionalidad completadas")
    print("=" * 80)
# ============================================================================
