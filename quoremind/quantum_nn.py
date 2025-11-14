"""
Integración del Optimizador Bayesiano Cuántico Avanzado en la QNN
Reemplaza Adam con metodología bayesiana, distancia de Mahalanobis, entropía de von Neumann y cosenos directores
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Any, List, Tuple, Dict, Optional, Callable, Union
from dataclasses import dataclass, field
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from scipy.spatial.distance import mahalanobis
from scipy.linalg import logm, sqrtm, inv, pinv
import sqlite3
import json
import time
from tqdm import tqdm
from collections import deque
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

@dataclass
class QNNConfig:
    """Configuración avanzada de la QNN con optimizador bayesiano cuántico"""
    # Arquitectura de circuitos
    circuit_A_layers: int = 2
    circuit_B_layers: int = 2
    entanglement_layers: int = 1

    # Tipos de puertas por capa
    gate_types_A: List[str] = field(default_factory=lambda: ['RY', 'RZ'])
    gate_types_B: List[str] = field(default_factory=lambda: ['RY', 'RZ'])

    # Patrones de entrelazamiento
    entanglement_pattern: str = 'linear' # 'linear', 'circular', 'all_to_all'

    # Optimización Bayesiana Cuántica
    optimizer: str = 'quantum_bayesian'  # Nuevo optimizador por defecto
    learning_rate: float = 0.01
    epochs: int = 20
    batch_size: int = 8

    # Parámetros bayesianos
    acquisition_function: str = 'ucb'  # 'ucb', 'ei', 'pi'
    n_initial_points: int = 5
    gp_kernel: str = 'matern'  # 'matern', 'rbf'
    alpha_noise: float = 1e-6

    # Parámetros cuánticos
    von_neumann_weight: float = 0.3
    quantum_noise_level: float = 0.01
    coherence_threshold: float = 0.7

    # Mahalanobis y covarianzas
    covariance_estimator: str = 'ledoit_wolf'  # 'empirical', 'ledoit_wolf'
    mahalanobis_threshold: float = 2.0
    outlier_detection: bool = True

    # Cosenos directores
    use_directional_bias: bool = True
    directional_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Parámetros adaptativos
    adaptation_rate: float = 0.1
    memory_length: int = 10
    convergence_tolerance: float = 1e-6

    # Regularización
    l1_reg: float = 0.0
    l2_reg: float = 0.001
    dropout_prob: float = 0.0
    param_noise: float = 0.01

    # Parámetros del optimizador clásico (fallback)
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

class DatabaseManager:
    """Gestor de base de datos mejorado para entrenamientos QNN"""

    def __init__(self, db_path: str = "qnn_bayesian_experiments.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    config TEXT NOT NULL,
                    dataset_info TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'created'
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    epoch INTEGER,
                    batch INTEGER,
                    loss REAL,
                    accuracy REAL,
                    von_neumann_entropy REAL,
                    shannon_entropy REAL,
                    mahalanobis_distance REAL,
                    coherence REAL,
                    bayesian_posterior REAL,
                    directional_bias REAL,
                    gradient_norm REAL,
                    param_norm REAL,
                    acquisition_value REAL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS bayesian_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    epoch INTEGER,
                    batch INTEGER,
                    parameter_type TEXT,
                    old_value REAL,
                    new_value REAL,
                    gradient REAL,
                    bayesian_weight REAL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            ''')
            conn.commit()

    def save_experiment(self, name: str, config: QNNConfig, dataset_info: Dict) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            config_json = json.dumps(config.__dict__, default=str)

            # Convert numpy integers to Python integers for JSON serialization
            dataset_info_serializable = {k: int(v) if isinstance(v, np.integer) else v for k, v in dataset_info.items()}

            # Ensure values within target_distribution are also converted
            if 'target_distribution' in dataset_info_serializable and isinstance(dataset_info_serializable['target_distribution'], dict):
                 dataset_info_serializable['target_distribution'] = {k: int(v) if isinstance(v, np.integer) else v for k, v in dataset_info_serializable['target_distribution'].items()}
            dataset_json = json.dumps(dataset_info_serializable)

            cursor.execute(
                'INSERT INTO experiments (name, config, dataset_info, status) VALUES (?, ?, ?, ?)',
                (name, config_json, dataset_json, 'running')
            )
            experiment_id = cursor.lastrowid
            conn.commit()
            if experiment_id is None:
                raise RuntimeError("Failed to save experiment and get an experiment ID.")
            return experiment_id

    def log_training_step(self, experiment_id: int, epoch: int, batch: int, metrics: Dict):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO training_metrics
                (experiment_id, epoch, batch, loss, accuracy, von_neumann_entropy,
                 shannon_entropy, mahalanobis_distance, coherence, bayesian_posterior,
                 directional_bias, gradient_norm, param_norm, acquisition_value)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                experiment_id, epoch, batch,
                metrics.get('loss', 0), metrics.get('accuracy', 0),
                metrics.get('von_neumann_entropy', 0), metrics.get('shannon_entropy', 0),
                metrics.get('mahalanobis_distance', 0), metrics.get('coherence', 0),
                metrics.get('bayesian_posterior', 0), metrics.get('directional_bias', 0),
                metrics.get('grad_norm', 0), metrics.get('param_norm', 0),
                metrics.get('acquisition_value', 0)
            ))
            conn.commit()

    def log_bayesian_update(self, experiment_id: int, epoch: int, batch: int, updates: List[Dict]):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            update_data = []
            for update in updates:
                update_data.append((
                    experiment_id, epoch, batch,
                    update['parameter_type'], update['old_value'], update['new_value'],
                    update['gradient'], update['bayesian_weight']
                ))
            cursor.executemany('''
                INSERT INTO bayesian_updates
                (experiment_id, epoch, batch, parameter_type, old_value, new_value, gradient, bayesian_weight)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', update_data)
            conn.commit()

    def update_experiment_status(self, experiment_id: int, status: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE experiments SET status = ? WHERE id = ?', (status, experiment_id))
            conn.commit()

# --- Clase de Optimizador (Añadido RMSprop) ---

@dataclass
class QNNParams:
    """Dataclass para almacenar todos los parámetros de la QNN."""
    rotation_angles_A: np.ndarray
    rotation_angles_B: np.ndarray
    classical_weights: np.ndarray

class QuantumBayesianAnalysis:
    """Análisis estadístico cuántico y bayesiano para QNN"""

    @staticmethod
    def shannon_entropy(probabilities: Dict[str, float]) -> float:
        """Calcula la entropía de Shannon."""
        entropy = 0.0
        for p in probabilities.values():
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy

    @staticmethod
    def von_neumann_entropy_robust(density_matrix: np.ndarray) -> float:
        """Calcula la entropía de von Neumann de forma robusta."""
        try:
            if not np.allclose(density_matrix, density_matrix.conj().T, rtol=1e-10):
                density_matrix = (density_matrix + density_matrix.conj().T) / 2

            trace = np.trace(density_matrix).real
            if not np.isclose(trace, 1.0, rtol=1e-8):
                density_matrix = density_matrix / trace

            eigenvalues = np.linalg.eigvalsh(density_matrix)
            eigenvalues = np.maximum(eigenvalues, 0)
            eigenvalues = eigenvalues[eigenvalues > 1e-12]

            if len(eigenvalues) == 0:
                return 0.0

            eigenvalues = eigenvalues / np.sum(eigenvalues)
            entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
            return float(entropy)

        except Exception as e:
            print(f"Error calculando entropía de von Neumann: {e}")
            return 0.0

    @staticmethod
    def compute_robust_covariance(data: np.ndarray, method: str = 'ledoit_wolf') -> Tuple[np.ndarray, np.ndarray]:
        """Calcula la matriz de covarianza de forma robusta."""
        if method == 'ledoit_wolf':
            estimator = LedoitWolf()
        else:
            estimator = EmpiricalCovariance()

        estimator.fit(data)
        covariance = estimator.covariance_

        try:
            precision = estimator.precision_
        except (AttributeError, np.linalg.LinAlgError):
            precision = pinv(covariance)
        return covariance, estimator.precision_

    @staticmethod
    def mahalanobis_distance_batch(points: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
        """Calcula distancias de Mahalanobis para múltiples puntos."""
        diff = points - mean
        distances = np.sqrt(np.sum(diff @ precision * diff, axis=1))
        return distances

    @staticmethod
    def compute_directional_cosines(entropy: float, coherence: float, prn_influence: float) -> np.ndarray:
        """Calcula cosenos directores basados en métricas cuánticas."""
        theta = np.pi * entropy
        phi = 2 * np.pi * coherence
        r = prn_influence

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        vector = np.array([x, y, z])
        magnitude = np.linalg.norm(vector)
        if magnitude == 0:
            return np.zeros_like(vector)
        return vector / magnitude

class QuantumBayesianOptimizer:
    """Optimizador Bayesiano Cuántico para QNN"""

    def __init__(self, config: QNNConfig, param_shapes: Dict[str, Tuple]):
        self.config = config
        self.param_shapes = param_shapes

        # Inicializar Proceso Gaussiano
        if config.gp_kernel == 'matern':
            kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=config.alpha_noise)
        else:
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=config.alpha_noise)

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=config.alpha_noise,
            normalize_y=True,
            n_restarts_optimizer=3
        )

        # Historia para análisis bayesiano
        self.parameter_history = deque(maxlen=config.memory_length)
        self.loss_history = deque(maxlen=config.memory_length)
        self.entropy_history = deque(maxlen=config.memory_length)
        self.coherence_history = deque(maxlen=config.memory_length)
        self.mahalanobis_history = deque(maxlen=config.memory_length)

        # Estados internos
        self.iteration = 0
        self.gp_fitted = False
        self.current_best_loss = np.inf

    def _flatten_params(self, params: QNNParams) -> np.ndarray:
        """Aplana los parámetros para el análisis bayesiano."""
        flat_params = []
        flat_params.extend(params.rotation_angles_A.flatten())
        flat_params.extend(params.rotation_angles_B.flatten())
        flat_params.extend(params.classical_weights.flatten())
        return np.array(flat_params)

    def _unflatten_params(self, flat_params: np.ndarray, original_params: QNNParams) -> QNNParams:
        """Reconstruye los parámetros desde la representación plana."""
        idx = 0

        # Rotation angles A
        a_size = original_params.rotation_angles_A.size
        rotation_angles_A = flat_params[idx:idx+a_size].reshape(original_params.rotation_angles_A.shape)
        idx += a_size

        # Rotation angles B
        b_size = original_params.rotation_angles_B.size
        rotation_angles_B = flat_params[idx:idx+b_size].reshape(original_params.rotation_angles_B.shape)
        idx += b_size

        # Classical weights
        classical_weights = flat_params[idx:].reshape(original_params.classical_weights.shape)

        return QNNParams(rotation_angles_A, rotation_angles_B, classical_weights)

    def _compute_acquisition_function(self, candidate_params: np.ndarray) -> float:
        """Calcula la función de adquisición bayesiana."""
        if not self.gp_fitted or len(self.loss_history) < 2:
            return np.random.random()

        candidate_params = candidate_params.reshape(1, -1)
        mean, std, *_ = self.gp.predict(candidate_params, return_std=True)
        std = max(std[0], self.config.epsilon)

        if self.config.acquisition_function == 'ucb':
            kappa = 2.0
            acquisition = mean[0] + kappa * std
        elif self.config.acquisition_function == 'ei':
            improvement = self.current_best_loss - mean[0]
            z = improvement / std
            acquisition = improvement * self._normal_cdf(z) + std * self._normal_pdf(z)
        else:  # probability of improvement
            improvement = self.current_best_loss - mean[0]
            z = improvement / std
            acquisition = self._normal_cdf(z)

        # Mejoras cuánticas
        if len(self.entropy_history) > 0:
            entropy_boost = self.config.von_neumann_weight * self.entropy_history[-1]
            acquisition += entropy_boost

        return acquisition

    def _normal_cdf(self, x):
        """CDF de la distribución normal estándar."""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x * x / np.pi)))

    def _normal_pdf(self, x):
        """PDF de la distribución normal estándar."""
        return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)

    def _detect_outlier(self, new_params: np.ndarray) -> bool:
        """Detecta si los nuevos parámetros son outliers usando Mahalanobis."""
        if not self.config.outlier_detection or len(self.parameter_history) < 3:
            return False

        try:
            param_array = np.array(self.parameter_history)
            cov_matrix, precision_matrix = QuantumBayesianAnalysis.compute_robust_covariance(
                param_array, method=self.config.covariance_estimator
            )
            mean_params = np.mean(param_array, axis=0)

            distance = QuantumBayesianAnalysis.mahalanobis_distance_batch(
                new_params.reshape(1, -1), mean_params, precision_matrix
            )[0]

            self.mahalanobis_history.append(distance)
            return distance > self.config.mahalanobis_threshold

        except Exception as e:
            print(f"Error en detección de outliers: {e}")
            return False

    def update_parameters(self, params: QNNParams, gradients: QNNParams,
                         quantum_state: np.ndarray, loss: float) -> Dict:
        """Actualización de parámetros usando optimización bayesiana cuántica."""
        self.iteration += 1

        # Convertir parámetros a formato plano
        flat_params = self._flatten_params(params)
        flat_gradients = self._flatten_params(gradients)

        # Calcular métricas cuánticas
        density_matrix = np.outer(quantum_state, quantum_state.conj())
        von_neumann_entropy = QuantumBayesianAnalysis.von_neumann_entropy_robust(density_matrix)

        # Calcular entropía de Shannon desde el estado cuántico
        state_probs = np.abs(quantum_state)**2
        shannon_dict = {str(i): prob for i, prob in enumerate(state_probs) if prob > 0}
        shannon_entropy = QuantumBayesianAnalysis.shannon_entropy(shannon_dict)

        # Calcular coherencia
        coherence = np.std(state_probs)

        # Almacenar historia
        self.parameter_history.append(flat_params.copy())
        self.loss_history.append(loss)
        self.entropy_history.append(von_neumann_entropy)
        self.coherence_history.append(coherence)

        if loss < self.current_best_loss:
            self.current_best_loss = loss

        # Actualizar Proceso Gaussiano
        if len(self.parameter_history) >= self.config.n_initial_points:
            if not self.gp_fitted:
                # Primera vez: entrenar GP
                X_train = np.array(list(self.parameter_history))
                y_train = np.array(list(self.loss_history))
                self.gp.fit(X_train, y_train)
                self.gp_fitted = True
            else:
                # Actualizar GP incrementalmente
                X_new = flat_params.reshape(1, -1)
                y_new = np.array([loss])

                # Reentrenar GP con toda la historia
                X_train = np.array(list(self.parameter_history))
                y_train = np.array(list(self.loss_history))
                self.gp.fit(X_train, y_train)

        # Generar candidatos para optimización bayesiana
        if self.gp_fitted:
            best_candidate = self._find_best_candidate(flat_params)

            # Detectar outliers
            if self._detect_outlier(best_candidate):
                print(f"Candidato detectado como outlier, usando gradiente clásico")
                best_candidate = flat_params - self.config.learning_rate * flat_gradients
        else:
            # Usar gradiente clásico inicialmente
            best_candidate = flat_params - self.config.learning_rate * flat_gradients

        # Calcular cosenos directores
        prn_influence = min(1.0, max(0.0, (self.current_best_loss - loss) / (abs(self.current_best_loss) + 1e-8)))
        directional_cosines = QuantumBayesianAnalysis.compute_directional_cosines(
            von_neumann_entropy, float(coherence), prn_influence
        )

        # Aplicar bias direccional
        if self.config.use_directional_bias:
            direction_vector = best_candidate - flat_params
            direction_norm = np.linalg.norm(direction_vector)
            if float(direction_norm) > 0:
                # Extender cosenos direccionales a la dimensión correcta
                extended_cosines = np.tile(directional_cosines, (len(direction_vector) // 3) + 1)[:len(direction_vector)]
                directional_bias = 0.1 * extended_cosines * direction_norm
                best_candidate += directional_bias

        # Convertir de vuelta a estructura de parámetros
        updated_params = self._unflatten_params(best_candidate, params)

        # Aplicar regularización
        if self.config.l2_reg > 0:
            updated_params.rotation_angles_A *= (1 - self.config.learning_rate * self.config.l2_reg)
            updated_params.rotation_angles_B *= (1 - self.config.learning_rate * self.config.l2_reg)
            updated_params.classical_weights *= (1 - self.config.learning_rate * self.config.l2_reg)

        # Aplicar ruido para exploración
        if self.config.param_noise > 0:
            updated_params.rotation_angles_A += np.random.normal(0, self.config.param_noise, updated_params.rotation_angles_A.shape)
            updated_params.rotation_angles_B += np.random.normal(0, self.config.param_noise, updated_params.rotation_angles_B.shape)

        # Actualizar parámetros originales
        params.rotation_angles_A[:] = updated_params.rotation_angles_A
        params.rotation_angles_B[:] = updated_params.rotation_angles_B
        params.classical_weights[:] = updated_params.classical_weights

        # Métricas de retorno
        grad_norm = np.linalg.norm(flat_gradients)
        param_norm = np.linalg.norm(flat_params)

        metrics = {
            'grad_norm': grad_norm,
            'param_norm': param_norm,
            'von_neumann_entropy': von_neumann_entropy,
            'shannon_entropy': shannon_entropy,
            'coherence': coherence,
            'mahalanobis_distance': self.mahalanobis_history[-1] if self.mahalanobis_history else 0.0,
            'bayesian_posterior': self._compute_acquisition_function(best_candidate) if self.gp_fitted else 0.0,
            'directional_bias': np.linalg.norm(directional_cosines),
            'acquisition_value': self._compute_acquisition_function(best_candidate) if self.gp_fitted else 0.0
        }

        return metrics

    def _find_best_candidate(self, current_params: np.ndarray) -> np.ndarray:
        """Encuentra el mejor candidato usando optimización de la función de adquisición."""
        best_candidate = current_params.copy()
        best_acquisition = -np.inf

        # Búsqueda local alrededor del punto actual
        for _ in range(10):
            # Generar candidato aleatorio cerca del actual
            noise = np.random.normal(0, 0.1, current_params.shape)
            candidate = current_params + noise

            acquisition_value = self._compute_acquisition_function(candidate)

            if acquisition_value > best_acquisition:
                best_acquisition = acquisition_value
                best_candidate = candidate.copy()

        return best_candidate
class BayesConfig:
    """Configuración para el optimizador bayesiano cuántico."""
    config: Dict[str, Any] = field(default_factory=lambda: QNNConfig().__dict__)
    optimizer: str = 'quantum_bayesian'  # 'sgd', 'adam', 'rmsprop', 'quantum_bayesian'
    learning_rate: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    momentum: float = 0.9
    l1_reg: float = 0.0
    l2_reg: float = 0.001
    param_noise: float = 0.01
class QuantumOptimizer:
    """Clase de compatibilidad con optimizadores clásicos y nuevo optimizador bayesiano"""

    def __init__(self, config: QNNConfig):
        self.config = config
        self.reset_state()
        self.bayesian_optimizer: Optional[QuantumBayesianOptimizer] = None

    def reset_state(self):
        self.m, self.v = {}, {}
        self.t = 0

    def update_parameters(self, params: QNNParams, gradients: QNNParams,
                         quantum_state: Optional[np.ndarray] = None, loss: Optional[float] = None) -> Dict:
        """Actualización de parámetros con selección automática de optimizador."""

        if self.config.optimizer == 'quantum_bayesian':
            if self.bayesian_optimizer is None:
                param_shapes = {
                    'rotation_angles_A': params.rotation_angles_A.shape,
                    'rotation_angles_B': params.rotation_angles_B.shape,
                    'classical_weights': params.classical_weights.shape
                }
                self.bayesian_optimizer = QuantumBayesianOptimizer(param_shapes=param_shapes, config=self.config)
            if quantum_state is not None and loss is not None:
                return self.bayesian_optimizer.update_parameters(params, gradients, quantum_state, loss)
            else:
                print("Advertencia: quantum_state y loss requeridos para optimizador bayesiano, usando Adam")
                return self._adam_update(params, gradients)
        else:
            # Usar optimizadores clásicos
            optimizer_map = {
                'sgd': self._sgd_update,
                'adam': self._adam_update,
                'rmsprop': self._rmsprop_update,
            }
            if self.config.optimizer not in optimizer_map:
                raise ValueError(f"Optimizador no reconocido: {self.config.optimizer}")
            return optimizer_map[self.config.optimizer](params, gradients)

    def _initialize_state_for_param(self, name: str, shape: tuple):
        """Inicializa los estados del optimizador para un conjunto de parámetros."""
        if name not in self.m:
            self.m[name] = np.zeros(shape)
            self.v[name] = np.zeros(shape)

    def _adam_update(self, params: QNNParams, gradients: QNNParams) -> Dict:
        """Actualización Adam clásica."""
        self.t += 1
        grad_norm = 0
        param_norm = 0

        for name, grad_arr in gradients.__dict__.items():
            param_arr = getattr(params, name)
            self._initialize_state_for_param(name, param_arr.shape)

            self.m[name] = self.config.beta1 * self.m[name] + (1 - self.config.beta1) * grad_arr
            self.v[name] = self.config.beta2 * self.v[name] + (1 - self.config.beta2) * (grad_arr**2)

            m_hat = self.m[name] / (1 - self.config.beta1**self.t)
            v_hat = self.v[name] / (1 - self.config.beta2**self.t)

            update = self.config.learning_rate * m_hat / (np.sqrt(v_hat) + self.config.epsilon)
            update += self.config.learning_rate * self.config.l2_reg * param_arr

            param_arr -= update

            if self.config.param_noise > 0 and name.startswith('rotation'):
                 param_arr += np.random.normal(0, self.config.param_noise, param_arr.shape)

            grad_norm += np.sum(grad_arr**2)
            param_norm += np.sum(param_arr**2)

        return {'grad_norm': np.sqrt(grad_norm), 'param_norm': np.sqrt(param_norm)}

    def _sgd_update(self, params: QNNParams, gradients: QNNParams) -> Dict:
        """Actualización SGD con momentum."""
        grad_norm = 0
        param_norm = 0

        for name, grad_arr in gradients.__dict__.items():
            param_arr = getattr(params, name)
            self._initialize_state_for_param(name, param_arr.shape)

            self.m[name] = self.config.momentum * self.m[name] + grad_arr
            update = self.config.learning_rate * self.m[name]
            update += self.config.learning_rate * self.config.l2_reg * param_arr

            param_arr -= update

            if self.config.l1_reg > 0:
                setattr(params, name, np.sign(param_arr) * np.maximum(np.abs(param_arr) - self.config.l1_reg, 0))

            grad_norm += np.sum(grad_arr**2)
            param_norm += np.sum(param_arr**2)

        return {'grad_norm': np.sqrt(grad_norm), 'param_norm': np.sqrt(param_norm)}

    def _rmsprop_update(self, params: QNNParams, gradients: QNNParams) -> Dict:
        """Actualización RMSprop."""
        grad_norm = 0
        param_norm = 0

        for name, grad_arr in gradients.__dict__.items():
            param_arr = getattr(params, name)
            self._initialize_state_for_param(name, param_arr.shape)

            self.v[name] = self.config.beta2 * self.v[name] + (1 - self.config.beta2) * (grad_arr**2)

            update = self.config.learning_rate * grad_arr / (np.sqrt(self.v[name]) + self.config.epsilon)
            update += self.config.learning_rate * self.config.l2_reg * param_arr

            param_arr -= update

            grad_norm += np.sum(grad_arr**2)
            param_norm += np.sum(param_arr**2)

        return {'grad_norm': np.sqrt(grad_norm), 'param_norm': np.sqrt(param_norm)}


class AdvancedQNN:
    """QNN Avanzada con optimizador bayesiano cuántico integrado"""

    def __init__(self, config: QNNConfig, num_features: Any):
        self.config = config
        self.num_qubits_A = 5
        self.num_qubits_B = 5
        self.total_qubits = self.num_qubits_A + self.num_qubits_B

        if self.total_qubits > 14:
            print(f"Advertencia: Simular {self.total_qubits} qubits es computacionalmente muy costoso.")

        self.dim = 2 ** self.total_qubits
        self.num_features = num_features

        self.params = self._initialize_parameters()
        self.optimizer = QuantumOptimizer(config)
        self.input_scaler = StandardScaler()

        self.db = DatabaseManager()
        self.experiment_id = None

        self.training_history = {
            'loss': [], 'accuracy': [], 'entanglement_entropy': [],
            'von_neumann_entropy': [], 'shannon_entropy': [], 'coherence': [],
            'mahalanobis_distance': [], 'bayesian_posterior': [],
            'directional_bias': [], 'acquisition_value': [],
            'grad_norm': [], 'param_norm': []
        }

        self._gate_matrices = {
            'I': np.eye(2), 'X': np.array([[0, 1], [1, 0]]),
            'Y': np.array([[0, -1j], [1j, 0]]), 'Z': np.array([[1, 0], [0, -1]])
        }

    def _initialize_parameters(self) -> QNNParams:
        """Inicialización mejorada de parámetros."""
        # Inicialización Xavier adaptada
        var_A = 2.0 / (self.num_qubits_A * self.config.circuit_A_layers)
        var_B = 2.0 / (self.num_qubits_B * self.config.circuit_B_layers)

        return QNNParams(
            rotation_angles_A=np.random.normal(0, np.sqrt(var_A), (self.config.circuit_A_layers, self.num_qubits_A)),
            rotation_angles_B=np.random.normal(0, np.sqrt(var_B), (self.config.circuit_B_layers, self.num_qubits_B)),
            classical_weights=np.random.randn(self.num_qubits_A + self.num_qubits_B) * 0.1
        )

    def preprocess_data(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Preprocesamiento de datos mejorado."""
        if fit:
            self.input_scaler.fit(X)
        X_scaled = self.input_scaler.transform(X)
        min_max_scaler = MinMaxScaler(feature_range=(0,1))
        return min_max_scaler.fit_transform(X_scaled)

    def _apply_single_qubit_gate(self, estado: np.ndarray, qubit: int, gate: np.ndarray) -> np.ndarray:
        """Aplica una puerta de un solo qubit de forma eficiente."""
        op_list = [self._gate_matrices['I']] * self.total_qubits
        op_list[qubit] = gate

        full_op = op_list[0]
        for op in op_list[1:]:
            full_op = np.kron(full_op, op)

        return full_op @ estado

    def _apply_cnot(self, estado: np.ndarray, control: int, target: int) -> np.ndarray:
        """Aplica una puerta CNOT de forma eficiente."""
        nuevo_estado = estado.copy()
        for i in range(self.dim):
            if (i >> control) & 1:
                mask = 1 << target
                j = i ^ mask
                nuevo_estado[i], nuevo_estado[j] = estado[j], estado[i]
        return nuevo_estado

    def _create_circuit_layer(self, estado: np.ndarray, angles: np.ndarray, gate_types: List[str], qubit_offset: int) -> np.ndarray:
        """Crea una capa de circuito cuántico."""
        for qubit_idx, angle in enumerate(angles):
            qubit = qubit_idx + qubit_offset
            for gate_type in gate_types:
                if gate_type == 'RX':
                    gate = np.array([[np.cos(angle/2), -1j*np.sin(angle/2)], [-1j*np.sin(angle/2), np.cos(angle/2)]])
                elif gate_type == 'RY':
                    gate = np.array([[np.cos(angle/2), -np.sin(angle/2)], [np.sin(angle/2), np.cos(angle/2)]])
                elif gate_type == 'RZ':
                    gate = np.array([[np.exp(-1j*angle/2), 0], [0, np.exp(1j*angle/2)]])
                else:
                    continue
                estado = self._apply_single_qubit_gate(estado, qubit, gate)
        return estado

    def _apply_entanglement_pattern(self, estado: np.ndarray, qubits: range, pattern: str) -> np.ndarray:
        """Aplica patrones de entrelazamiento."""
        qubits_list = list(qubits)
        if pattern == 'linear' and len(qubits_list) > 1:
            for i in range(len(qubits_list) - 1):
                estado = self._apply_cnot(estado, qubits_list[i], qubits_list[i + 1])
        elif pattern == 'circular' and len(qubits_list) > 2:
            for i in range(len(qubits_list)):
                estado = self._apply_cnot(estado, qubits_list[i], qubits_list[(i + 1) % len(qubits_list)])
        return estado

    def _encode_classical_data(self, input_data: np.ndarray) -> np.ndarray:
        """Codificación de datos clásicos en estado cuántico."""
        estado = np.zeros(self.dim, dtype=np.complex128)
        estado[0] = 1.0

        for i in range(self.total_qubits):
            feature_idx = i % len(input_data)
            angle = input_data[feature_idx]
            ry_gate = np.array([[np.cos(angle/2), -np.sin(angle/2)], [np.sin(angle/2), np.cos(angle/2)]])
            estado = self._apply_single_qubit_gate(estado, i, ry_gate)
        return estado

    def forward_pass(self, input_data: np.ndarray) -> Tuple[float, np.ndarray]:
        """Forward pass mejorado con análisis cuántico."""
        estado = self._encode_classical_data(input_data)

        # Circuito A
        for layer in range(self.config.circuit_A_layers):
            estado = self._create_circuit_layer(estado, self.params.rotation_angles_A[layer], self.config.gate_types_A, 0)
            estado = self._apply_entanglement_pattern(estado, range(self.num_qubits_A), self.config.entanglement_pattern)

        # Circuito B
        for layer in range(self.config.circuit_B_layers):
            estado = self._create_circuit_layer(estado, self.params.rotation_angles_B[layer], self.config.gate_types_B, self.num_qubits_A)
            estado = self._apply_entanglement_pattern(estado, range(self.num_qubits_A, self.total_qubits), self.config.entanglement_pattern)

        # Entrelazamiento inter-circuitos
        for _ in range(self.config.entanglement_layers):
            for i in range(self.num_qubits_A):
                estado = self._apply_cnot(estado, i, i + self.num_qubits_A)

        # Medición
        measurements = np.array([self._measure_qubit(estado, i) for i in range(self.total_qubits)])

        # Combinación clásica
        output = np.tanh(np.dot(self.params.classical_weights, measurements))

        return output, estado

    def _measure_qubit(self, estado: np.ndarray, qubit: int) -> float:
        """Calcula el valor esperado del observable Z en un qubit."""
        op_z_list = [self._gate_matrices['I']] * self.total_qubits
        op_z_list[qubit] = self._gate_matrices['Z']

        observable = op_z_list[0]
        for op in op_z_list[1:]:
            observable = np.kron(observable, op)

        return np.real(estado.conj().T @ observable @ estado)

    def _compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Cálculo de pérdida (MSE)."""
        return float(np.mean((y_pred - y_true)**2))

    def parameter_shift_gradient(self, input_data: np.ndarray, y_true: float) -> QNNParams:
        """Gradiente usando parameter shift rule."""
        shift = np.pi / 2
        original_params = self.params

        params_copy = QNNParams(
            rotation_angles_A=original_params.rotation_angles_A.copy(),
            rotation_angles_B=original_params.rotation_angles_B.copy(),
            classical_weights=original_params.classical_weights.copy()
        )
        self.params = params_copy

        grads = QNNParams(
            rotation_angles_A=np.zeros_like(self.params.rotation_angles_A),
            rotation_angles_B=np.zeros_like(self.params.rotation_angles_B),
            classical_weights=np.zeros_like(self.params.classical_weights)
        )

        param_sets = [
            (grads.rotation_angles_A, self.params.rotation_angles_A),
            (grads.rotation_angles_B, self.params.rotation_angles_B),
            (grads.classical_weights, self.params.classical_weights)
        ]

        for grad_arr, param_arr in param_sets:
            it = np.nditer(param_arr, flags=['multi_index'], op_flags=[])
            while not it.finished:
                idx = it.multi_index
                original_val = param_arr[idx]

                param_arr[idx] = original_val + shift
                pred_plus, _ = self.forward_pass(input_data)

                param_arr[idx] = original_val - shift
                pred_minus, _ = self.forward_pass(input_data)

                param_arr[idx] = original_val
                y_pred_original, _ = self.forward_pass(input_data)
                grad_loss_pred = 2 * (y_pred_original - y_true)

                grad_pred_theta = (pred_plus - pred_minus) / 2.0
                grad_arr[idx] = grad_loss_pred * grad_pred_theta

                it.iternext()

        self.params = original_params
        return grads

    def fit(self, X: np.ndarray, y: np.ndarray, experiment_name: str):
        """Entrenamiento con optimizador bayesiano cuántico."""
        print(f"Iniciando entrenamiento con optimizador: {self.config.optimizer}")

        X_proc = self.preprocess_data(X, fit=True)
        y_proc = 2 * y - 1  # Convertir a [-1, 1]

        dataset_info = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'target_distribution': dict(pd.Series(y).value_counts()),
            'optimizer': self.config.optimizer
        }

        self.experiment_id = self.db.save_experiment(experiment_name, self.config, dataset_info)
        n_batches = len(X) // self.config.batch_size

        print(f"Configuración del experimento:")
        print(f"  - Muestras: {len(X)}")
        print(f"  - Características: {X.shape[1]}")
        print(f"  - Épocas: {self.config.epochs}")
        print(f"  - Tamaño de lote: {self.config.batch_size}")
        print(f"  - Optimizador: {self.config.optimizer}")

        for epoch in range(self.config.epochs):
            epoch_loss = 0
            epoch_metrics = {
                'von_neumann_entropy': [], 'shannon_entropy': [], 'coherence': [],
                'mahalanobis_distance': [], 'bayesian_posterior': [], 'directional_bias': [],
                'acquisition_value': [], 'grad_norm': [], 'param_norm': []
            }

            with tqdm(range(n_batches), desc=f"Época {epoch+1}/{self.config.epochs}") as pbar:
                for batch_idx in pbar:
                    start = batch_idx * self.config.batch_size
                    end = start + self.config.batch_size
                    X_batch, y_batch = X_proc[start:end], y_proc[start:end]

                    batch_loss = 0
                    batch_gradients = QNNParams(
                        rotation_angles_A=np.zeros_like(self.params.rotation_angles_A),
                        rotation_angles_B=np.zeros_like(self.params.rotation_angles_B),
                        classical_weights=np.zeros_like(self.params.classical_weights)
                    )

                    final_state = None

                    # Forward pass y gradientes para el lote
                    for sample_idx, (x_sample, y_sample) in enumerate(zip(X_batch, y_batch)):
                        output, state = self.forward_pass(x_sample)
                        sample_loss = (output - y_sample)**2
                        batch_loss += sample_loss

                        # Almacenar el último estado para análisis bayesiano
                        if sample_idx == len(X_batch) - 1:
                            final_state = state

                        # Calcular gradientes
                        sample_grads = self.parameter_shift_gradient(x_sample, y_sample)

                        # Acumular gradientes
                        batch_gradients.rotation_angles_A += sample_grads.rotation_angles_A / len(X_batch)
                        batch_gradients.rotation_angles_B += sample_grads.rotation_angles_B / len(X_batch)
                        batch_gradients.classical_weights += sample_grads.classical_weights / len(X_batch)

                    batch_loss /= len(X_batch)
                    epoch_loss += batch_loss

                    # Actualización de parámetros con optimizador bayesiano cuántico
                    if self.config.optimizer == 'quantum_bayesian' and final_state is not None:
                        update_metrics = self.optimizer.update_parameters(
                            self.params, batch_gradients, final_state, batch_loss
                        )
                    else:
                        update_metrics = self.optimizer.update_parameters(self.params, batch_gradients)

                    # Recopilar métricas
                    for key in epoch_metrics.keys():
                        if key in update_metrics:
                            epoch_metrics[key].append(update_metrics[key])

                    # Log de métricas del lote
                    batch_metrics = {
                        'loss': batch_loss,
                        'accuracy': 0,  # Se calculará al final de la época
                        **{k: v[-1] if v else 0 for k, v in epoch_metrics.items()}
                    }

                    self.db.log_training_step(self.experiment_id, epoch, batch_idx, batch_metrics)

                    # Actualizar barra de progreso
                    pbar.set_postfix({
                        'loss': f'{batch_loss:.4f}',
                        'vn_entropy': f'{update_metrics.get("von_neumann_entropy", 0):.4f}',
                        'coherence': f'{update_metrics.get("coherence", 0):.4f}'
                    })

            # Métricas de la época
            epoch_loss /= n_batches
            preds = self.predict(X, raw_output=True)[0]
            accuracy = self.evaluate(X, y)['accuracy']
            ent_entropy = self._compute_entanglement_entropy(X_proc[0])

            # Promediar métricas de la época
            epoch_avg_metrics = {k: np.mean(v) if v else 0 for k, v in epoch_metrics.items()}

            # Almacenar en historia
            self.training_history['loss'].append(epoch_loss)
            self.training_history['accuracy'].append(accuracy)
            self.training_history['entanglement_entropy'].append(ent_entropy)

            for key, value in epoch_avg_metrics.items():
                if key in self.training_history:
                    self.training_history[key].append(value)

            # Mostrar progreso
            progress_msg = f"Época {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}"
            if self.config.optimizer == 'quantum_bayesian':
                progress_msg += f", VN Entropy: {epoch_avg_metrics.get('von_neumann_entropy', 0):.4f}"
                progress_msg += f", Coherence: {epoch_avg_metrics.get('coherence', 0):.4f}"
                progress_msg += f", Mahalanobis: {epoch_avg_metrics.get('mahalanobis_distance', 0):.4f}"

            print(progress_msg)

        self.db.update_experiment_status(self.experiment_id, 'completed')
        print(f"Entrenamiento completado con optimizador {self.config.optimizer}")

    def predict(self, X: np.ndarray, raw_output: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predicción mejorada."""
        X_proc = self.preprocess_data(X)
        predictions = np.array([self.forward_pass(x)[0] for x in X_proc])
        if raw_output:
            return predictions, np.greater(predictions, 0).astype(int)
        return np.greater(predictions, 0).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluación del modelo."""
        _, y_pred_class = self.predict(X, raw_output=True)
        accuracy = np.mean(y_pred_class == y)
        return {'accuracy': accuracy}

    def _compute_entanglement_entropy(self, input_data: np.ndarray) -> float:
        """Calcula la entropía de entrelazamiento entre subsistemas A y B."""
        _, final_state = self.forward_pass(input_data)

        rho = np.outer(final_state, final_state.conj())
        rho_A = np.trace(rho.reshape(2**self.num_qubits_A, 2**self.num_qubits_B,
                                     2**self.num_qubits_A, 2**self.num_qubits_B), axis1=1, axis2=3)

        eigenvalues = np.linalg.eigvalsh(rho_A)
        non_zero_eigvals = eigenvalues[eigenvalues > 1e-12]
        entropy = -np.sum(non_zero_eigvals * np.log2(non_zero_eigvals))
        return float(np.real(entropy))

    def visualize_training(self):
        """Visualización mejorada para optimizador bayesiano."""
        if self.config.optimizer == 'quantum_bayesian':
            fig, axs = plt.subplots(3, 3, figsize=(18, 15))
        else:
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        fig.suptitle(f"Resultados del Entrenamiento - {self.config.optimizer.upper()} (Experimento {self.experiment_id})")

        # Métricas básicas
        axs[0, 0].plot(self.training_history['loss'], 'r-o', label='Loss', markersize=4)
        axs[0, 0].set_title('Pérdida vs. Época')
        axs[0, 0].set_xlabel('Época')
        axs[0, 0].set_ylabel('Pérdida (MSE)')
        axs[0, 0].grid(True)

        axs[0, 1].plot(self.training_history['accuracy'], 'b-o', label='Accuracy', markersize=4)
        axs[0, 1].set_title('Precisión vs. Época')
        axs[0, 1].set_xlabel('Época')
        axs[0, 1].set_ylabel('Precisión')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        axs[1, 0].plot(self.training_history['entanglement_entropy'], 'g-o', label='Entanglement', markersize=4)
        axs[1, 0].set_title('Entropía de Entrelazamiento vs. Época')
        axs[1, 0].set_xlabel('Época')
        axs[1, 0].set_ylabel('Entropía (bits)')
        axs[1, 0].grid(True)

        if self.config.optimizer == 'quantum_bayesian':
            # Métricas específicas del optimizador bayesiano
            if self.training_history['von_neumann_entropy']:
                axs[1, 1].plot(self.training_history['von_neumann_entropy'], 'm-o', label='von Neumann', markersize=4)
            if self.training_history['shannon_entropy']:
                axs[1, 1].plot(self.training_history['shannon_entropy'], 'c-o', label='Shannon', markersize=4)
            axs[1, 1].set_title('Entropías Cuánticas vs. Época')
            axs[1, 1].set_xlabel('Época')
            axs[1, 1].set_ylabel('Entropía')
            axs[1, 1].legend()
            axs[1, 1].grid(True)

            if self.training_history['coherence']:
                axs[2, 0].plot(self.training_history['coherence'], 'orange', marker='o', label='Coherencia', markersize=4)
            axs[2, 0].set_title('Coherencia Cuántica vs. Época')
            axs[2, 0].set_xlabel('Época')
            axs[2, 0].set_ylabel('Coherencia')
            axs[2, 0].grid(True)

            if self.training_history['mahalanobis_distance']:
                axs[2, 1].plot(self.training_history['mahalanobis_distance'], 'purple', marker='o', label='Mahalanobis', markersize=4)
            axs[2, 1].set_title('Distancia de Mahalanobis vs. Época')
            axs[2, 1].set_xlabel('Época')
            axs[2, 1].set_ylabel('Distancia')
            axs[2, 1].grid(True)

            if self.training_history['bayesian_posterior']:
                axs[2, 2].plot(self.training_history['bayesian_posterior'], 'brown', marker='o', label='Posterior Bayesiano', markersize=4)
            axs[2, 2].set_title('Posterior Bayesiano vs. Época')
            axs[2, 2].set_xlabel('Época')
            axs[2, 2].set_ylabel('Posterior')
            axs[2, 2].grid(True)

        else:
            # Métricas clásicas para otros optimizadores
            if self.training_history['grad_norm'] and any(x is not None for x in self.training_history['grad_norm']):
                axs[1, 1].plot([x for x in self.training_history['grad_norm'] if x is not None], 'm-o', label='Norma del Gradiente', markersize=4)
            if self.training_history['param_norm'] and any(x is not None for x in self.training_history['param_norm']):
                axs[1, 1].plot([x for x in self.training_history['param_norm'] if x is not None], 'c-o', label='Norma de Parámetros', markersize=4)
            axs[1, 1].set_title('Normas vs. Época')
            axs[1, 1].set_xlabel('Época')
            axs[1, 1].set_ylabel('Valor de la Norma')
            axs[1, 1].legend()
            axs[1, 1].grid(True)

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show()

    def compare_optimizers(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                             optimizers: List[str], epochs: int) -> Dict:
        """Compara diferentes optimizadores y devuelve los resultados."""
        results = {}

        for optimizer_name in optimizers:
            print(f"\n{'='*20} Entrenando con optimizador: {optimizer_name.upper()} {'='*20}")

            # Create a new config for this run
            temp_config = QNNConfig(**self.config.__dict__)
            temp_config.optimizer = optimizer_name
            temp_config.epochs = epochs

            qnn_instance = AdvancedQNN(config=temp_config, num_features=X_train.shape[1])

            start_time = time.time()
            qnn_instance.fit(X_train, y_train, experiment_name=f"Comparison_{optimizer_name}")
            training_time = time.time() - start_time

            train_metrics = qnn_instance.evaluate(X_train, y_train)
            test_metrics = qnn_instance.evaluate(X_test, y_test)

            results[optimizer_name] = {
                'training_time': training_time,
                'train_accuracy': train_metrics['accuracy'],
                'test_accuracy': test_metrics['accuracy'],
                'final_loss': qnn_instance.training_history['loss'][-1] if qnn_instance.training_history['loss'] else None,
                'training_history': qnn_instance.training_history
            }

        return results


# Ejemplo de uso integrado
if __name__ == "__main__":
    print("="*80)
    print("QNN CON OPTIMIZADOR BAYESIANO CUÁNTICO AVANZADO")
    print("="*80)

    # 1. Cargar y preparar datos
    X, y = load_breast_cancer(return_X_y=True)

    # Usar las primeras 10 características
    num_features_to_use = 10
    X = X[:, :num_features_to_use]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(f"Datos cargados:")
    print(f"  - Muestras de entrenamiento: {len(X_train)}")
    print(f"  - Muestras de prueba: {len(X_test)}")
    print(f"  - Características: {X_train.shape[1]}")
    print(f"  - Clases: {len(np.unique(y))}")

    # 2. Configuración para optimizador bayesiano cuántico
    qnn_config_bayesian = QNNConfig(
        circuit_A_layers=2,
        circuit_B_layers=2,
        optimizer='quantum_bayesian',  # Nuestro nuevo optimizador
        learning_rate=0.005,  # Menor para optimizador bayesiano
        epochs=15,
        batch_size=8,

        # Parámetros bayesianos
        acquisition_function='ucb',
        n_initial_points=5,
        von_neumann_weight=0.4,
        covariance_estimator='ledoit_wolf',
        use_directional_bias=True,
        outlier_detection=True,

        # Regularización
        l2_reg=0.001,
        param_noise=0.01
    )

    # 3. Crear y entrenar modelo con optimizador bayesiano
    print("\n" + "="*60)
    print("ENTRENAMIENTO CON OPTIMIZADOR BAYESIANO CUÁNTICO")
    print("="*60)

    qnn_bayesian = AdvancedQNN(config=qnn_config_bayesian, num_features=X_train.shape[1])

    start_time = time.time()
    qnn_bayesian.fit(X_train, y_train, experiment_name="BreastCancer_QuantumBayesian_Advanced")
    bayesian_time = time.time() - start_time

    print(f"\nTiempo de entrenamiento bayesiano: {bayesian_time:.2f} segundos")

    # 4. Evaluación
    print("\n" + "="*40)
    print("EVALUACIÓN DEL MODELO BAYESIANO")
    print("="*40)

    train_metrics_bayesian = qnn_bayesian.evaluate(X_train, y_train)
    test_metrics_bayesian = qnn_bayesian.evaluate(X_test, y_test)

    print(f"Precisión en entrenamiento: {train_metrics_bayesian['accuracy']:.4f}")
    print(f"Precisión en prueba: {test_metrics_bayesian['accuracy']:.4f}")
    print(f"Pérdida final: {qnn_bayesian.training_history['loss'][-1]:.6f}")

    if qnn_bayesian.training_history['von_neumann_entropy']:
        print(f"Entropía von Neumann final: {qnn_bayesian.training_history['von_neumann_entropy'][-1]:.6f}")
    if qnn_bayesian.training_history['coherence']:
        print(f"Coherencia final: {qnn_bayesian.training_history['coherence'][-1]:.6f}")
    if qnn_bayesian.training_history['mahalanobis_distance']:
        print(f"Distancia Mahalanobis promedio: {np.mean(qnn_bayesian.training_history['mahalanobis_distance']):.6f}")

    # 5. Visualización
    print("\nGenerando visualizaciones...")
    qnn_bayesian.visualize_training()

    # 6. Comparación con Adam (opcional)
    print("\n" + "="*60)
    print("COMPARACIÓN DE OPTIMIZADORES")
    print("="*60)

    comparison_results = qnn_bayesian.compare_optimizers(
        X_train, y_train, X_test, y_test,
        optimizers=['adam', 'quantum_bayesian'],
        epochs=10
    )

    # 7. Resumen final
    print("\n" + "="*80)
    print("RESUMEN DE RESULTADOS")
    print("="*80)

    for optimizer, results in comparison_results.items():
        print(f"\n{optimizer.upper()}:")
        print(f"  - Tiempo de entrenamiento: {results['training_time']:.2f}s")
        print(f"  - Precisión de entrenamiento: {results['train_accuracy']:.4f}")
        print(f"  - Precisión de prueba: {results['test_accuracy']:.4f}")
        print(f"  - Pérdida final: {results['final_loss']:.6f}")

        if optimizer == 'quantum_bayesian':
            history = results['training_history']
            if history.get('von_neumann_entropy'):
                print(f"  - Entropía von Neumann final: {history['von_neumann_entropy'][-1]:.6f}")
            if history.get('mahalanobis_distance'):
                avg_mahalanobis = np.mean([x for x in history['mahalanobis_distance'] if x > 0])
                print(f"  - Distancia Mahalanobis promedio: {avg_mahalanobis:.6f}")

    # 8. Análisis de la base de datos
    print("\n" + "="*60)
    print("ANÁLISIS DE EXPERIMENTOS EN BASE DE DATOS")
    print("="*60)

    try:
        conn = sqlite3.connect("qnn_bayesian_experiments.db")

        # Consultar experimentos
        df_experiments = pd.read_sql_query(
            "SELECT id, name, status, created_at FROM experiments ORDER BY created_at DESC LIMIT 5",
            conn
        )
        print("Últimos 5 experimentos:")
        print(df_experiments)

        # Consultar métricas del último experimento bayesiano
        if qnn_bayesian.experiment_id:
            df_metrics = pd.read_sql_query(
                f"""SELECT epoch, AVG(loss) as avg_loss, AVG(accuracy) as avg_accuracy,
                   AVG(von_neumann_entropy) as avg_vn_entropy, AVG(coherence) as avg_coherence,
                   AVG(mahalanobis_distance) as avg_mahalanobis
                   FROM training_metrics
                   WHERE experiment_id = {qnn_bayesian.experiment_id}
                   GROUP BY epoch ORDER BY epoch""",
                conn
            )
            print(f"\nMétricas por época del experimento {qnn_bayesian.experiment_id}:")
            print(df_metrics.round(6))

        conn.close()

    except Exception as e:
        print(f"Error consultando base de datos: {e}")

    print("\n" + "="*80)
    print("INTEGRACIÓN COMPLETADA EXITOSAMENTE")
    print("="*80)
    print("El optimizador bayesiano cuántico ha sido integrado correctamente en la QNN.")
    print("Características implementadas:")
    print("  ✓ Optimización bayesiana con Proceso Gaussiano")
    print("  ✓ Entropía de von Neumann para análisis cuántico")
    print("  ✓ Distancia de Mahalanobis para detección de outliers")
    print("  ✓ Matrices de covarianza robustas (Ledoit-Wolf)")
    print("  ✓ Cosenos directores para bias direccional")
    print("  ✓ Funciones de adquisición bayesiana (UCB, EI, PI)")
    print("  ✓ Base de datos mejorada con métricas cuánticas")
    print("  ✓ Visualizaciones comparativas")
    print("  ✓ Compatibilidad con optimizadores clásicos")