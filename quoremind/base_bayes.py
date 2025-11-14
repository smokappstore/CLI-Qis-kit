from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Callable
from dataclasses import dataclass
import functools
import time

def timer_decorator(func: Callable) -> Callable:
    """Decorador que mide el tiempo de ejecución."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱️  {func.__name__} ejecutado en {end_time - start_time:.4f}s")
        return result
    return wrapper


def validate_input_decorator(min_val: float = 0.0, max_val: float = 1.0) -> Callable:
    """Decorador que valida argumentos numéricos en un rango."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i, arg in enumerate(args[1:], 1):
                if isinstance(arg, (int, float)) and not (min_val <= arg <= max_val):
                    raise ValueError(f"Argumento {i} debe estar entre {min_val} y {max_val}. Valor: {arg}")
            for name, arg in kwargs.items():
                if isinstance(arg, (int, float)) and not (min_val <= arg <= max_val):
                    raise ValueError(f"Argumento {name} debe estar entre {min_val} y {max_val}. Valor: {arg}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@dataclass
class BayesLogicConfig:
    """Configuración para BayesLogic."""
    epsilon: float = 1e-6
    high_entropy_threshold: float = 0.8
    high_coherence_threshold: float = 0.6
    action_threshold: float = 0.5


class BayesLogic:
    """Lógica bayesiana para cálculo de probabilidades."""

    def __init__(self, config: Optional[BayesLogicConfig] = None):
        self.config = config or BayesLogicConfig()

    @validate_input_decorator(0.0, 1.0)
    def calculate_posterior_probability(self, prior_a: float, prior_b: float,
                                       conditional_b_given_a: float) -> float:
        """P(A|B) = (P(B|A) * P(A)) / P(B)"""
        prior_b = max(prior_b, self.config.epsilon)
        return (conditional_b_given_a * prior_a) / prior_b

    @validate_input_decorator(0.0, 1.0)
    def calculate_conditional_probability(self, joint_probability: float, prior: float) -> float:
        """P(X|Y) = P(X,Y) / P(Y)"""
        prior = max(prior, self.config.epsilon)
        return joint_probability / prior

    def calculate_high_entropy_prior(self, entropy: float) -> float:
        """Prior adaptativo según entropía."""
        return 0.3 if entropy > self.config.high_entropy_threshold else 0.1

    @validate_input_decorator(0.0, 1.0)
    def calculate_high_coherence_prior(self, coherence: float) -> float:
        """Prior adaptativo según coherencia."""
        return 0.6 if coherence > self.config.high_coherence_threshold else 0.2

    @validate_input_decorator(0.0, 1.0)
    def calculate_joint_probability(self, coherence: float, action: int,
                                   prn_influence: float) -> float:
        """Probabilidad conjunta basada en coherencia, acción e influencia PRN."""
        if coherence > self.config.high_coherence_threshold:
            if action == 1:
                return prn_influence * 0.8 + (1 - prn_influence) * 0.2
            else:
                return prn_influence * 0.1 + (1 - prn_influence) * 0.7
        return 0.3

    @timer_decorator
    def calculate_probabilities_and_select_action(
        self, entropy: float, coherence: float, prn_influence: float, action: int
    ) -> Dict[str, float]:
        """Integra cálculos bayesianos para determinar acción."""
        # Normalizar si está fuera de rango
        if entropy > 1.0:
            entropy = 1.0 / (1.0 + np.exp(-entropy))
        entropy = np.clip(entropy, 0.0, 1.0)

        if coherence > 1.0:
            coherence = 1.0 / (1.0 + np.exp(-coherence))
        coherence = np.clip(coherence, 0.0, 1.0)

        prn_influence = np.clip(prn_influence, 0.0, 1.0)

        high_entropy_prior = self.calculate_high_entropy_prior(entropy)
        high_coherence_prior = self.calculate_high_coherence_prior(coherence)

        conditional_b_given_a = (
            prn_influence * 0.7 + (1 - prn_influence) * 0.3
            if entropy > self.config.high_entropy_threshold else 0.2
        )

        posterior_a_given_b = self.calculate_posterior_probability(
            high_entropy_prior, high_coherence_prior, conditional_b_given_a
        )

        joint_probability_ab = self.calculate_joint_probability(
            coherence, action, prn_influence
        )

        conditional_action_given_b = self.calculate_conditional_probability(
            joint_probability_ab, high_coherence_prior
        )

        action_to_take = 1 if conditional_action_given_b > self.config.action_threshold else 0

        return {
            "action_to_take": action_to_take,
            "high_entropy_prior": high_entropy_prior,
            "high_coherence_prior": high_coherence_prior,
            "posterior_a_given_b": posterior_a_given_b,
            "conditional_action_given_b": conditional_action_given_b
        }
