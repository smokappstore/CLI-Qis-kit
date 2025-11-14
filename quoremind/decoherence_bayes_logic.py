import numpy as np
from typing import Optional, Dict
from quoremind.base_bayes import BayesLogic, BayesLogicConfig, timer_decorator, validate_input_decorator

class DecoherenceBayesLogic(BayesLogic):
    """
    Extiende la lógica bayesiana para que las probabilidades a priori
    sean influenciadas por una métrica de decoherencia.
    """

    def __init__(self, config: Optional[BayesLogicConfig] = None):
        super().__init__(config)

    def calculate_decoherence_prior(self, decoherence_metric: float, high_decoherence_threshold: float = 0.5) -> float:
        """
        Calcula una probabilidad a priori que aumenta con la decoherencia.

        Args:
            decoherence_metric (float): Un valor entre 0 y 1 que indica el nivel de decoherencia.
            high_decoherence_threshold (float): El umbral a partir del cual se considera alta la decoherencia.

        Returns:
            float: La probabilidad a priori.
        """
        clipped_metric = np.clip(decoherence_metric, 0.0, 1.0)
        if clipped_metric > high_decoherence_threshold:
            # A mayor decoherencia, mayor es la probabilidad a priori
            return 0.7 * clipped_metric
        else:
            return 0.3 * clipped_metric

    def _normalize_input(self, value: float) -> float:
        if value > 1.0:
            value = 1.0 / (1.0 + np.exp(-value))
        return np.clip(value, 0.0, 1.0)

    def calculate_probabilities_with_decoherence(
        self, entropy: float, coherence: float, prn_influence: float, action: int, decoherence_metric: float
    ) -> Dict[str, float]:
        """
        Integra cálculos bayesianos para determinar acción, usando la decoherencia
        para calcular el prior.
        """
        entropy = self._normalize_input(entropy)
        coherence = self._normalize_input(coherence)
        prn_influence = np.clip(prn_influence, 0.0, 1.0)

        # Usar la métrica de decoherencia para el prior en lugar de la entropía
        decoherence_prior = self.calculate_decoherence_prior(decoherence_metric)
        high_coherence_prior = self.calculate_high_coherence_prior(coherence)

        conditional_b_given_a = (
            prn_influence * 0.7 + (1 - prn_influence) * 0.3
            if entropy > self.config.high_entropy_threshold else 0.2
        )

        posterior_a_given_b = self.calculate_posterior_probability(
            decoherence_prior, high_coherence_prior, conditional_b_given_a
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
            "decoherence_prior": decoherence_prior,
            "high_coherence_prior": high_coherence_prior,
            "posterior_a_given_b": posterior_a_given_b,
            "conditional_action_given_b": conditional_action_given_b
        }
