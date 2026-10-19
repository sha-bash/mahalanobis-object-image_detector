"""Apply a saved training-only text projection and covariance without refitting."""
import numpy as np

from moid.semantic import SemanticVerifier, DescriptionError


class FrozenTextVerifier(SemanticVerifier):
    def __init__(self, reference, *, parameters, metric, **kwargs):
        super().__init__(reference, **kwargs)
        if metric not in {'cosine', 'mahalanobis'}:
            raise ValueError('Unknown metric')
        self.metric = metric
        self.mean = np.asarray(parameters['mean'], dtype=float)
        self.components = np.asarray(parameters['components'], dtype=float)
        self.full_covariance = np.asarray(parameters['covariance'], dtype=float)
        if np.linalg.eigvalsh(self.full_covariance).min() <= 0:
            raise ValueError('Covariance is not positive definite')
        self.reference_projected = self.project(self.reference_vector)
        self.metadata.update(metric=metric, frozen_parameters=parameters,
                             threshold_source='old car-1 validation; no held-out tuning',
                             covariance_source='old car-0 training residuals',
                             formula='PCA8/L2 cosine' if metric == 'cosine' else 'PCA8/L2 full Mahalanobis')

    def project(self, vector):
        z = self.components @ (vector-self.mean)
        norm = np.linalg.norm(z)
        if not np.isfinite(norm) or norm < 1e-12:
            raise ValueError('Degenerate frozen projection')
        return z/norm

    def verify(self, candidate):
        try:
            description = self.captioner.describe(candidate)
        except DescriptionError as exc:
            return dict(match='no', accepted=False, distance=None, threshold=self.threshold,
                        reason='unusable_description', error=str(exc))
        z = self.project(self._vector(description['caption']))
        diff = z-self.reference_projected
        if self.metric == 'cosine':
            distance = float(np.clip(1-z@self.reference_projected, 0, 2))
        else:
            distance = float(np.sqrt(max(0, diff@np.linalg.solve(self.full_covariance, diff))))
        accepted = distance <= self.threshold
        return dict(match='yes' if accepted else 'no', accepted=bool(accepted), distance=distance,
                    threshold=self.threshold, reason=self.metric, candidate_description=description)
