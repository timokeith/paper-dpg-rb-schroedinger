"""
DPG-specific ROM components: error estimator and reductor, using normal equations and affine error estimator expansion
"""

import numpy as np
from pymor.core.base import ImmutableObject
from pymor.reductors.basic import StationaryRBReductor
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.constructions import LincombOperator, VectorOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace

# =============================================================================
# 1. CUSTOM ROM ERROR ESTIMATOR
# =============================================================================
class ROMErrorEstimator(ImmutableObject):
    """
    Computes the dual norm of the residual from precomputed terms (offline-online decomposition), which limits
    its accuracy to about the square root of the machine precision.
    """

    def __init__(self, op_f, op_G_inv):
        self.__auto_init(locals())

        f_vec = self.op_f.operators[0].as_range_array()
        ginv_f = self.op_G_inv.apply(f_vec)
        self.const_ff = f_vec.inner(ginv_f)[0, 0].real
        print(f"  > Error Estimator: ||f||^2_{{G^{{-1}}}} = {self.const_ff:.6e}")

    def estimate_error(self, U, mu, rom):
        """
        Evaluate ||f(mu) - B(mu) U||_{G_{mu_mean}^{-1}} using reduced normal equation components.
        """

        # Quadratic Term
        AU = rom.operator.apply(U, mu)
        quad = U.inner(AU)[0, 0].real

        # Linear Term
        F = rom.rhs.as_range_array(mu)
        lin = U.inner(F)[0, 0].real

        # ||r||²_{G^{-1}} = ||f||²_{G^{-1}} + u_N^H A_NE u_N - 2 Re(u_N^H F_NE)
        resid_norm_sq = self.const_ff + quad - 2.0 * lin

        # Handle numerical zero
        if resid_norm_sq < 0:
            if abs(resid_norm_sq) < 1e-9 * max(abs(self.const_ff), abs(quad), 1e-15):
                # Within numerical tolerance - treat as zero
                resid_norm_sq = 0.0
            else:
                print(f" >> WARNING: Resid^2 = {resid_norm_sq:.4e} is significantly negative!"
                      f"\n(const={self.const_ff:.4e}, quad={quad:.4e}, lin={lin:.4e})")
                resid_norm_sq = 0.0

        resid_norm = np.sqrt(resid_norm_sq)

        return np.array([resid_norm])

# =============================================================================
# 2. PROJECTED REDUCTOR
# =============================================================================
class DPGProjectedReductor(StationaryRBReductor):
    """
    Galerkin projection reductor with DPG error estimator.
    """
    def __init__(self, fom, RB=None, product=None,
                check_orthonormality=None, check_tol=None):

        super().__init__(fom, RB=RB, product=product,
                         check_orthonormality=check_orthonormality, check_tol=check_tol)

        # Store references for error estimator construction
        self._op_f = fom.op_f
        self._op_G_inv = fom.op_G_inv
        self.error_estimator = None

        # Cache applied operators to avoid repeated solves during projection
        self._b_applied = [fom.op_G_inv.source.empty() for _ in fom.op_B.operators]
        self._supremizers = [fom.op_G_inv.range.empty() for _ in fom.op_B.operators]
        self._ginv_f = [fom.op_G_inv.apply(f.as_range_array()) for f in fom.op_f.operators]

    def extend_basis(self, U, **kwargs):
        old_size = len(self.bases['RB'])
        super().extend_basis(U, **kwargs)
        U_new = self.bases['RB'][old_size:]

        for q, B_q in enumerate(self.fom.op_B.operators):
            b_q_U = B_q.apply(U_new)
            w_q = self.fom.op_G_inv.apply(b_q_U)
            self._b_applied[q].append(b_q_U)
            self._supremizers[q].append(w_q)

    def project_operators(self):
        red_space = NumpyVectorSpace(len(self.bases['RB']))
        ops_b, coeffs_b = self.fom.op_B.operators, self.fom.op_B.coefficients

        lhs_ops, lhs_coeffs = [], []
        for i in range(len(ops_b)):
            for j in range(i, len(ops_b)):
                mat = self._b_applied[i].inner(self._supremizers[j])
                lhs_ops.append(NumpyMatrixOperator(mat))
                lhs_coeffs.append(coeffs_b[i] * coeffs_b[j])
                if i != j:
                    mat_sym = self._b_applied[j].inner(self._supremizers[i])
                    lhs_ops.append(NumpyMatrixOperator(mat_sym))
                    lhs_coeffs.append(coeffs_b[i] * coeffs_b[j])
        rom_op = LincombOperator(lhs_ops, lhs_coeffs)

        rhs_ops, rhs_coeffs = [], []
        coeffs_f = self.fom.op_f.coefficients
        for k in range(len(self.fom.op_f.operators)):
            for i in range(len(ops_b)):
                vec = self._b_applied[i].inner(self._ginv_f[k]).flatten()
                rhs_ops.append(VectorOperator(red_space.make_array(vec)))
                rhs_coeffs.append(coeffs_b[i] * coeffs_f[k])
        rom_rhs = LincombOperator(rhs_ops, rhs_coeffs)

        return {'operator': rom_op, 'rhs': rom_rhs, 'products': {}}

    def assemble_error_estimator(self):
        """Called by reduce() to attach error estimator to ROM."""
        if self.error_estimator is None:
            self.error_estimator = ROMErrorEstimator(self._op_f, self._op_G_inv)
        return self.error_estimator
