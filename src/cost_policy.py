"""Cost-matrix approval policy (documented assumptions, not bank policy).

Assumptions used in train.py (stated in README/metrics.json):
  - Exposure per applicant = LIMIT_BAL (credit limit, NT$).
  - Loss given default (LGD) = 0.45.
  - Foregone interest margin on a declined good applicant = 0.12
    of LIMIT_BAL (one-year margin, illustrative).
  - Expected cost of approving = P(default) * LGD * LIMIT_BAL.
  - Expected cost of declining = (1 - P(default)) * 0.12 * LIMIT_BAL.
  - Cost-optimal approve rule: approve iff
        P * 0.45 < (1 - P) * 0.12
    i.e. P < 0.12 / (0.45 + 0.12) = 0.2105.

Baseline for the "constant approval rate" comparison:
  Approve the same fraction of test applicants as the cost-optimal
  policy, but ranked by a single-feature heuristic (PAY_0, the most
  recent repayment status, lower = better) instead of the model.
  Credit loss for a policy = sum over APPROVED actual defaulters of
  LGD * LIMIT_BAL. Foregone interest is reported separately and is
  NOT counted in "credit loss".
"""

LGD = 0.45
MARGIN = 0.12


def cost_optimal_threshold() -> float:
    return MARGIN / (LGD + MARGIN)


def credit_loss(approved_mask, y_true, limit_bal) -> float:
    """Sum of LGD * LIMIT_BAL over approved applicants who defaulted."""
    import numpy as np

    approved_mask = np.asarray(approved_mask, dtype=bool)
    y_true = np.asarray(y_true)
    limit_bal = np.asarray(limit_bal, dtype=float)
    defaulted_and_approved = approved_mask & (y_true == 1)
    return float((limit_bal[defaulted_and_approved] * LGD).sum())
