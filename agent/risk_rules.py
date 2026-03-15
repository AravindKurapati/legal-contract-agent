"""
Deterministic risk rules applied on top of model-extracted clauses.
Each rule checks for a specific pattern that signals legal risk.
"""


def apply_risk_rules(clause_results: dict) -> list[str]:
    """
    Input:  dict of {clause_display_name: [extracted_clause_texts]}
    Output: list of human-readable risk flag strings
    """
    flags = []

    # ── Rule 1: No indemnification clause ─────────────────────────────────────
    # Indemnification protects you if the other party causes harm.
    # Missing = you have no protection for third-party claims.
    if not clause_results.get("Indemnification"):
        flags.append(
            "⚠️ HIGH RISK: No indemnification clause found. "
            "Neither party is protected against third-party claims."
        )

    # ── Rule 2: No liability cap ───────────────────────────────────────────────
    # Without a cap, one party could be liable for unlimited damages.
    # Uncapped liability is almost always a red flag.
    has_cap = bool(clause_results.get("Cap On Liability"))
    has_uncapped = bool(clause_results.get("Uncapped Liability"))

    if not has_cap:
        flags.append(
            "⚠️ HIGH RISK: No liability cap found. "
            "Exposure to unlimited damages."
        )
    if has_uncapped:
        flags.append(
            "⚠️ HIGH RISK: Uncapped liability clause present. "
            "Review carefully before signing."
        )

    # ── Rule 3: Auto-renewal without termination right ────────────────────────
    # Auto-renewal locks you in. If there's no termination for convenience,
    # you may be stuck in the contract indefinitely.
    has_renewal = bool(clause_results.get("Renewal Term"))
    has_termination = bool(clause_results.get("Termination For Convenience"))
    has_notice = bool(clause_results.get("Notice Period To Terminate Renewal"))

    if has_renewal and not has_termination:
        flags.append(
            "⚠️ MEDIUM RISK: Auto-renewal present but no termination "
            "for convenience clause. You may not be able to exit easily."
        )

    if has_renewal and not has_notice:
        flags.append(
            "⚠️ MEDIUM RISK: Renewal term present but no notice period "
            "specified. Unclear how to prevent automatic renewal."
        )

    # ── Rule 4: IP ownership assigned away ────────────────────────────────────
    # If IP created under the contract belongs to the counterparty,
    # you lose ownership of your own work. Always worth flagging.
    if clause_results.get("IP Ownership Assignment"):
        flags.append(
            "⚠️ MEDIUM RISK: IP Ownership Assignment clause present. "
            "Intellectual property may transfer to counterparty. "
            "Review scope carefully."
        )

    # ── Rule 5: Non-compete present ───────────────────────────────────────────
    # Non-competes restrict your ability to operate your business.
    # Always flag for lawyer review.
    if clause_results.get("Non-Compete"):
        flags.append(
            "⚠️ MEDIUM RISK: Non-compete clause present. "
            "Restrictions on business activities may apply after contract ends."
        )

    # ── Rule 6: No governing law specified ────────────────────────────────────
    # Without governing law, jurisdiction disputes become expensive.
    if not clause_results.get("Governing Law"):
        flags.append(
            "⚠️ LOW RISK: No governing law clause found. "
            "Jurisdiction in case of dispute is unclear."
        )

    # ── Rule 7: Change of control clause ──────────────────────────────────────
    # If your company gets acquired, this clause may let the
    # counterparty terminate or renegotiate. Important for startups.
    if clause_results.get("Change Of Control"):
        flags.append(
            "ℹ️ NOTICE: Change of control clause present. "
            "Contract terms may be affected by acquisition or merger."
        )

    # ── Rule 8: No expiration date ────────────────────────────────────────────
    # Contracts with no end date can run indefinitely.
    if not clause_results.get("Expiration Date"):
        flags.append(
            "⚠️ LOW RISK: No expiration date found. "
            "Contract may have no defined end date."
        )

    return flags