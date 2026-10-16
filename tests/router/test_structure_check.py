"""Structure check: payment certificate needs a filled certifier block (ORDERS_1940Z item 1).

Synthetic forms only. The same certification wording with a blank certifier block is a payment application.
"""

from router.structure_check import certifier_block_filled, primary, rules

BAR = "\u2588\u2588\u2588"

_FORM = (
    "Interim Payment Application No. 06 Period ending 30 June 2023. Previous cumulative amount "
    "AED 1,100,000.00. Amount due certified for payment including VAT AED 1,359,487.89. "
    "Name Signature Date Prepared by Contractor Quantity Surveyor {preparer} 05 July 2023. "
)


def test_blank_certifier_block_is_a_payment_application() -> None:
    for blank in (
        "Certified by: ________ Signature: ________ Date: ________",
        "Certified by: Signature: Date:",
        "Certified by ........ Date ........",
        "",
    ):
        text = _FORM.format(preparer=BAR) + blank
        assert not certifier_block_filled(text), blank
        assert primary(text) == "payment application", blank
        assert "payment certificate" not in rules(text), blank


def test_filled_certifier_block_is_a_payment_certificate() -> None:
    for filled in (
        f"Certified by: {BAR} Signature: {BAR} Date: 12 July 2023",
        f"Certified By: Approved By: {BAR} - Director {BAR}",
        "Approved by: Resident QS, signed 12 July 2023",
    ):
        text = _FORM.format(preparer=BAR) + filled
        assert certifier_block_filled(text), filled
        assert primary(text) == "payment certificate", filled


def test_preparer_signature_alone_does_not_certify() -> None:
    text = _FORM.format(preparer=BAR) + "Signature " + BAR
    assert not certifier_block_filled(text)
    assert primary(text) == "payment application"


def test_filled_block_without_certified_amount_is_not_a_certificate() -> None:
    text = (
        "Payment Certificate No. 03 see attached. Certified by: "
        + BAR
        + " Date: 12 July 2023. Previous claim submitted."
    )
    assert primary(text) != "payment certificate"
