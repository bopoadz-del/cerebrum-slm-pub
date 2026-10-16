"""Content-gate tests — the three blank templates reject; filled documents pass."""

from collect.content_gate import gate

BLANK_NCR_FORM = """GEOSCIENCES AND ENGINEERING DIVISION
NONCONFORMANCE REPORT
NCR NUMBER (from register):
PROJECT:
DATE OF OCCURRENCE:
RECORDED BY:
DESCRIPTION OF NON-CONFORMANCE AND REASON FOR NON-CONFORMANCE
DISPOSITION
APPROVED BY:
""" * 4  # padded to >200 chars of pure labels

BLANK_G702 = """APPLICATION AND CERTIFICATE FOR PAYMENT
Application No:  Application Date:
Period To:  Project:
CONTRACTOR:  CONTRACT FOR:
ORIGINAL CONTRACT SUM
NET CHANGE BY CHANGE ORDERS
CONTRACT SUM TO DATE
TOTAL COMPLETED AND STORED TO DATE
RETAINAGE
TOTAL EARNED LESS RETAINAGE
LESS PREVIOUS CERTIFICATES FOR PAYMENT
CURRENT PAYMENT DUE
BALANCE TO FINISH, INCLUDING RETAINAGE
ARCHITECT'S CERTIFICATE FOR PAYMENT
""" * 4

FILLED_G702 = """APPLICATION AND CERTIFICATE FOR PAYMENT
Application No: 22  Application Date: 9/14/2020
Period To: 8/31/2020
Project: McCordsville WWTP Improvements
CONTRACTOR: Walsh Construction
CONTRACT FOR: 17-001
ORIGINAL CONTRACT SUM $ 4,821,300.00
NET CHANGE BY CHANGE ORDERS $ 27,360.00
CONTRACT SUM TO DATE $ 4,848,660.00
TOTAL COMPLETED AND STORED TO DATE $ 3,102,442.00
RETAINAGE 5% $ 155,122.10
TOTAL EARNED LESS RETAINAGE $ 2,947,319.90
LESS PREVIOUS CERTIFICATES FOR PAYMENT $ 2,701,000.00
CURRENT PAYMENT DUE $ 246,319.90
BALANCE TO FINISH, INCLUDING RETAINAGE $ 1,746,218.00
ARCHITECT'S CERTIFICATE FOR PAYMENT: The undersigned certifies that payment is due.
""" * 2

FILLED_NCR = """PPPL NONCONFORMANCE REPORT NO: 3635
Open Date 03/16/06
PROJECT: NCSX Modular Coil Fabrication
SPECIFICATION: MCWF-103
DESCRIPTION OF NONCONFORMANCE: P.B. Shim is shifted 0.010 inches beyond tolerance at station 14.
DISPOSITION: Use-as-is with engineering approval.
RECORDED BY: J. Smith 03/16/06
APPROVED BY: M. Chen 03/21/06
CA: Corrective action required. Rework shim alignment procedure.
""" * 5


def test_blank_ncr_form_rejected():
    ok, reason = gate(BLANK_NCR_FORM)
    assert not ok
    assert "no filled field value" in reason


def test_blank_g702_rejected():
    ok, reason = gate(BLANK_G702)
    assert not ok


def test_filled_g702_passes():
    ok, reason = gate(FILLED_G702)
    assert ok, reason


def test_filled_ncr_passes():
    ok, reason = gate(FILLED_NCR)
    assert ok, reason


def test_short_text_rejected():
    ok, reason = gate("tiny")
    assert not ok and "200" in reason
