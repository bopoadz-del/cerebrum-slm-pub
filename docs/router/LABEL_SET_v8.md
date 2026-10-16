# Router label set v8 (approved)

Status: APPROVED by the owner (CHADi Mahmoud), 2026-10-07 01:18 +04.
Source: Google Doc "Router label set v7 rev2 — owner edits applied (approve)"
(file id 13gyr_cLGROoYpa2by0cwSkc-znlBzQ_O09-pae7x9Hs). The owner calls the
resulting sealed set / labelling sheet "v8".

These definitions are the single reference: printed on the labelling sheet
and given to the teacher model. Machine-readable copy:
`configs/router/label_set_v8.json`. The text below is copied verbatim from the
approved Doc (sections "Changes from rev1" through "Out of scope"); only the
blank owner decision block was replaced by the approval record above.

Sealed v5 records are unchanged and are not re-scored under this label set.

---

```text
Changes from rev1 (owner, 2026-10-07)

- Added label: register/log (VO log, submittal register, risk register, action tracker, drawing register, stakeholder tracker and similar).

- Split payment certificate into: payment application (contractor's claim for payment) and payment certificate (engineer's/certifier's certificate).

- Consequential wording: transmittal, report and other no longer mention registers; regional mappings updated for the payment split.

- Everything else unchanged (contract, method statement, NCR, site instruction, inspection request, VO/CO instrument-not-memo rule, regional mappings).

Design rules

- A label is decided from document content only. Filename, folder, source site and form number never decide a label.

- Regional names map to one label (see Regional mappings).

- "other" is only for construction project documents that fit none of the named labels.

- Letter-form notices that are a claim, RFI, NCR, site instruction or payment application take that specific label, not correspondence.

- A register lists many items of one kind; a single item listed in it takes its own label (one VO is VO/CO; the VO log is register/log).

────────────────────────────────────────

LABELS (21)

────────────────────────────────────────

1. RFI

Definition: A formal request for information about design, specification, drawing or site condition, seeking an answer that unlocks work.

Is not: A general email; a submittal; a claim; a response letter alone; an RFI log (that is register/log).

2. submittal

Definition: Contractor-issued package seeking approval or review of product data, shop drawings, samples or method material before use or installation.

Is not: A drawing issued for construction; a transmittal cover with no submittal substance; an RFI; a submittal register (that is register/log).

3. transmittal

Definition: A cover sheet that records the issue or return of a specific batch of documents between parties, with issue purpose and distribution.

Is not: The document being transmitted; a submittal package itself; meeting minutes; a running document or drawing register (that is register/log).

4. VO/CO

Definition: The change or variation instrument itself that alters scope, price and/or time.

Is not: A memo, agenda, staff report or email that merely recommends or approves a change; a claim seeking a change; a programme update; a VO log (that is register/log).

Regional names that map here: variation order (VO), change order (CO), variation instruction, engineer's instruction (EI) when it varies the works, architect's instruction (AI) when it varies the works.

5. claim

Definition: A contractor (or other party) notice or submission asserting entitlement to time and/or money under the contract, with particulars of cause and quantum or an express reservation of rights.

Is not: Policy or guidance about claims; a change order already agreed; a routine progress report; a blank claim template; a claims register (that is register/log); a payment application.

Regional names that map here: notice of claim, claim for extension of time (EOT claim), claim for additional payment, request for equitable adjustment (REA) when framed as a contractual claim.

6. payment application

Definition: The contractor's (or subcontractor's) application for payment stating the amount it claims is due for work done and materials, submitted for review and certification.

Is not: The engineer's/certifier's payment certificate; a supplier invoice outside the contract payment cycle; a claim for additional entitlement; a cost report.

Regional names that map here: application for payment, interim application, pay application / pay app, progress claim (AU/NZ), payment claim (AU/NZ security of payment), contractor's statement (FIDIC Sub-Clause 14.3).

Combined forms (e.g. one form with both the contractor's application and the certifier's block): label by what the document as returned contains. If the certifier's section is blank or not signed, it is payment application.

7. payment certificate

Definition: The certificate issued by the engineer, architect, QS or employer's representative stating the amount certified as due to the contractor.

Is not: The contractor's application; a change order; a cost report without a certified amount; a payment register (that is register/log).

Regional names that map here: interim payment certificate (IPC), final payment certificate, certificate of payment, payment certificate, valuation certified by the QS, engineer's certificate under FIDIC Sub-Clause 14.6/14.13.

Combined forms: if the certifier's section is completed and signed with a certified amount, it is payment certificate.

8. drawing

Definition: A design or construction drawing (plan, section, elevation, detail, shop drawing issued as a drawing) showing geometry or arrangement of the works.

Is not: A schedule bar chart; a specification; a photograph report; a sketch inside an RFI unless the whole document is issued as a drawing; a drawing register (that is register/log).

9. specification

Definition: Prescriptive or performance text stating materials, workmanship, standards or procedures for the works (specification sections, particular specifications, employer's requirements technical clauses).

Is not: A method statement; a contract commercial clause set; a programme.

10. schedule

Definition: A time programme for the works: baseline, updated or look-ahead programme with activities, logic, durations and dates (Gantt, CPM, P6 print, construction sequence programme).

Is not: Narrative text about how schedules must be prepared; a meeting agenda; a list of items with status but no time network (that is register/log).

11. report

Definition: A narrative or tabulated report of status, progress, inspection findings, investigation or advice, issued for a point in time or period (progress report, site report, technical report, consultant advice report).

Is not: Meeting minutes; an NCR; a payment certificate; a claim; a running record of many items maintained over time (that is register/log).

12. correspondence

Definition: A project letter or email between parties that is not primarily one of the named instruments above (general notice, cover letter, reply, chase, confirmation).

Is not: An RFI; a claim notice; an NCR; a site instruction; a payment application; a variation order.

13. BOQ

Definition: Bill of quantities or schedule of rates: itemised measured work with quantities and (usually) rates or price columns used for tender or payment measurement.

Is not: A payment certificate or application; a cost report; a material delivery note.

14. contract

Definition: The agreement or deed that forms the contract (articles of agreement, contract agreement, letter of acceptance with annexed conditions, main conditions of contract as executed).

Is not: A variation order; a specification volume alone; meeting minutes of a commercial discussion.

15. method statement

Definition: Contractor's method statement or work method: step-by-step how a work activity will be carried out, including plant, sequence and controls.

Is not: A specification; a risk assessment alone unless issued as the method statement; a programme.

16. meeting minutes

Definition: Record of a project meeting (kick-off, progress, technical, commercial) listing attendees, matters discussed and actions.

Is not: A report; an agenda with no record of discussion; correspondence confirming one action; a standalone action tracker (that is register/log).

17. NCR

Definition: Non-conformance / non-compliance report: formal record that work, material or process does not meet the specified requirement, usually with disposition.

Is not: A routine inspection checklist with no non-conformance; a site instruction; an RFI; an NCR log (that is register/log).

18. inspection request

Definition: Request for inspection or test: notice that work is ready for inspection, witness or test.

Is not: An information RFI; an NCR; an inspection report (that is report); an inspection request log (that is register/log).

Regional names that map here: inspection and test request (ITR), request for inspection (called "RFI" in some Gulf practice — disambiguate from information RFI by content), notice of inspection, work inspection request (WIR), material inspection request (MIR).

19. site instruction

Definition: Direction issued on site to proceed, hold, or change work methods or temporary works without (by itself) agreeing time or money. An engineer's instruction that does not vary price or time stays here; if it varies price or time it is VO/CO.

Is not: A variation order; a method statement; meeting minutes.

20. register/log

Definition: A tabulated running record of many items of one kind, with status columns, maintained over time. The register is not the items it lists.

Examples: VO log, change order log, submittal register, RFI log, drawing register, document register, risk register, issues log, action tracker, NCR log, claims register, payment register, stakeholder tracker, permit tracker.

Is not: A single item from the list (label that item by its own type); a BOQ (priced quantities, not a status record); a programme (time network); a one-off report table.

21. other

Definition: A genuine construction project document that does not fit any named label above (for example: insurance certificate, permit/NOC, survey certificate, test certificate, warranty, handover certificate).

Is not: A way to park a document you have not read; a template with no project content.

────────────────────────────────────────

REGIONAL MAPPINGS (one label)

────────────────────────────────────────

Variation order / change order / variation instruction / engineer's or architect's instruction that varies the works → VO/CO

Application for payment / pay app / interim application / progress claim or payment claim (AU/NZ) / contractor's statement (FIDIC 14.3) → payment application

Interim payment certificate / final payment certificate / certificate of payment / QS-certified valuation / engineer's certificate (FIDIC 14.6, 14.13) → payment certificate

Notice of claim / EOT claim / REA framed as claim → claim

Request for inspection / ITR / WIR / MIR / "RFI" meaning inspection readiness → inspection request

Information RFI / query seeking design or spec clarification → RFI

VO log / submittal register / RFI log / drawing register / risk register / action tracker / stakeholder tracker → register/log

────────────────────────────────────────

OUT OF SCOPE

────────────────────────────────────────

Emails that are pure chat with no project instrument; marketing PDFs; blank vendor templates; legislation; textbooks.
```
