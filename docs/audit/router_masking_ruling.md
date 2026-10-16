# Router masking ruling

Date: 2026-10-07, 17:03 +04 (owner order, router v9 continue).

## The ruling, as given

- The router adapter trains on UNMASKED rows.
- The masker stays PARKED. No spend on it.
- The masking bar still applies to shipped or published data and to posture-model training text: "never train the posture adapter on unmasked text".
- The NEW GATE is the name-blind check.

## What it means in plain words

1. Router training rows keep their original text. Names of people, firms, projects and places stay in the text the router learns from. This PR puts the 219 pr2 owner-source rows back in their original, unmasked form.
2. The masker is not used, tuned or paid for. It stays parked.
3. Masking is still required before any data is shipped or published, and for any text used to train the posture adapter. The posture adapter is never trained on unmasked text.
4. Router data stays internal. It never leaves this repository or the owner's Drive.
5. The check that replaces masking for the router is the name-blind test (`scripts/router/name_blind.py`). Title-block and party fields are blanked by their position in the layout, never by a list of names or words. The same adapter, with the same settings, classifies the documents again. Any changed answer on a passable class (n >= 12) is a FAIL. The fix for a FAIL is more training rows from other projects, never a patch.

## Forbidden under this ruling

- Masking by a name list.
- Bigger-teacher spend on the masker.
- Training the posture adapter on unmasked text.
- Router data leaving the repo or the owner's Drive.
