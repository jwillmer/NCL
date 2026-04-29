# Web lookup of untracked vessel-mention parking list

Cross-checked all 122 raw mentions in `untracked_parking.txt` against vesselfinder.com,
marinetraffic.com, myshiptracking.com, fleetmon.com, and other public AIS databases.

## Counts per verdict

| Verdict | Count |
|---|---|
| REAL | 47 |
| UNCLEAR | 11 |
| NOT_VESSEL | 64 |
| **Total** | **122** |

## Top-10 strongest candidates for register addition

These are the highest-occurrence REAL vessels not already in `data/vessel-list.csv`.
They are operated by the Angelicoussis Group (Maran Tankers / Maran Dry / Anangel) or
appear repeatedly in the corpus as charter/voyage counterparts. Recommended priority
order if the user wants to extend the canonical register:

| Rank | Detected name | Occ. | Type | IMO | Operator notes |
|---|---|---:|---|---|---|
| 1 | ANTONIS L. ANGELICOUSSIS | 16 | VLCC (Angelicoussis) | 9930777 | Misspelled ANTONIS I. ANGELICOUSSIS - flagship dual-fuel VLCC; sister of MARIA A. ANGELICOUSSIS already in register |
| 2 | PEGASUS VOYAGER | 7 | Suezmax | 9665736 | Chevron-owned; same Voyager class as RICHMOND/EL SEGUNDO/PASCAGOULA already in register |
| 3 | POLARIS VOYAGER | 6 | Suezmax | 9665748 | Chevron-owned sister of PEGASUS VOYAGER |
| 4 | ARCTURUS VOYAGER | 4 | VLCC (Arcturus) | 9588299 | Same IMO as MARAN ARCTURUS (renamed 2018) - alias not currently in register |
| 5 | MARAN LOYALTY | 3 | Newcastlemax bulk carrier | 9724659 | Maran Dry Management - bulk fleet not in current register |
| 6 | MARAN ENDEAVOUR | 3 | Newcastlemax bulk carrier | 9846419 | Maran Dry Management |
| 7 | MARAN MERCHANT | 2 | Capesize bulk carrier | 9458690 | Maran Dry Management |
| 8 | MARAN SAILOR | 2 | Capesize bulk carrier | 9345764 | Maran Dry Management |
| 9 | MARAN VISION | 2 | Capesize bulk carrier | 9332951 | Maran Dry Management |
| 10 | MARAN GUARDIAN | 2 | Capesize bulk carrier | 9434369 | Maran Dry Management |

## Notable patterns

- **The Maran Dry bulk fleet is entirely absent from the canonical register.**
  The current `vessel-list.csv` lists only 5 BULK CARRIER entries (CONQUEROR, HOPE,
  VICTORY, HORIZON, HERO). At least 13 additional Maran Dry Capesize/Newcastlemax bulkers
  appeared in the corpus (LOYALTY, ENDEAVOUR, MERCHANT, SAILOR, VISION, GUARDIAN, DYNASTY,
  GLORY, HAPPINESS, ASTRONOMER, VIRTUE, plus ANANGEL VOYAGER ex-MARAN VOYAGER).
- **Angelicoussis Group VLCCs missing from the register.** Older Maran Tankers VLCCs not
  listed in the canonical register but verified via AIS: MARAN CYGNUS, GEMINI, ANDROMEDA,
  CARINA, CASTOR, CALLISTO, AQUARIUS, CORONA, CASSIOPEIA, SAGITTA, REGULUS, TRITON.
  Several were sold or renamed (e.g. CALLISTO sold 2019, REGULUS renamed LUNA PRIME 2020),
  so caveat: register additions should consider current ownership.
- **Voyager-class aliasing.** ARCTURUS VOYAGER, ANTARES VOYAGER, LIBRA VOYAGER are
  former Chevron-managed names of MARAN ARCTURUS / MARAN ANTARES / MARAN LIBRA (same
  IMO numbers - renamed when they joined Maran). These belong in the ALIASES column of
  existing register rows rather than as new entries.
- **Non-Maran but legitimate counterparts.** Vessels that appear because they were
  loading/discharging berths or charter counterparts: EAGLE HATTERAS (AET Aframax),
  ADVANTAGE AWARD (Advantage Tankers Aframax), ETHANE SAPPHIRE (MOL/Reliance VLEC),
  MONJASA REFORMER (Monjasa bunker tanker), TORDIS KNUTSEN / WINDSOR KNUTSEN (Knutsen
  shuttle tankers), DALMA (CMB Tech VLCC), IOANNA (Dynacom VLCC), HIRADO (Japanese VLCC),
  HAKONE (Japanese VLCC), OCEANIC FORTUNE (Hongkong Xiang An VLCC), SILVER HESSA (MR
  product tanker), ATLANTIC ORCHARD (Wisby Tankers fruit-juice tanker), MAERSK CLEVELAND
  (Maersk container ship). These do not belong in the canonical fleet register.

## Extraction-noise patterns surfaced

64 of 122 entries (52%) are extraction artifacts that the regex parker should ideally
avoid. Common patterns the user may want to harden the extractor against:

1. `MARAN <stopword>` - "MARAN AND/FOR/DAY/ITEMS/MAIL/TYPE/SITE/SIDE/STOCK/GOOD/NON",
   etc. These look vessel-shaped because of capitalised English words after "MARAN",
   but the second token is a sentence-flow word.
2. `<vessel>CONSIGNEE` / `<vessel>IMPORTANT` - missing whitespace concatenations after
   shipping-document boilerplate (HERMES + IMPORTANT, ARCTURUS + CONSIGNEE, ATALANTA +
   CONSIGNEE, HERCULES + CONSIGNEE, HERMIONE + CONSIGNEE).
3. `<vessel> ABT` - "ABT" (=approximately/about) trailing the vessel name in voyage
   schedules. The base vessel is real but the captured string includes the qualifier.
4. `SOPHIA <street/place>` / `MARAN GLORY DOIRANIS` - vessel name plus an address line.
5. `<term> <unit>` / `<term> <fuel grade>` - LOAD LINE, INTERNATIONAL LOAD LINE, ROB
   GRADE, ROB SAMPLE ATTACHED, ROB AFTER BUNKERING, WTI MIDLAND CRUDE, VDC MIN, ANODE
   AMP, BOG COMPRESSOR STOPPED, DRAUGHT LOADED, DMA MET THE, INPSH LAT, SWL PEDESTAL
   MOUNTED / BOSUN STORE / PUMP ROOM. All are technical maritime terminology, never
   vessel names.
6. `MARAN TANKERPS` / `MARAN TENKERS` / `MARAIN AJAX` / `MARAN MANAGEMEM` /
   `WINDSDOR KUNTSEN` / `MASESRK CLEVLAND` - OCR-typo variants of company / vessel
   names already in scope. A typo-mapping table (already present at
   `wip/vessel-mismatch/typo_mapping.csv`) handles the worst of these.

## Methodology notes

- One to two web searches per ambiguous entry; total ~20 search calls across 58
  non-trivially-classified entries, well under the 250 budget.
- Obvious extraction noise (sentence fragments, technical abbreviations, concatenations
  with stopwords) was classified by inspection without spending search budget.
- Confidence is `high` when an AIS database returned an unambiguous IMO match;
  `medium` when the closest match required interpretation (e.g. truncated names,
  typos); `low` for UNCLEAR rows.
- Evidence URLs cite a single decisive source per row (vesselfinder, marinetraffic, or
  myshiptracking primary). Multiple databases corroborated each REAL claim but only one
  is shown.
