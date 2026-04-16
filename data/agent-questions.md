# AI Agent Questions & Expected Answers

Questions an AI agent should be able to answer given the imported data from the `output/` folder.

**Dataset:** 483 documents (201 emails, 187 images, 58 PDFs, 27 spreadsheets), 966 chunks, 93 topics, 80+ vessels across 3 types (VLCC, SUEZMAX, AFRAMAX) and 13 vessel classes, 104 external email domains.

---

## 1. Vessel Operations & Status

### Q: What maintenance issues are currently open for MARAN CANOPUS?

CANOPUS has two key issues:

1. **Main engine governor failure** — governor not regulating RPM correctly (RPM remains higher than setpoints, fuel index at 45%). A quotation for repair has been requested (CRM:0504838) and spare parts shipment is being coordinated.
2. **SOx scrubber/EGCS internal inspection** — CANOPUS is one of multiple vessels (along with Arcturus, Cleo, Plato class) flagged for fleet-wide EGCS internal inspection covering leakage in bellows, structural integrity of steel plating, and deposits/contamination checks.

### Q: Which vessels have reported equipment failures in the last month?

Equipment failures reported across the fleet:

- **DANAE** (Angelicoussis class) — ICMS RCP-X32 control system failure
- **CANOPUS** — Main engine governor malfunction, RPM regulation issue
- **ARTEMIS** (Apollo class) — Tank remote level system error
- **THETIS** (Arcturus class) — EGCS bellows leakage during internal inspection
- **THALEIA** (Arcturus class) — Gas detection system malfunction for ballast tanks, caused by water ingress through cracked inlet air pipe
- **MIRA** (Aphrodite class) — RADAR/ECDIS target acquisition display failure
- **ANTONIS I. ANGELICOUSSIS** — Defective Inmarsat-C No.1 antenna
- **CLEO** — Service air compressor drive belt failure; BWTS water chiller PLC malfunction with main breaker trip
- **MARAN HERCULES** (Hermione class) — S-Band radar software issues, communication failure to ARPA/VDR/ECDIS
- **ARES** (Ajax class) — Vapour & manifold pressure monitoring system audible alarm failure
- **MARAN APOLLO** — TeamTec PLC/HMI failure, urgent spare parts order
- **MARAN ARETE** (Apollo class) — Service air compressor malfunction
- **ARIADNE** (Apollo class) — Main air compressor failures, fleet experience sharing bulletin No.18-22

### Q: Which vessels have upcoming statutory surveys or certification renewals due?

The ANTONIS I. ANGELICOUSSIS vessel status report shows multiple inspection items due by 02-Jan-2028, including Inert Gas Blowers No.02 and No.03 examinations, and SCR system components (burner gas oil pump, SCR chambers for generator engines 1-3). Additionally, MARAN HELEN has a Lloyd's Register remote survey in progress for an incinerator issue (class LR), with Dr. Andreas Ioannou from LR Cyprus coordinating. Three documents relate to Cargo Ship Safety Construction Certificate issuances following vessel surveys in Greece.

---

## 2. Procurement & Supply Chain

### Q: What purchase orders have been rejected and why?

PO 4500497660 was rejected. Poulopoulou Sofia from Angelicoussis Group Accounting/Accounts Payable notified Mr. Lionakis on 2025-07-01 that the PO is rejected and they cannot proceed with invoice posting. The specific rejection reason is not stated in the email — it requests verification of the PO before further processing.

### Q: What spare parts are currently being sourced for the fleet?

Active procurement includes:

- **HP LNG pump discharge thermal relief valve TSV212** — replacement being coordinated with shipping to Netherlands warehouse, 6-8 week lead time (PO 274050, FTR 24-0261)
- **Cargo valve HRM882V spare parts** — clarification on part numbers and actuator types (PAC80U-04P/05P) for correct matching
- **High-pressure fuel oil pipe union nipples** (part 4272-0100-0105) — upgrade advisory, not immediately required
- **NT1765 temperature sensor** — decision needed on version for 8000HRS maintenance kit SPH806
- **TeamTec spare parts for MARAN APOLLO** — urgent PLC/HMI components (APL1014 order)
- **MARAN THALEIA** — new PO 4500561906 from Erma First ESK Engineering Solutions for CHEMS & GASES
- **MARAN ATHENA** — urgent PO via ATT Electric & Machinery Pte Ltd (TA0202506000234)

### Q: What are the liferaft exchange costs and logistics for Singapore?

Per the vendor's response for MARAN MARS liferaft exchange at Singapore Roads:

- **Lead time:** 5 working days to prepare liferafts; overtime charges apply if shorter
- **Overtime:** USD 187 per liferaft
- **Delivery expenses at Singapore:** USD 353
- **Barge/crane service** must be arranged by customer through local agent
- Vendor requires approval before preparing/reserving the requested liferafts
- Local agent details needed to check vessel ETA and apply permit

---

## 3. Crew & Logistics

### Q: What crew changes are scheduled and for which vessels?

MT MARAN ATHENA has a crew change in Yeosu (June 2025):

- **4 on-signers** (21 June via boat): MST Delidakis Ioannis, C/O Sopiropoulos Dionysios, 2/O Kostopoulos Panagiotis, 3/E Tsilipanis Alexandros. Flight route: ATH to AUH (EY0190, 19 June) to ICN (EY0822, 20 June) to RSU (OZ8735, 21 June)
- **3 off-signers** (22 June via boat): C/O Karakonstantakis Stylianos, 2/O Dimitriou Christos, 3/E Pailas Kyriakos. Flight route: RSU to GMP (OZ8734, 23 June) to AUH (EY0823) to ATH (EY0187, 24 June)
- Superintendent Mr. Asproulis Georgios also aboard for the port call

### Q: What airfreight shipments are being tracked?

There is an active airfreight booking with flight schedule and cargo manifest for an international shipment. Related topics include documentation and pre-alert requirements, with pending pre-alert documents and final cargo list preparation. The shipment involves an AWB (air waybill) 16 8156 1114 for 1kg from Maran Ship Supplies PTE LTD as part of the MARAN ATHENA owner's matters in Yeosu.

---

## 4. Technical & Maintenance

### Q: What EGCS/scrubber inspection findings have been reported across the fleet?

Fleet-wide EGCS internal inspection covers vessels from Arcturus, Canopus, Cleo, and Plato classes. Key findings and inspection areas:

- **THETIS** (Arcturus) — bellows leakage in the EGCS tower identified
- **Fleet VDL maintenance** — inspection scope includes: steel plating wastage/cracks in Chinese hat plates, weld joint integrity, soot accumulation, salt deposits, water ingress indicators, and demister panel condition
- The inspection requires port-side attendance with a detailed report and representative photographs
- Multiple related topics tracked: SOx Scrubber internal inspection, leakage in EGCS components, structural integrity of scrubber components, deposits and contamination checks

### Q: What is the BWTS situation on CLEO?

CLEO has a BWTS water chiller PLC malfunction. The main breaker tripped affecting the BWTS, and the water chiller cannot auto/remote start. There is both an electrical power supply interruption issue and a PLC control failure. Multiple email threads are tracking this across the Cleo vessel class.

### Q: Summarize the ICCP MGPS monthly report feedback for MARAN ASPASIA.

A request was sent to review and provide comments/feedback on MARAN ASPASIA's June 2025 ICCP MGPS monthly report. This is one of the highest-coverage topics in the dataset (2 documents, 14 chunks). Related topics include regulatory reporting for ICCP MGPS and a feedback/review request cycle. Monthly ICCP MGPS report submission is part of the vessel's compliance documentation workflow.

### Q: What boiler and auxiliary system maintenance logs have been submitted?

Monthly vessel maintenance/inspection logs for boilers and water analysis have been submitted for review. The related topics include maintenance documentation submission and external report review requests. The workflow involves reviewing monthly maintenance/operational logs (boilers and auxiliary systems) and providing a report, which involves document exchange between vessel crew and shore-side technical management.

---

## 5. Compliance & Classification

### Q: Which vessels have had PSC deficiencies identified?

A Korean Register PSC information circular (I45-670754) flagged lifesaving equipment deficiencies — specifically issues with lifeboat release mechanisms and lifeboats identified during Port State Control inspections. This is a fleet-wide advisory rather than a single-vessel finding.

### Q: What safety certificates have been issued recently?

Three documents relate to Cargo Ship Safety Construction Certificate issuance. Certificates were issued following vessel surveys/audits in Greece (Greece B and Greece C designations). This is the highest-coverage topic in the dataset with 3 documents and 12 chunks. The certificates were published following completion of regulatory compliance surveys.

### Q: Are there any regulatory fee changes that affect our operations?

Yes. A circular notification was issued regarding an increase in transit sanitary dues charged per NRT, effective 01/07/2025. This impacts transit costs for all vessels in the fleet passing through the affected jurisdiction.

---

## 6. Bunker & Fuel

### Q: What are the latest bunker fuel prices by port?

Per the Spectra Fuels Bunker Price Report dated 01.07.2025 (USD/PMT):

| Port | VLSFO 0.5% | HSFO 3.5% | LSMGO 0.1% |
|------|-----------|-----------|------------|
| Busan | 559 | 473 | 689 |
| Hong Kong | 533 | 462 | 652 |
| Shanghai | 572 | 475 | 686 |
| Tokyo | 543 | 470 | 797 |
| Singapore | (continues in full report) | | |

### Q: What is the status of urea supply procurement for SCR-equipped Tier III engines?

Coordination and approval of urea supply for SCR-equipped Tier III engines is in progress, including Rotterdam delivery. The vessel status report for ANTONIS I. ANGELICOUSSIS shows SCR system components (SCR Burner Marine Gas Oil Pump, SCR Chambers for Generator Engines No.01-03) with examinations due by 02-Jan-2028. Supplier evaluation is ongoing.

---

## 7. IT & Connectivity

### Q: What network issues have been reported on MARAN DIONE and what are the troubleshooting steps?

MARAN DIONE (Angelicoussis class) has a SHI Smartship BIG System connectivity issue to cloud services. The troubleshooting steps from the engineer:

1. Check LAN cable A75-1-1 between port No.0 of SMARTSHIP Firewall and VSAT rack — verify proper connection and LED blinking
2. Have vessel IT engineer verify IP configuration: IP 172.16.1.138, Subnet 255.255.255.248, Gateway 172.16.1.137, DNS 172.16.1.137
3. Confirm whitelist includes teamviewer.com and amazonaws.com (required for ship data transmission to AWS)
4. If cable is connected and LEDs are normal, run nslookup diagnostics, disconnect/reconnect VSAT, and test domain resolution

---

## 8. Cross-Cutting / Analytics

### Q: Which vessel classes have the most operational issues?

By topic-linked chunk count:

1. **Angelicoussis** — 53 chunks (ICMS failure, Inmarsat antenna, DANAE issues)
2. **Apollo** — 47 chunks (tank level error, air compressor, PLC/HMI failure, crew changes)
3. **Hermione** — 41 chunks (S-band radar software, Helen class/incinerator survey)
4. **Arcturus** — 39 chunks (EGCS inspections, gas detection malfunction, THETIS bellows leakage)
5. **Phoebe** — 27 chunks
6. **Canopus** — 22 chunks (governor failure, scrubber inspection)
7. **Cleo** — 18 chunks (air compressor belt, BWTS PLC malfunction)
8. **Ajax** — 17 chunks
9. **Aphrodite** — 16 chunks
10. **Lupus** — 8 chunks
11. **Antiope** — 5 chunks
12. **Atlas** — 5 chunks

### Q: Who are the most active external parties in fleet communications?

Top external parties by email volume:

- **technical@marantankers.gr** — 71 emails (primary internal technical team)
- **spares@angelicoussisgroup.com** — 11 emails (group spare parts procurement)
- **sales@seascapetec.com** — 8 emails (equipment vendor)
- **noreply.maritime@dnv.com** — 6 emails (DNV classification/certification)
- **info@sealegend.com.cn** — 4 emails (Chinese supplier)
- **support@sl-sail.com** — 4 emails (maritime service provider)
- **noreply@angelicoussisgroup.com** — 4 emails (group notifications)

Plus 104 unique email domains representing vendors, port agents, classification societies, and suppliers across Korea, Singapore, Greece, China, and more.

### Q: What are the top operational concerns across the fleet right now?

Based on topic coverage and severity:

1. **Equipment failures** — multiple vessels with control system, radar, gas detection, and compressor failures requiring urgent parts and service attendance
2. **EGCS/scrubber integrity** — fleet-wide internal inspection program across 4 vessel classes with confirmed leakage on THETIS
3. **Spare parts procurement bottlenecks** — several urgent POs, one rejected PO blocking invoice processing, 6-8 week lead times on critical valves
4. **IT/connectivity** — Smartship cloud connectivity issues affecting data transmission to AWS
5. **Compliance/certification** — ongoing Lloyd's Register surveys, PSC deficiency advisories on lifesaving equipment, transit fee increases
