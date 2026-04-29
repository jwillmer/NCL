# Vessel-metadata mismatch report

_Generated 2026-04-29 20:08 UTC from `mtss validate ingest` check #36._

## Corpus snapshot

- Documents scanned: **17,275**
- Chunks scanned: **43,618**
- Chunks already tagged with `vessel_ids`: **24,598** (56.4%)
- Distinct canonical vessel UUIDs in use: **50**
- Canonical names + aliases loaded from `data/vessel-list.csv`: **58**

## Findings summary

| Bucket | Unique mentions | Total occurrences |
|---|---:|---:|
| Likely typos (review + add to alias / fix) | 55 | 137 |
| Likely missing from register (review + register) | 154 | 319 |
| Signature-block concatenations (extractor noise) | 7 | 160 |
| `MARAN X` org/department tokens (extractor noise) | 5 | 69 |
| Common phrases caught by `M.[TV]` regex (extractor noise) | 19 | 237 |
| Other (people names, unclear) | 58 | 123 |
| **TOTAL** | **298** | **1045** |

## Likely typos — fix at source or add as alias

| Mention | Likely canonical | Occurrences | Sample emails |
|---|---|---:|---|
| `MARAN ARIES` | `MARAN ARES` | 15 | `91519066_uvcmde03.jyv.eml`<br>`100326213_d5p5uusp.axx.eml`<br>`86541117_0kvsxlij.4pf.eml` |
| `MARAN ASIA` | `MARAN ASPASIA` | 8 | `91507649_4w3zt2bf.o5w.eml`<br>`99961870_lsawzrjw.2jw.eml`<br>`99960360_n2dzyz3l.rzb.eml` |
| `MARAN HERCUES` | `MARAN HERCULES` | 6 | `90957502_mzbhpwuh.z0c.eml`<br>`90967123_23feryji.d1b.eml` |
| `MARAN ORPEHUS` | `MARAN ORPHEUS` | 6 | `98235822_nvutvjlk.ify.eml`<br>`95937252_12lpnkmb.age.eml`<br>`95936108_mx5ep1yb.bxk.eml` |
| `MARAN TRUST` | `MARAN TAURUS` | 6 | `95684494_i1xrtj15.nj4.eml`<br>`95685649_1klptlpt.oog.eml` |
| `MARAN POSEIDON AND` | `MARAN POSEIDON` | 6 | `98201556_sjdkrx0f.zb4.eml`<br>`98202008_ut340r2q.uq5.eml`<br>`98206509_aep4dvcr.jbz.eml` |
| `MARAN ORHEUS` | `MARAN ORPHEUS` | 5 | `95768969_hucvwepz.iea.eml`<br>`95764583_2yzsv4fl.r3k.eml`<br>`95780910_c0xzrhiz.eqb.eml` |
| `MARAN MARIA` | `MARAN MIRA` | 5 | `90957502_mzbhpwuh.z0c.eml`<br>`90967123_23feryji.d1b.eml`<br>`91141626_azk2brcd.oin.eml` |
| `MARAN APRHRODITE` | `MARAN APHRODITE` | 4 | `87077045_25mfzvfv.1km.eml` |
| `MARAN HELLEN` | `MARAN HELEN` | 4 | `91514677_kpkghq4q.4jf.eml` |
| `MARAN APO` | `MARAN APOLLO` | 4 | `99963398_wcwyqok2.1mu.eml`<br>`99965424_rqmq0xmo.0pg.eml` |
| `MARAN TAURUS DUE` | `MARAN TAURUS` | 3 | `95742422_3ehkto5u.hmr.eml` |
| `MARAN ANTIOP` | `MARAN ANTIOPE` | 3 | `95746798_k340x50x.qr0.eml` |
| `MARAN ARUCTURUS` | `MARAN ARCTURUS` | 3 | `95926784_lcfbjykv.krr.eml` |
| `MARAN ARIDNE` | `MARAN ARIADNE` | 3 | `96241419_c2d3mcbg.5dm.eml`<br>`90947043_ksenssfx.muc.eml` |
| `MARAN THE` | `MARAN THETIS` | 3 | `100308826_ugf00cmk.fed.eml` |
| `MARAN HERCULE` | `MARAN HERCULES` | 3 | `91511864_yu0c5sin.htc.eml`<br>`91512378_c2hfdhfx.bra.eml` |
| `MARAN SOLO` | `MARAN SOLON` | 3 | `95775373_5urcptw2.zhn.eml` |
| `MARAN ASPASISA` | `MARAN ASPASIA` | 2 | `87067286_zfulk3fa.44m.eml`<br>`95781186_5qbpk21m.u4o.eml` |
| `MARAN HELIO` | `MARAN HELIOS` | 2 | `87077427_0smdm0ly.4q1.eml`<br>`87077741_qso2mmb5.rgr.eml` |
| `MARAN HERMIOME` | `MARAN HERMIONE` | 2 | `90940747_mwtt2ix3.aw1.eml`<br>`90944639_y3gapyq2.ief.eml` |
| `MARAN CAPRICORN FROM` | `MARAN CAPRICORN` | 2 | `91151096_ttrdr4x0.kfn.eml` |
| `MARAN CANOPUS FOR` | `MARAN CANOPUS` | 2 | `91498381_i0xzxggz.lpo.eml` |
| `MARAN HEL` | `MARAN HELEN` | 2 | `91509548_iosdmeoq.ef1.eml` |
| `MARAN HERCUELES` | `MARAN HERCULES` | 2 | `95672339_xbtwh1ie.b05.eml`<br>`95755586_h1klavck.w3m.eml` |
| `MARAN ACTURUS` | `MARAN ARCTURUS` | 2 | `95693131_lqe1lymm.v2o.eml`<br>`95693324_elg0bfia.a42.eml` |
| `MARAN LUPW` | `MARAN LUPUS` | 2 | `95703775_eveddj1o.ks3.eml` |
| `MARAN HERMIONES` | `MARAN HERMIONE` | 2 | `98213639_gynu4kvf.gsq.eml`<br>`98213735_vvsl4yyj.jlz.eml` |
| `MARAN THEMS` | `MARAN THETIS` | 1 | `100324373_bgevmsly.xmi.eml` |
| `MARAN SOL` | `MARAN SOLON` | 1 | `91520459_y4i0be2r.i0f.eml` |
| `MARAN ATALATA` | `MARAN ATALANTA` | 1 | `95722178_5vfuxq1k.rmc.eml` |
| `MARAN APPOLO` | `MARAN APOLLO` | 1 | `100304571_f5oepjpm.130.eml` |
| `MARAN THETISC` | `MARAN THETIS` | 1 | `90929895_xqywrzmy.goa.eml` |
| `MARAN THETHIS` | `MARAN THETIS` | 1 | `90957562_dhe0sryx.tun.eml` |
| `MARAN TAURS` | `MARAN TAURUS` | 1 | `90965260_q1tvezhm.vbp.eml` |
| `MARAN ATALANTA MNC` | `MARAN ATALANTA` | 1 | `91131709_wkj15w20.idf.eml` |
| `MARAN CANOPUS ABT` | `MARAN CANOPUS` | 1 | `91181417_cqq5kv3q.g3o.eml` |
| `MARAN ATLANTA` | `MARAN ATALANTA` | 1 | `95756416_kztv1h0d.iio.eml` |
| `MARAN THETES` | `MARAN THETIS` | 1 | `90951760_dl1ywmoh.dwg.eml` |
| `MARAN ORPH` | `MARAN ORPHEUS` | 1 | `95782352_52uczzwg.ew5.eml` |
| `MARAN ORPHEU` | `MARAN ORPHEUS` | 1 | `95782352_52uczzwg.ew5.eml` |
| `MARAN TAUROS` | `MARAN TAURUS` | 1 | `95938362_d00i4nzd.ac5.eml` |
| `MARAN TAJUS` | `MARAN TAURUS` | 1 | `95938362_d00i4nzd.ac5.eml` |
| `MARAN TAJRUS` | `MARAN TAURUS` | 1 | `95938362_d00i4nzd.ac5.eml` |
| `MARAN ARAIDNE` | `MARAN ARIADNE` | 1 | `98178489_neolsuti.hwh.eml` |
| `MARAN ARCTURUS WAS` | `MARAN ARCTURUS` | 1 | `98206776_srsudlqf.g1y.eml` |
| `MARAN DION` | `MARAN DIONE` | 1 | `98251825_tixx1trh.w0j.eml` |
| `MARAN PHPOEBE` | `MARAN PHOEBE` | 1 | `98258258_qayckz5u.ndm.eml` |
| `MARAN ATALAN` | `MARAN ATALANTA` | 1 | `98631438_xiswava2.0nr.eml` |
| `MARAN OPRHEUS` | `MARAN ORPHEUS` | 1 | `98253788_jloeknmb.pqx.eml` |
| `MARAN PYTHI` | `MARAN PYTHIA` | 1 | `99959378_23a1lvro.0zz.eml` |
| `MARAN PHOEBE DUE` | `MARAN PHOEBE` | 1 | `99954731_ulvfl5z3.2ln.eml` |
| `MARAN ARTEMIS IMO` | `MARAN ARTEMIS` | 1 | `99978422_22dile10.j2p.eml` |
| `MARAN ANTIO` | `MARAN ANTIOPE` | 1 | `99994954_b2qo25af.5dp.eml` |
| `MARAN CLED` | `MARAN CLEO` | 1 | `99999374_3hrq1zof.dc3.eml` |

## Likely missing from register — review and add

| Mention | Occurrences | Sample emails |
|---|---:|---|
| `PEGASUS VOYAGER` | 7 | `100323275_uevum4um.nxl.eml`<br>`100325397_xoh00lif.zfy.eml`<br>`99957420_eofaicwi.nab.eml` |
| `MARAN MIM` | 7 | `99959378_23a1lvro.0zz.eml` |
| `MARAN CANOPUS DESLOPPING` | 6 | `95703369_1c4ilttl.wfz.eml`<br>`95703520_onzczd21.nwm.eml` |
| `MARAN VESSEL` | 6 | `100297212_mw0fear0.oqb.eml`<br>`100326401_lrqpo43o.1iu.eml`<br>`100338270_q0xjevwz.sj0.eml` |
| `POLARIS VOYAGER` | 6 | `100323275_uevum4um.nxl.eml`<br>`100324557_tyqcijxy.5ik.eml`<br>`100325397_xoh00lif.zfy.eml` |
| `MARAN TRANSPORTER` | 6 | `90967207_g2wpd1ea.jgg.eml`<br>`91522248_zb2lw1yx.ehb.eml`<br>`91523215_g23bzjth.bdp.eml` |
| `MARAN SOPHIA` | 5 | `90968463_2twnvtkn.uh3.eml`<br>`90961140_kaokmi4o.plh.eml`<br>`95701684_nfumyt1v.xsu.eml` |
| `MARAN APOLLO DURING` | 5 | `100297478_3vrkfidk.zsg.eml`<br>`100304571_f5oepjpm.130.eml`<br>`100305462_k3a5xbqo.v25.eml` |
| `MARAN HELEN FROM` | 5 | `100322957_dzti3b02.own.eml`<br>`90953998_5thi4bm1.o1l.eml`<br>`91152190_gzwtsulb.dma.eml` |
| `MARAN CLEO FROM` | 5 | `90951382_r4lp4c45.x52.eml`<br>`95929668_hxnv1y14.fk0.eml` |
| `MARAN SHIPS` | 5 | `90952518_3mjdjxcb.ksd.eml`<br>`90955534_saziz0wd.nzq.eml`<br>`90956984_5iogkb4j.5ns.eml` |
| `MARAN HELIOS SHIP` | 5 | `91473924_jfx1szzn.p2d.eml`<br>`91515644_oaygiza2.5yu.eml`<br>`91515750_y2eipwp2.wxv.eml` |
| `MARAN HELEN FRIDAY` | 4 | `91152190_gzwtsulb.dma.eml` |
| `MARAN WILL` | 4 | `87076565_z354hmil.gx3.eml`<br>`95938664_xt3aor2g.ugs.eml`<br>`99998810_3ct0frag.335.eml` |
| `MARAN ORPHEUS FROM` | 4 | `90952646_nk2bmnha.4yl.eml`<br>`100343050_a450qy1k.gws.eml` |
| `MARAN ORDER` | 4 | `90953834_15uuclud.13b.eml`<br>`90954158_bqise25x.qxb.eml`<br>`90954622_kfov3fer.qpw.eml` |
| `MARAN MIRA FROM` | 4 | `90953456_1kymsofh.53d.eml`<br>`99996100_ieoee2dc.tme.eml` |
| `MARAN LUPUS FROM` | 4 | `90953140_amsnmjc5.1is.eml`<br>`91152230_hc1vyjhl.nnj.eml` |
| `MARAN PHOEBE FROM` | 4 | `90953818_fr0dof5t.dii.eml`<br>`95674009_bp4eoz2v.5uh.eml` |
| `MARAN HERMIONE FROM` | 4 | `91261085_lbcs4k5u.iux.eml`<br>`91260328_1gdlbass.mkw.eml` |
| `MARAN PORTFOLIO` | 4 | `91521830_nbo3vycu.j21.eml`<br>`91522332_z33ka0ms.emh.eml`<br>`91538824_iomkertq.oow.eml` |
| `MARAN DWT` | 4 | `95780608_siw4eji1.cfo.eml`<br>`95781614_0dg3g303.1he.eml` |
| `ARCTURUS VOYAGER` | 4 | `99992785_0g3qnbic.kmv.eml`<br>`99993667_o3o2fte3.kow.eml` |
| `MARAN PLATO CREW` | 3 | `86902740_owzlfki2.cpf.eml` |
| `MARAN THETIS TECHNICAL` | 3 | `95695688_nzwo4wtp.lyq.eml` |
| `MARAN LYRA SERVICEREPORT` | 3 | `96233382_d44pdvox.10z.eml` |
| `MARAN MIRA CODES` | 3 | `100300040_yd0rps4m.4mj.eml`<br>`100300604_b43kcgq3.djc.eml`<br>`100301737_45mluifw.dw5.eml` |
| `MARAN ACCORDINGLY` | 3 | `100301698_5adlt3tu.2a2.eml`<br>`100341988_jhti4xg2.bdo.eml`<br>`100353243_f01i4jrm.cyp.eml` |
| `MARAN APOLLO SHIP` | 3 | `100316806_u2yz5yme.joq.eml`<br>`99959786_nwfav4yu.tsn.eml`<br>`99975113_4k3ccbwm.e30.eml` |
| `MARAN PLATO FROM` | 3 | `100351512_tc43rsvn.04w.eml`<br>`90957516_e0l2qaap.fsy.eml` |
| `MARAN SIDE` | 3 | `86479418_nynhjjzx.rwi.eml`<br>`86547747_w43hibyf.yai.eml`<br>`98237125_etfwqdff.4zt.eml` |
| `MARAN SITE` | 3 | `87077103_tbzgugpr.d3x.eml`<br>`95938000_qf1prmfa.r0c.eml` |
| `MARAN ARTEMIS FROM` | 3 | `90935199_mxqxq3e5.rnn.eml`<br>`98949808_t0o1tqar.wec.eml` |
| `MARAN AJAX FROM` | 3 | `90943179_cw0xii5k.q30.eml`<br>`99986866_2lwurkat.5ix.eml` |
| `MARAN ARETE FROM` | 3 | `90940609_rgdh3x3y.tmo.eml`<br>`99997464_dmwr1dgz.hnz.eml` |
| `MARAN MARS FROM` | 3 | `90930145_pvx5zay5.m4o.eml`<br>`99990702_b5o4vewd.qk4.eml` |
| `LAST VOYAGER` | 3 | `90950550_fu4ea0oq.55t.eml`<br>`91261085_lbcs4k5u.iux.eml`<br>`91260328_1gdlbass.mkw.eml` |
| `MARAN LIBRA FROM` | 3 | `90953198_wsfnll4e.dzf.eml` |
| `MARAN LOYALTY` | 3 | `90967207_g2wpd1ea.jgg.eml`<br>`91522248_zb2lw1yx.ehb.eml` |
| `MARAN ENDEAVOUR` | 3 | `91522248_zb2lw1yx.ehb.eml`<br>`95742908_apa1ifh5.0ye.eml` |
| `MARAN LEO AND` | 3 | `95678330_chlmfoo2.1ru.eml`<br>`95675092_cffqhhxl.d5d.eml`<br>`95693436_a0dezcnt.xui.eml` |
| `MARAN PENELOPE SUPPLY` | 2 | `96229412_3nmpvr2s.kue.eml` |
| `MARAN AND` | 2 | `91518325_xgx0ydls.lhf.eml`<br>`95768371_w2jcnuvx.uih.eml` |
| `MARAN TAURUS SHIP` | 2 | `100319688_5a0htwdt.d2k.eml` |
| `MARAN MERCHANT` | 2 | `100340595_kvcgcofr.ez5.eml`<br>`90967207_g2wpd1ea.jgg.eml` |
| `MARAN SAILOR` | 2 | `100340595_kvcgcofr.ez5.eml`<br>`90937593_hbkqpvbp.t0x.eml` |
| `MARAN INSTRUCTIONS` | 2 | `87076565_z354hmil.gx3.eml`<br>`95938664_xt3aor2g.ugs.eml` |
| `MARAN DAY` | 2 | `86856560_ny4yfcfw.3bv.eml` |
| `MARAN PURCHASING` | 2 | `87067028_xdd0fdss.i10.eml` |
| `MARAN HERCULES SHIP` | 2 | `90937609_4ifclhwv.1yl.eml` |
| `MARAN ANTARES SHIPS` | 2 | `90938721_t2rxaz5n.ky5.eml`<br>`90939801_v3uaq5bj.eu2.eml` |
| `MARAN ATHENA FROM` | 2 | `90943111_d4k5kp3n.kb3.eml` |
| `MARAN RETROFITS` | 2 | `90950348_vtvttjau.00s.eml`<br>`90968469_idmvgzlm.qab.eml` |
| `MARAN SOLON FROM` | 2 | `90947607_wvvwx1zh.e3d.eml` |
| `MARAN ANTIOPE FROM` | 2 | `90947297_n2cdhkt4.dt2.eml` |
| `MARAN POSEIDON FROM` | 2 | `90951224_apadpjbd.fjj.eml` |
| `MARAN ANTARES FROM` | 2 | `90952006_1omxotto.chc.eml`<br>`99991102_gv0mrxbg.rma.eml` |
| `MARAN ORPHEUS MAINTENANCE` | 2 | `90952646_nk2bmnha.4yl.eml`<br>`90968277_dqsfxryz.5gx.eml` |
| `MARAN APOLLO FROM` | 2 | `90953712_df114zhw.5uw.eml`<br>`91143054_dib3vii2.x0d.eml` |
| `MARAN ITEMS` | 2 | `90957502_mzbhpwuh.z0c.eml`<br>`90967123_23feryji.d1b.eml` |
| `MARAN ASPASIA FROM` | 2 | `90961680_furvscwb.keg.eml` |
| `MARAN VISION` | 2 | `90967207_g2wpd1ea.jgg.eml`<br>`91522248_zb2lw1yx.ehb.eml` |
| `MARAN DYNASTY` | 2 | `90967207_g2wpd1ea.jgg.eml`<br>`91522248_zb2lw1yx.ehb.eml` |
| `MARAN GUARDIAN` | 2 | `90967207_g2wpd1ea.jgg.eml`<br>`91522248_zb2lw1yx.ehb.eml` |
| `MARAN LEO FROM` | 2 | `91517567_icjeb4yl.w12.eml` |
| `MARAN MAIL` | 2 | `91512018_dl0wrx51.02d.eml`<br>`91524926_zodz4wje.mtg.eml` |
| `MARAN PLATO NON` | 2 | `91520175_pybgxezf.udc.eml`<br>`91520439_zsz5bgp2.4l0.eml` |
| `ANTARES VOYAGER` | 2 | `91530178_vgz4e5sr.qea.eml` |
| `MARAN DIONE FOR` | 2 | `95675092_cffqhhxl.d5d.eml`<br>`95693436_a0dezcnt.xui.eml` |
| `MARAN TANKERPS` | 2 | `95694282_22vn4ozd.onm.eml`<br>`95695004_l4narbsf.rjt.eml` |
| `MARAN ARIADNE FROM` | 2 | `91153914_gpm2nibb.i01.eml` |
| `MARAN ARCTURUSCONSIGNEE` | 2 | `95700209_gitnnx1c.1c1.eml`<br>`95704357_u2zmdho5.td5.eml` |
| `MARAN ARES FROM` | 2 | `90939791_ikqeq2py.yjn.eml` |
| `MARAN LUA` | 2 | `95703775_eveddj1o.ks3.eml` |
| `MARAN PYTHIA FROM` | 2 | `90951232_rz4ttwra.cjm.eml`<br>`99995876_trpkbqlq.g3e.eml` |
| `MARAN ATALANTA FROM` | 2 | `95932342_epcruqwt.3i3.eml`<br>`95932338_kf2ytph5.da4.eml` |
| `MARAN TAJROS` | 2 | `95938362_d00i4nzd.ac5.eml` |
| `MARAN CONFIRMATION` | 2 | `95938664_xt3aor2g.ugs.eml` |
| `MARAN HERMES FROM` | 2 | `98226424_oqtfnqat.vol.eml` |
| `MARAN TYPE` | 2 | `98224079_zin4cx4w.ert.eml`<br>`99996058_gzpq4hl2.lo3.eml` |
| `MARAN THALEIA FROM` | 2 | `99999612_cx1lc0np.ngu.eml` |
| `MARAN LUPUS AND` | 1 | `91500045_cxtkwdqw.qp4.eml` |
| `MARAN GLORY` | 1 | `100310324_2st2ln2c.flt.eml` |
| `MARAN GLORY DOIRANIS` | 1 | `100310324_2st2ln2c.flt.eml` |
| `MARAN ARETE ARRIVED` | 1 | `100321465_gs4j1i5m.saj.eml` |
| `MARAN CYGNUS` | 1 | `100326213_d5p5uusp.axx.eml` |
| `MARAN CASTOR` | 1 | `100326213_d5p5uusp.axx.eml` |
| `MARAN GEMINI` | 1 | `100326213_d5p5uusp.axx.eml` |
| `MARAN ANDROMEDA` | 1 | `100326213_d5p5uusp.axx.eml` |
| `MARAN CARINA` | 1 | `100326213_d5p5uusp.axx.eml` |
| `MARAN AQUARIUS` | 1 | `100326213_d5p5uusp.axx.eml` |
| `MARAN CALLISTO` | 1 | `100326213_d5p5uusp.axx.eml` |
| `MARAN CASSIOPEIA` | 1 | `100326213_d5p5uusp.axx.eml` |
| `MARAN CORONA` | 1 | `100326213_d5p5uusp.axx.eml` |
| `MARAN TRITON` | 1 | `100326213_d5p5uusp.axx.eml` |
| `MARAN REGULUS` | 1 | `100326213_d5p5uusp.axx.eml` |
| `MARAN SAGITTA` | 1 | `100326213_d5p5uusp.axx.eml` |
| `MARAN HAPPINESS` | 1 | `100340595_kvcgcofr.ez5.eml` |
| `MARAN ASTRONOMER` | 1 | `100340595_kvcgcofr.ez5.eml` |
| `MARAN VOYAGER` | 1 | `100340595_kvcgcofr.ez5.eml` |
| `MARAN HERMESIMPORTANT` | 1 | `87077725_jroh3vyl.2nv.eml` |
| `MARAN DIONE FREEPORT` | 1 | `87077819_2amtnugo.abr.eml` |
| `MARAN DANAE SINGAPORE` | 1 | `87077819_2amtnugo.abr.eml` |
| `MARAN CANOPUS SHANGHAI` | 1 | `86557401_kqsti1uu.g3r.eml` |
| `MARAN CANOPUS SHIP` | 1 | `86557401_kqsti1uu.g3r.eml` |
| `MARAN PLATO WILL` | 1 | `86877907_agmoxn5g.xls.eml` |
| `MARAN HERMIONECONSIGNEE` | 1 | `90933407_aka2yc3t.qag.eml` |
| `MARAN PENELOPE PIRAEUS` | 1 | `90936237_jrjhdmw5.xme.eml` |
| `MARAN LIBRA SHIP` | 1 | `90937863_14fai2q5.t3m.eml` |
| `MARAN HELIOS PIRAEUS` | 1 | `90945077_3c4q5fug.200.eml` |
| `MARAN PENELOPE FROM` | 1 | `90950550_fu4ea0oq.55t.eml` |
| `MARAN VIRTUE` | 1 | `90967207_g2wpd1ea.jgg.eml` |
| `MARAN ATALANTA SHIP` | 1 | `91131709_wkj15w20.idf.eml` |
| `MARAN ATALANTACONSIGNEE` | 1 | `91131709_wkj15w20.idf.eml` |
| `MARAN GOOD` | 1 | `91129202_oty1bqsr.jtf.eml` |
| `MARAN NON` | 1 | `91495651_tl5kqtvo.doc.eml` |
| `MARAN HERCULESCONSIGNEE` | 1 | `91515378_pfwsrhxo.ir2.eml` |
| `MARAN DANAE FREEPORT` | 1 | `91524738_lnx3r0l1.yzs.eml` |
| `MARAN STOCK` | 1 | `91529044_n2225plh.3f5.eml` |
| `MARAN LUPUS MARAN` | 1 | `95679394_wcxp0v5o.hea.eml` |
| `MARAN DIONE AND` | 1 | `95678330_chlmfoo2.1ru.eml` |
| `MARAN HELEN DATE` | 1 | `95689670_siibwdsl.2dg.eml` |
| `MARAN SPARTA` | 1 | `95686170_4xqzflxx.t5h.eml` |
| `MARAN THETIS COMING` | 1 | `95695688_nzwo4wtp.lyq.eml` |
| `MARAN COMMERCIALLY` | 1 | `95768371_w2jcnuvx.uih.eml` |
| `MARAN FOR` | 1 | `95695722_maje3my0.kmv.eml` |
| `MARAN LUPUS SHIP` | 1 | `95765319_1glhdayf.1vk.eml` |
| `MARAN ARCTURUS SHIP` | 1 | `95765319_1glhdayf.1vk.eml` |
| `MARAN CANOPUS MAXFREIGHT` | 1 | `95772448_abevkoaa.iz3.eml` |
| `MARAN CANOPUS FROM` | 1 | `90944319_ngndopcx.54c.eml` |
| `LIBRA VOYAGER` | 1 | `95701776_v3j5tdpj.45n.eml` |
| `MARAN TAUA` | 1 | `90948177_5jbxxvcc.tr0.eml` |
| `MARAN HOMER FROM` | 1 | `90954498_tbugpuig.rch.eml` |
| `MARAN HERCULES FROM` | 1 | `90948317_tdxzu24z.phe.eml` |
| `MARAN LYNX FROM` | 1 | `90947959_4qim40ox.2gl.eml` |
| `MARAN HELIOS FROM` | 1 | `90947069_kiohqoja.skb.eml` |
| `MARAN MTM` | 1 | `95775373_5urcptw2.zhn.eml` |
| `MARAN TENKERS` | 1 | `95939120_ogctdbmk.iag.eml` |
| `MARAN MANAGEMEM` | 1 | `95938362_d00i4nzd.ac5.eml` |
| `MARAN POSSIBILITY` | 1 | `96226347_umtcq342.gwa.eml` |
| `MARAN ANTONIS` | 1 | `95938664_xt3aor2g.ugs.eml` |
| `MARAN MARS IMO` | 1 | `98211315_4tfdve0e.as3.eml` |
| `MARAN ATLAS CONSIGNED` | 1 | `98223977_0d03xf0q.uhe.eml` |
| `MARAN PLATO RECORD` | 1 | `98226342_yfpmlpu4.u1y.eml` |
| `MARAN PLATO INSTRUCTOR` | 1 | `98226342_yfpmlpu4.u1y.eml` |
| `MARAN ANTARES MASTER` | 1 | `98226864_u05f3fjj.ihe.eml` |
| `MARAN MANAGEMENT` | 1 | `98230870_31gq5kol.kqx.eml` |
| `MARAN WAREHOUSE` | 1 | `98260082_zok243pn.i54.eml` |
| `MARAN TANUGR` | 1 | `98260391_i3w20a2h.kd3.eml` |
| `MARAN ORPHEUS SWIFT` | 1 | `99729434_mmj5hoid.5oc.eml` |
| `MARAN ORPHEUS FORWARDING` | 1 | `99954551_cznk5f4f.34k.eml` |
| `MARAN ARCTURUS PLEASE` | 1 | `99727409_ubr1om1z.jeu.eml` |
| `MARAN HELEN SITE` | 1 | `99994664_nroj2110.hnc.eml` |
| `MARAN THETIS FROM` | 1 | `99995918_fu4fwjdh.uds.eml` |

## Signature-block concatenations — extractor noise

| Mention | Likely canonical | Occurrences | Sample emails |
|---|---|---:|---|
| `MARAN ORPHEUS GOOD` | `MARAN ORPHEUS` | 55 | `91514962_bo3fsqi3.0cv.eml`<br>`95714474_wd31pyr1.q3s.eml`<br>`95769173_mxmq4jvc.ndz.eml` |
| `MARAN ORPHEUS KALISPERA` | `MARAN ORPHEUS` | 53 | `90943551_ax0inyae.ztr.eml`<br>`90945443_gbr4xn4f.upt.eml`<br>`90968180_fr1exnqr.bnk.eml` |
| `MARAN PLATO TEL` | `MARAN PLATO` | 27 | `100307264_br33z14a.q0g.eml`<br>`90944923_ivjkndmk.3vh.eml`<br>`90964598_mbybu5if.4k4.eml` |
| `MARAN HERCULES CAPT` | `MARAN HERCULES` | 14 | `91511864_yu0c5sin.htc.eml`<br>`91512378_c2hfdhfx.bra.eml`<br>`95663398_a2afgxk4.mxs.eml` |
| `MARAN HELEN ETA` | `MARAN HELEN` | 7 | `86479007_cxvtaqky.qyu.eml` |
| `MARAN LYRA ETA` | `MARAN LYRA` | 3 | `90932965_1yfhgag2.ji0.eml` |
| `MARAN CLEO ETA` | `MARAN CLEO` | 1 | `99953199_qmglowz5.rt2.eml` |

## `MARAN X` org/department tokens — extractor noise

| Mention | Occurrences | Sample emails |
|---|---:|---|
| `MARAN VESSELS` | 26 | `100297212_mw0fear0.oqb.eml`<br>`100326401_lrqpo43o.1iu.eml`<br>`100330655_vtd1qzog.hza.eml` |
| `MARAN HAS` | 14 | `100312572_m4c200ls.u1a.eml`<br>`100334814_ekakn245.qah.eml`<br>`100336188_ih0gj5aw.ern.eml` |
| `MARAN SUPERINTENDENT` | 11 | `90926246_pqli15tz.hvy.eml`<br>`90942339_pu34mt40.hyh.eml`<br>`90948219_2u3wx0p5.4y4.eml` |
| `MARAN TEAM` | 10 | `100298793_rtlnko2x.rum.eml`<br>`100312572_m4c200ls.u1a.eml`<br>`87076565_z354hmil.gx3.eml` |
| `MARAN LIBRA SHIPS` | 8 | `90937863_14fai2q5.t3m.eml` |

## Common phrases via `M.[TV]` regex — extractor noise

| Mention | Occurrences | Sample emails |
|---|---:|---|
| `PORT LNG TANK` | 117 | `100312898_getnarkk.nzy.eml`<br>`90939839_5wg1qxih.1hp.eml`<br>`100356459_xhok0ymv.12r.eml` |
| `SUMMER DRAFT` | 43 | `100292998_icdk5bah.geu.eml`<br>`87077087_kcfoja4j.tdc.eml`<br>`90926620_05ub4kcm.l4k.eml` |
| `CALCULATED WEIGHT` | 14 | `95934010_sfbrvjvu.fpv.eml`<br>`95934270_upszqdou.oxa.eml`<br>`96234217_tx4bwpox.ibh.eml` |
| `TOTAL LNG CONSUMPTION` | 13 | `100312898_getnarkk.nzy.eml`<br>`90939839_5wg1qxih.1hp.eml`<br>`100356459_xhok0ymv.12r.eml` |
| `BEST REGARDS` | 9 | `87066826_oggyfjg1.1ln.eml`<br>`90967545_n3y00z3h.40z.eml`<br>`95670338_i0cmfdqo.03c.eml` |
| `LOW SULPHUR` | 9 | `90967545_n3y00z3h.40z.eml`<br>`90947373_2ojgbicl.thb.eml`<br>`90951760_dl1ywmoh.dwg.eml` |
| `HSFO PRICE` | 7 | `100322827_keawevuh.0bf.eml`<br>`91524738_lnx3r0l1.yzs.eml`<br>`95713882_zrkw51aa.vgr.eml` |
| `CALCULATED WEIGHT CALCULATED` | 4 | `95934270_upszqdou.oxa.eml`<br>`96234217_tx4bwpox.ibh.eml`<br>`96234389_uklxcfya.ljs.eml` |
| `THE CALCULATED QUANTITIES` | 4 | `95934270_upszqdou.oxa.eml`<br>`96234217_tx4bwpox.ibh.eml`<br>`96234389_uklxcfya.ljs.eml` |
| `LSMGO PRICE` | 3 | `100322827_keawevuh.0bf.eml`<br>`99956306_1d2oraai.40e.eml`<br>`99993781_loleezkd.gwq.eml` |
| `VERY LOW SULPHUR` | 3 | `90967545_n3y00z3h.40z.eml`<br>`90947373_2ojgbicl.thb.eml`<br>`90951760_dl1ywmoh.dwg.eml` |
| `KINDEST REGARDS` | 2 | `100337853_mx5judtx.5g3.eml`<br>`90926580_tjk3picc.md0.eml` |
| `CALCULATED MASS` | 2 | `95934010_sfbrvjvu.fpv.eml`<br>`98224922_qnzrhccr.vz2.eml` |
| `CALCULATED WEIGHT THE` | 2 | `95934010_sfbrvjvu.fpv.eml`<br>`98224922_qnzrhccr.vz2.eml` |
| `BEST REGARDS CHIEF` | 1 | `100342032_tpiroj5l.elg.eml` |
| `LNG CONSUMED` | 1 | `90938073_3x5da51y.dlm.eml` |
| `LNG BUNKERING` | 1 | `98226666_2nft3gzn.jej.eml` |
| `OVER ITS HSFO` | 1 | `99956306_1d2oraai.40e.eml` |
| `OVER THE PORT` | 1 | `99956306_1d2oraai.40e.eml` |

## Other (people names, unclassified)

| Mention | Occurrences | Sample emails |
|---|---:|---|
| `ANTONIS L. ANGELICOUSSIS` | 16 | `95938664_xt3aor2g.ugs.eml` |
| `SOPHIA GOOD DAY` | 13 | `100315064_db4gm2ha.ymi.eml`<br>`100318009_fafa3lgy.w5d.eml`<br>`100317991_u5lye4ba.lg0.eml` |
| `ADDITIONAL PLEASE NOTE` | 5 | `90939839_5wg1qxih.1hp.eml`<br>`90953644_hsw1pqvw.rgo.eml`<br>`90953766_tomg520p.lwi.eml` |
| `SEA WATER` | 5 | `95677043_vbsk2xmj.wt1.eml`<br>`95679274_3ekdcldl.mih.eml`<br>`95678588_3lvctmot.jhk.eml` |
| `LOAD LINE` | 4 | `90953222_k1i2p4i2.5zs.eml`<br>`90953428_23updpzp.o0n.eml`<br>`90961588_nynymuc4.hhy.eml` |
| `WTI MIDLAND CRUDE` | 4 | `91514172_125qqdwe.fea.eml`<br>`91515460_4klrkin4.kdz.eml`<br>`91521128_ufoy1tx4.rm1.eml` |
| `SOPHIA FROM` | 4 | `95756314_vgtgtc0e.slw.eml`<br>`99989798_34rr12kz.sjv.eml` |
| `ROB GRADE` | 4 | `96230764_mii2keub.5zf.eml`<br>`96230770_dtz4zqkw.ufu.eml`<br>`98209541_15sxbndt.wkk.eml` |
| `BUNKER ROB HFO` | 4 | `98201022_lccootem.ycb.eml`<br>`98239718_semivy02.fsn.eml`<br>`99960228_tlgvq4d3.2xj.eml` |
| `VDC MIN` | 3 | `95702811_wrrwfhru.eqk.eml`<br>`95722174_qxuhrvxr.gqv.eml` |
| `WHILE SHIP` | 3 | `96226347_umtcq342.gwa.eml` |
| `ROB SAMPLE ATTACHED` | 3 | `96230764_mii2keub.5zf.eml`<br>`96230770_dtz4zqkw.ufu.eml`<br>`98228808_plwmn42z.dmc.eml` |
| `AND VISUAL INSPECTION` | 3 | `99993357_obigx4zc.hrq.eml`<br>`99996060_yno512zt.c5a.eml`<br>`99996388_umb3ytjb.urd.eml` |
| `EAGLE HATTERAS` | 2 | `87067438_ieqj3obm.4dn.eml`<br>`87067436_qpa0pqip.1m2.eml` |
| `ADVANTAGE AWARD` | 2 | `87067438_ieqj3obm.4dn.eml`<br>`87067436_qpa0pqip.1m2.eml` |
| `INTERNATIONAL LOAD LINE` | 2 | `90951668_wq4scc21.ip5.eml`<br>`99995298_1vppcqwp.ppx.eml` |
| `ENDORSEMENT WHERE THE` | 2 | `90951668_wq4scc21.ip5.eml`<br>`99995298_1vppcqwp.ppx.eml` |
| `ETHANE SAPPHIRE` | 2 | `95696678_pv1lzsfv.p45.eml`<br>`95781080_t5egmmjr.dmo.eml` |
| `MONJASA REFORME` | 2 | `98228046_ho1kaqfo.2rm.eml` |
| `MONJASA REFORMER` | 2 | `98228046_ho1kaqfo.2rm.eml` |
| `WHILE CELL` | 1 | `100333421_ah2cgumh.shi.eml` |
| `SWL PUMP ROOM` | 1 | `100301436_4bdxonne.xx0.eml` |
| `SWL BOSUN STORE` | 1 | `100301436_4bdxonne.xx0.eml` |
| `SWL PEDESTAL MOUNTED` | 1 | `100301436_4bdxonne.xx0.eml` |
| `MARAIN AJAX` | 1 | `100315779_axuw5pqn.40u.eml` |
| `ESTIMATED PAL SCHEDULE` | 1 | `100325397_xoh00lif.zfy.eml` |
| `ROB AFTER BUNKERING` | 1 | `87066826_oggyfjg1.1ln.eml` |
| `FUEL MET THE` | 1 | `87066826_oggyfjg1.1ln.eml` |
| `DRAUGHT LOADED` | 1 | `90938849_t0y0yfcx.1tm.eml` |
| `SOPHIA CALLING ROTTERDAM` | 1 | `90969544_fk50jc5d.coz.eml` |
| `SOPHIA BUNSCHOTENWEG` | 1 | `90969544_fk50jc5d.coz.eml` |
| `HIRADO ABT` | 1 | `91181417_cqq5kv3q.g3o.eml` |
| `OCEANIC FORTUNE ABT` | 1 | `91181417_cqq5kv3q.g3o.eml` |
| `DALMA ABT` | 1 | `91181417_cqq5kv3q.g3o.eml` |
| `IOANNA ABT` | 1 | `91181417_cqq5kv3q.g3o.eml` |
| `NEW TINOS ABT` | 1 | `91181417_cqq5kv3q.g3o.eml` |
| `NEW NAXOS ABT` | 1 | `91181417_cqq5kv3q.g3o.eml` |
| `HAKONE ABT` | 1 | `91181417_cqq5kv3q.g3o.eml` |
| `NEW DYNASTY ABT` | 1 | `91181417_cqq5kv3q.g3o.eml` |
| `BOG COMPRESSOR STOPPED` | 1 | `91524660_nyxqffd4.trl.eml` |
| `INPSH LAT` | 1 | `90947373_2ojgbicl.thb.eml` |
| `THE METER` | 1 | `95783095_xtq4p1hf.jku.eml` |
| `SILVER HESSA` | 1 | `98202945_s1o1qhkp.lii.eml` |
| `ATLANTIC ORCHARD` | 1 | `98206904_xedzsdix.wbu.eml` |
| `ROB GRADE ZJQCMQRYFPFPTBANNERSTART` | 1 | `98209541_15sxbndt.wkk.eml` |
| `ANODE AMP` | 1 | `98216667_03xbxrlt.xyq.eml` |
| `MONJASA REFORMER LOCATION` | 1 | `98228046_ho1kaqfo.2rm.eml` |
| `PLEASE ACKNOWLEDGE RECEIPT` | 1 | `98228046_ho1kaqfo.2rm.eml` |
| `MASESRK CLEVLAND REMARK` | 1 | `98237548_e14ccsih.esf.eml` |
| `CHIEF ENGINEER` | 1 | `98239601_fsdtna5u.qbw.eml` |
| `ORPHEUS DELIVERED` | 1 | `99954551_cznk5f4f.34k.eml` |
| `SEAVISION WHICH` | 1 | `99955762_dhryddfy.1un.eml` |
| `NMARAN CAPRICORN ASF` | 1 | `99964680_qqoq2ylv.ke1.eml` |
| `DMA MET THE` | 1 | `99994841_zut20m1m.lto.eml` |
| `TORDIS KNUTSEN` | 1 | `99994954_b2qo25af.5dp.eml` |
| `WINDSDOR KUNTSEN` | 1 | `99994954_b2qo25af.5dp.eml` |
| `MORAM ARDEMS OWNER` | 1 | `99995326_v0xuqtlb.exe.eml` |
| `MORAN ARLENES OWNER` | 1 | `99995326_v0xuqtlb.exe.eml` |

## Method

Generated by check #36 (`_check_unknown_vessel_mentions` in `src/mtss/cli/validate_cmd.py`) against `data/ingest.db`. Candidate regex lives in `src/mtss/processing/vessel_mention_extractor.py`. Categorisation buckets above are post-hoc heuristics: typos use `difflib.get_close_matches` with cutoff 0.85; signature-block concatenations match `MARAN <name> <SIG_TAIL>` where `SIG_TAIL ∈ {GOOD, KALISPERA, TEL, CAPT, …}`. The buckets are advisory — review before acting.

## Suggested next actions

1. **Typos bucket** — fix at the source (correct the email content if it's an internal artifact, or add the typo as an alias on the canonical vessel via `data/vessel-list.csv` ALIASES column).
2. **Missing bucket** — verify with the fleet team whether each is a real vessel; if so, add a row to `data/vessel-list.csv` and re-run ingest's vessel retag (`scripts/retag_vessel_ids_from_db.py`).
3. **Extractor noise** — track in task #20: tighten the M.[TV] regex and add the recurring phrases to `HARDCODED_NOISE`.