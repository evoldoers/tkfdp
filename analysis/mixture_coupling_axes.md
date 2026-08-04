# Coupling-mixture components: axes of correlation/anticorrelation

Correspondence analysis of each fitted joint stationary pi_c (results/mixture_component_char/components_K*.npz). sigma>0 = correlated (like-with-like), sigma<0 = anticorrelated (complementary). `frac` = share of this component's total coupling inertia on the axis; `r` = rho-weighted Pearson of the axis coordinate vs the named scale.

## K = 2

### Washout decomposition (what survives pooling the components)

Pooled pair correlation on each axis = within-class (genuine per-contact coupling, sign-aware) + between-class (composition heterogeneity, >=0), standardized to the pooled marginal. `strength` = mean per-class |coupling|.

| axis | per-class strength | within (survives) | between (composition) | pooled | per-class signs |
|---|---:|---:|---:|---:|---|
| charge | 0.073 | -0.073 | +0.004 | -0.069 | -- |
| hydropathy | 0.141 | +0.141 | +0.022 | +0.163 | ++ |
| volume | 0.131 | +0.070 | +0.002 | +0.072 | -+ |
| aromatic | 0.031 | +0.031 | +0.000 | +0.031 | ++ |

### component 0 (w=0.584, MI=0.050, inertia=0.1547)

| axis | sigma | frac | property (r) | + pole | - pole | reading |
|---:|---:|---:|---|---|---|---|
| hydropathy | +0.2766 | 0.49 | hydropathy (+0.21) | CHGMF | KRTSE | corr: like hydropathy together |
| volume | -0.1688 | 0.18 | volume (-0.63) | GANPW | LIFVM | ANTIcorr: opposite volume paired |
| hydropathy | +0.1371 | 0.12 | hydropathy (-0.61) | TKCSR | WFAGY | corr: like hydropathy together |

### component 1 (w=0.416, MI=0.120, inertia=0.2394)

| axis | sigma | frac | property (r) | + pole | - pole | reading |
|---:|---:|---:|---|---|---|---|
| hydropathy | +0.4088 | 0.70 | hydropathy (-0.85) | DNGPS | VILFM | corr: like hydropathy together |
| volume | +0.1682 | 0.12 | volume (+0.52) | DHRYN | AGPST | corr: like volume together |
| charge | -0.1330 | 0.07 | charge (+0.29) | KTARS | GHNCL | ANTIcorr: opposite charge paired |


## K = 3

### Washout decomposition (what survives pooling the components)

Pooled pair correlation on each axis = within-class (genuine per-contact coupling, sign-aware) + between-class (composition heterogeneity, >=0), standardized to the pooled marginal. `strength` = mean per-class |coupling|.

| axis | per-class strength | within (survives) | between (composition) | pooled | per-class signs |
|---|---:|---:|---:|---:|---|
| charge | 0.078 | -0.077 | +0.006 | -0.071 | --0 |
| hydropathy | 0.124 | +0.046 | +0.105 | +0.151 | -+- |
| volume | 0.072 | +0.065 | +0.009 | +0.074 | -++ |
| aromatic | 0.030 | +0.030 | +0.001 | +0.031 | +++ |

### component 0 (w=0.390, MI=0.069, inertia=0.2901)

| axis | sigma | frac | property (r) | + pole | - pole | reading |
|---:|---:|---:|---|---|---|---|
| hydropathy | +0.4431 | 0.68 | hydropathy (+0.16) | CGHMN | KSTRD | corr: like hydropathy together |
| hydropathy | -0.1724 | 0.10 | hydropathy (+0.61) | ILVRK | DEWYP | ANTIcorr: opposite hydropathy paired |
| aromatic | +0.1678 | 0.10 | aromatic (+0.51) | WYMLI | TSKRQ | corr: like aromatic together |

### component 1 (w=0.322, MI=0.106, inertia=0.2642)

| axis | sigma | frac | property (r) | + pole | - pole | reading |
|---:|---:|---:|---|---|---|---|
| hydropathy | +0.3680 | 0.51 | hydropathy (-0.92) | DRPKE | ILVFM | corr: like hydropathy together |
| volume | -0.2188 | 0.18 | volume (-0.54) | GWAPS | DHKRV | ANTIcorr: opposite volume paired |
| volume | +0.1978 | 0.15 | volume (-0.10) | PEQSA | DRGFI | corr: like volume together |

### component 2 (w=0.288, MI=0.086, inertia=0.1872)

| axis | sigma | frac | property (r) | + pole | - pole | reading |
|---:|---:|---:|---|---|---|---|
| volume | +0.2568 | 0.35 | volume (-0.87) | GPAST | RKLYI | corr: like volume together |
| hydropathy | -0.2401 | 0.31 | hydropathy (-0.77) | KRQTE | ILVFM | ANTIcorr: opposite hydropathy paired |
| hydropathy | +0.1525 | 0.12 | hydropathy (-0.45) | DHWNP | VALRI | corr: like hydropathy together |


## K = 4

### Washout decomposition (what survives pooling the components)

Pooled pair correlation on each axis = within-class (genuine per-contact coupling, sign-aware) + between-class (composition heterogeneity, >=0), standardized to the pooled marginal. `strength` = mean per-class |coupling|.

| axis | per-class strength | within (survives) | between (composition) | pooled | per-class signs |
|---|---:|---:|---:|---:|---|
| charge | 0.082 | -0.078 | +0.007 | -0.071 | ---+ |
| hydropathy | 0.134 | +0.026 | +0.141 | +0.167 | --++ |
| volume | 0.117 | +0.037 | +0.038 | +0.074 | ++-+ |
| aromatic | 0.028 | +0.028 | +0.003 | +0.031 | ++++ |

### component 0 (w=0.372, MI=0.038, inertia=0.0824)

| axis | sigma | frac | property (r) | + pole | - pole | reading |
|---:|---:|---:|---|---|---|---|
| hydropathy | -0.1821 | 0.40 | hydropathy (-0.65) | PDHWE | ILFVM | ANTIcorr: opposite hydropathy paired |
| aromatic | +0.1503 | 0.27 | aromatic (+0.50) | WPFCI | TSKNE | corr: like aromatic together |
| charge | -0.1063 | 0.14 | charge (-0.81) | EYFCS | KRHPW | ANTIcorr: opposite charge paired |

### component 1 (w=0.219, MI=0.120, inertia=0.2639)

| axis | sigma | frac | property (r) | + pole | - pole | reading |
|---:|---:|---:|---|---|---|---|
| hydropathy | -0.3568 | 0.48 | hydropathy (-0.80) | RKQTE | ILFVM | ANTIcorr: opposite hydropathy paired |
| volume | +0.2616 | 0.26 | volume (-0.77) | PGWAD | RILKV | corr: like volume together |
| aromatic | +0.1527 | 0.09 | aromatic (+0.43) | HDNFW | ATVQG | corr: like aromatic together |

### component 2 (w=0.209, MI=0.109, inertia=0.2592)

| axis | sigma | frac | property (r) | + pole | - pole | reading |
|---:|---:|---:|---|---|---|---|
| volume | -0.3349 | 0.43 | volume (-0.71) | GAWYS | DIHKL | ANTIcorr: opposite volume paired |
| hydropathy | +0.2332 | 0.21 | hydropathy (-0.58) | DRGKN | CYATV | corr: like hydropathy together |
| hydropathy | +0.1888 | 0.14 | hydropathy (-0.36) | PHNQE | FYMGI | corr: like hydropathy together |

### component 3 (w=0.200, MI=0.182, inertia=0.6737)

| axis | sigma | frac | property (r) | + pole | - pole | reading |
|---:|---:|---:|---|---|---|---|
| volume | +0.5210 | 0.40 | volume (-0.42) | CGDPS | VFLIM | corr: like volume together |
| hydropathy | +0.4871 | 0.35 | hydropathy (-0.80) | DPSGN | CIVLF | corr: like hydropathy together |
| charge | -0.2678 | 0.11 | charge (+0.49) | GPSTA | DENHQ | ANTIcorr: opposite charge paired |


## K = 8

### Washout decomposition (what survives pooling the components)

Pooled pair correlation on each axis = within-class (genuine per-contact coupling, sign-aware) + between-class (composition heterogeneity, >=0), standardized to the pooled marginal. `strength` = mean per-class |coupling|.

| axis | per-class strength | within (survives) | between (composition) | pooled | per-class signs |
|---|---:|---:|---:|---:|---|
| charge | 0.107 | -0.096 | +0.025 | -0.071 | ---+---+ |
| hydropathy | 0.139 | +0.008 | +0.150 | +0.158 | -0-+++0+ |
| volume | 0.154 | -0.007 | +0.082 | +0.074 | -+++--++ |
| aromatic | 0.044 | +0.027 | +0.006 | +0.033 | ++-0-++0 |

### component 0 (w=0.169, MI=0.087, inertia=0.1902)

| axis | sigma | frac | property (r) | + pole | - pole | reading |
|---:|---:|---:|---|---|---|---|
| hydropathy | -0.3265 | 0.56 | hydropathy (-0.64) | PDEGN | LFIMV | ANTIcorr: opposite hydropathy paired |
| hydropathy | +0.2065 | 0.22 | hydropathy (+0.56) | PLFMW | TKRSN | corr: like hydropathy together |
| charge | -0.1431 | 0.11 | charge (+0.73) | KRPQG | SDTFE | ANTIcorr: opposite charge paired |

### component 1 (w=0.152, MI=0.111, inertia=0.2674)

| axis | sigma | frac | property (r) | + pole | - pole | reading |
|---:|---:|---:|---|---|---|---|
| aromatic | +0.3530 | 0.47 | aromatic (+0.69) | FWYLI | PRDAK | corr: like aromatic together |
| hydropathy | -0.2618 | 0.26 | hydropathy (-0.59) | WNEDQ | ILVMC | ANTIcorr: opposite hydropathy paired |
| charge | -0.1665 | 0.10 | charge (+0.67) | WRNGK | LEDIP | ANTIcorr: opposite charge paired |

### component 2 (w=0.148, MI=0.118, inertia=0.2639)

| axis | sigma | frac | property (r) | + pole | - pole | reading |
|---:|---:|---:|---|---|---|---|
| hydropathy | -0.3509 | 0.47 | hydropathy (-0.93) | KRQHE | FIVLM | ANTIcorr: opposite hydropathy paired |
| volume | +0.2389 | 0.22 | volume (-0.70) | GNADR | EHQKI | corr: like volume together |
| hydropathy | -0.1846 | 0.13 | hydropathy (-0.12) | GDKHY | NACSE | ANTIcorr: opposite hydropathy paired |

### component 3 (w=0.125, MI=0.174, inertia=0.8493)

| axis | sigma | frac | property (r) | + pole | - pole | reading |
|---:|---:|---:|---|---|---|---|
| volume | +0.6125 | 0.44 | volume (-0.21) | CILVM | DPENQ | corr: like volume together |
| hydropathy | +0.4945 | 0.29 | hydropathy (-0.75) | DPENQ | IVLFM | corr: like hydropathy together |
| charge | -0.3762 | 0.17 | charge (-0.44) | DENQH | PAKRW | ANTIcorr: opposite charge paired |

### component 4 (w=0.118, MI=0.164, inertia=0.3311)

| axis | sigma | frac | property (r) | + pole | - pole | reading |
|---:|---:|---:|---|---|---|---|
| volume | -0.4322 | 0.56 | volume (-0.74) | GMFPN | VKYLI | ANTIcorr: opposite volume paired |
| hydropathy | +0.2532 | 0.19 | hydropathy (-0.58) | PHADS | MFIVL | corr: like hydropathy together |
| hydropathy | +0.1999 | 0.12 | hydropathy (-0.63) | NRKWD | PCMFA | corr: like hydropathy together |

### component 5 (w=0.116, MI=0.113, inertia=0.3454)

| axis | sigma | frac | property (r) | + pole | - pole | reading |
|---:|---:|---:|---|---|---|---|
| volume | -0.3781 | 0.41 | volume (+0.54) | DILVF | RGASK | ANTIcorr: opposite volume paired |
| hydropathy | +0.3385 | 0.33 | hydropathy (-0.56) | DRGNE | ACYFL | corr: like hydropathy together |
| volume | -0.2693 | 0.21 | volume (+0.70) | RHILF | DASCE | ANTIcorr: opposite volume paired |

### component 6 (w=0.090, MI=0.181, inertia=0.5095)

| axis | sigma | frac | property (r) | + pole | - pole | reading |
|---:|---:|---:|---|---|---|---|
| volume | +0.5065 | 0.50 | volume (-0.52) | GPWAD | FIYTV | corr: like volume together |
| hydropathy | +0.2971 | 0.17 | hydropathy (-0.48) | HDRWE | ILVTA | corr: like hydropathy together |
| hydropathy | -0.2715 | 0.14 | hydropathy (+0.58) | ILVMF | TSRDK | ANTIcorr: opposite hydropathy paired |

### component 7 (w=0.082, MI=0.290, inertia=0.8180)

| axis | sigma | frac | property (r) | + pole | - pole | reading |
|---:|---:|---:|---|---|---|---|
| volume | +0.6039 | 0.45 | volume (-0.77) | DSTNG | LIFYV | corr: like volume together |
| charge | -0.3889 | 0.18 | charge (+0.33) | TSHAM | GDKEC | ANTIcorr: opposite charge paired |
| charge | +0.2912 | 0.10 | charge (+0.47) | GHTWK | SDNAE | corr: like charge together |

