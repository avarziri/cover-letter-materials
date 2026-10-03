"""Reproduce the derived quantities cited in Results_Discussion_Conclusion.docx.

All inputs are the published estimates in the Results tables (Tables 3-6):
  - MXL WTP and implied share of respondents with positive preferences (Table 3)
  - LCM reservation prices and message effects in dollars (Table 4)
  - Share-weighted vs. median-class WTP (Tables 4-5)
  - Reproduction of Table 6 and the segment decomposition reported as Table 7
  - Market-expansion and message-scenario simulations

Run: python3 analysis/derived_results.py   (requires numpy)
"""
from math import erf, sqrt

import numpy as np

PHI = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
ATTRS = ["c10", "c20", "c30", "pb", "bm", "gmo", "seas", "ai", "bio", "pf", "org"]

# Table 3: mixed logit (mean, SD); price fixed at -1.09
MXL_PRICE = -1.09
MXL = {"c10": (.61, .36), "c20": (.86, .23), "c30": (.95, .20), "pb": (.02, .08),
       "bm": (.30, .85), "gmo": (-.25, .99), "seas": (.38, .41), "ai": (.04, .92),
       "bio": (.51, .26), "pf": (.66, 1.32), "org": (.13, 1.51)}

# Table 4: latent class coefficients (non-intrusive wording; messages act on opt-out)
LCM = {
    1: dict(c10=.23, c20=.33, c30=.06, pb=.07, bm=.19, gmo=.47, seas=.04, ai=-.38, bio=.06, pf=.47, org=.29,
            asc=-3.02, p=-.26, pers=-.25, soc=.12, comb=.35),
    2: dict(c10=-.03, c20=.14, c30=.35, pb=.02, bm=-.43, gmo=-.70, seas=.32, ai=.32, bio=.66, pf=-.22, org=-.48,
            asc=-5.64, p=-1.82, pers=.72, soc=-3.14, comb=.88),
    3: dict(c10=1.28, c20=1.70, c30=2.78, pb=-.32, bm=.25, gmo=-.94, seas=.79, ai=1.41, bio=1.28, pf=.63, org=.65,
            asc=-1.96, p=-.33, pers=.77, soc=.37, comb=-.11),
    4: dict(c10=.43, c20=.84, c30=.42, pb=.03, bm=.63, gmo=-.39, seas=.89, ai=-.59, bio=.32, pf=2.18, org=.30,
            asc=-1.35, p=-1.00, pers=.08, soc=-.14, comb=-.05),
    5: dict(c10=1.62, c20=1.93, c30=2.28, pb=-.52, bm=-.40, gmo=-1.86, seas=1.23, ai=.61, bio=.81, pf=-.11, org=-1.17,
            asc=-13.19, p=-3.51, pers=.59, soc=4.71, comb=.37),
}
SHARE = {1: .349, 2: .106, 3: .279, 4: .107, 5: .158}

# Table 6 profiles (Product A = conventional baseline: all omitted levels, city water, not organic)
PROFILES = {"B1a": ["c20", "bm", "pf", "org"], "B1b": ["c20", "gmo", "pf"], "B2": ["bio"],
            "B3": ["c30", "bm", "ai", "pf", "org"], "B4": ["c20", "bm", "seas", "pf", "org"],
            "B5": ["c30", "seas"]}
PRICES = [2.19, 2.69, 3.19, 3.69, 4.19]


def market(alts, msg=None):
    """alts: list of (attribute list, price). Returns (sample shares %, {class: shares %}); opt-out is last."""
    total, per = np.zeros(len(alts) + 1), {}
    for c, b in LCM.items():
        v = [sum(b[a] for a in lst) + b["p"] * price for lst, price in alts]
        v.append(b["asc"] + (b[msg] if msg else 0))
        e = np.exp(np.array(v) - max(v))
        pr = e / e.sum()
        per[c] = pr * 100
        total += SHARE[c] * pr
    return total * 100, per


def main():
    print("== Mixed logit: WTP ($/pack) and share with positive coefficient")
    for a, (m, s) in MXL.items():
        print(f"  {a:5s} WTP={-m / MXL_PRICE:6.2f}  P(>0)={PHI(m / s) * 100:5.1f}%")
    print(f"  GMO under intrusive wording: WTP {(-.25 - .72) / -MXL_PRICE:.2f} (vs {-.25 / -MXL_PRICE:.2f})")

    print("\n== LCM: reservation price for conventional pack and message effects ($)")
    for c, b in LCM.items():
        rp = b["asc"] / b["p"]
        fx = {m: round(b[m] / b["p"], 2) for m in ("pers", "soc", "comb")}  # + raises reservation price
        print(f"  Class {c}: reservation=${rp:5.2f}  message shifts={fx}")

    print("\n== Share-weighted vs. median-class WTP")
    for a in ATTRS:
        w = {c: SHARE[c] * LCM[c][a] / -LCM[c]["p"] for c in LCM}
        print(f"  {a:5s} weighted=${sum(w.values()):5.2f}  Class 3 share of weighted mean={w[3] / sum(w.values()) * 100:6.1f}%")

    print("\n== Table 6 reproduction and Table 7 (B choice probability by class, %)")
    for k, lst in PROFILES.items():
        for price in PRICES:
            tot, per = market([([], 2.19), (lst, price)])
            cls = " ".join(f"{per[c][1]:5.1f}" for c in LCM)
            print(f"  {k:4s} ${price:.2f}  A/B/opt={tot.round(2)}  by class: {cls}")

    print("\n== Market expansion (opt-out %)")
    print("  A only:", market([([], 2.19)])[0][-1].round(1))
    print("  A + identical A:", market([([], 2.19), ([], 2.19)])[0][-1].round(1))
    print("  A + B4 @2.19:", market([([], 2.19), (PROFILES["B4"], 2.19)])[0][-1].round(1),
          " @4.19:", market([([], 2.19), (PROFILES["B4"], 4.19)])[0][-1].round(1))
    print("  Class 4 buys A when A is the only product:", market([([], 2.19)])[1][4][0].round(1))

    print("\n== Message scenarios: change in opt-out and B share (pp) vs. no message")
    for price in (2.19, 3.19, 4.19):
        for k, lst in PROFILES.items():
            base = market([([], 2.19), (lst, price)])[0]
            out = []
            for m in ("pers", "soc", "comb"):
                r = market([([], 2.19), (lst, price)], m)[0]
                out.append(f"{m}: opt {r[2] - base[2]:+.2f}, B {r[1] - base[1]:+.2f}")
            print(f"  ${price:.2f} {k:4s} " + " | ".join(out))


if __name__ == "__main__":
    main()
