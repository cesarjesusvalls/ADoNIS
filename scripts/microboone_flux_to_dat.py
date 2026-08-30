"""Write the ACHILLES MicroBooNE flux table from the HEPData YAML the image carries.

configs/achilles/run_MicroBooNE_*.yml read flux/microboone_numu.dat, which upstream does not
ship; it holds the same bins and values as flux/microboone_flux_numu.yaml, in ACHILLES' format.

    python scripts/microboone_flux_to_dat.py <in.yaml> <out.dat>
"""
import sys

import yaml

HEADER = ("Achilles\n"
          "MicroBooNE Flux\n"
          "units: v/m^2/POT/500MeV\n"
          "  lower edge      upper edge   value                    error\n")


def main(src, dst):
    d = yaml.safe_load(open(src))
    bins = d["independent_variables"][0]["values"]
    vals = d["dependent_variables"][0]["values"]
    if len(bins) != len(vals):
        raise SystemExit(f"{src}: {len(bins)} bins but {len(vals)} values")
    with open(dst, "w") as f:
        f.write(HEADER)
        for b, v in zip(bins, vals):
            f.write(f"  {b['low']:<15.5f} {b['high']:<12.5f} {float(v['value']):<24.6e} "
                    f"{0.0:.6e}\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
