"""Regenerate summary.csv and convergence PNG/PDF files from exported runs."""
import argparse
from workbench.reproduction import summarize

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="experiment output directory")
    parser.add_argument("--plots", action="store_true", help="requires the plots extra")
    args = parser.parse_args()
    summarize(args.input, plots=args.plots)
