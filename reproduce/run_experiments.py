"""Run after installing the project (``python -m pip install -e .``)."""
from workbench.reproduction import main

if __name__ == "__main__":
    raise SystemExit(main())
