"""Console commands for local browsing, batch solves and validation."""
import argparse
import json
import sys

from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fmm-workbench", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="start the local browser workbench")
    serve.add_argument("--port", type=int, default=8321)
    serve.add_argument("--runs-dir")
    commands.add_parser("env", help="print runtime versions and numerical flags")
    check = commands.add_parser("selfcheck", help="run the existing numerical self-check")
    check.add_argument("--bench", action="store_true")
    for name in ("solve", "optimize"):
        p = commands.add_parser(name, help="run a JSON configuration without a web server")
        p.add_argument("--config", required=True, help="schema-v1 JSON file")
        p.add_argument("--output", required=True, help="new output directory (must not exist)")
        p.add_argument("--fields", action="store_true", help="also compute the web field maps")
    args = parser.parse_args(argv)
    try:
        if args.command == "serve":
            from .app import main as serve_main
            serve_args = ["--port", str(args.port)]
            if args.runs_dir:
                serve_args += ["--runs-dir", args.runs_dir]
            serve_main(serve_args)
        elif args.command == "selfcheck":
            from .selfcheck import main as check_main
            return check_main(["--bench"] if args.bench else [])
        elif args.command == "env":
            from . import app  # apply the same numerical configuration as a run
            from .provenance import environment
            print(json.dumps(environment(), indent=2))
        else:
            from .batch import run_experiment
            from .configuration import load_config
            config = load_config(args.config)
            if config["type"] != args.command:
                raise ValueError("configuration type does not match the command")
            result = run_experiment(config, args.output, fields=args.fields)
            value = result.get("eta_target") if args.command == "solve" else result["best"]["eta_top"]
            print(json.dumps({"status": "complete", "output": args.output, "eta": value}))
        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError) as exc:
        print("fmm-workbench: " + str(exc), file=sys.stderr)
        return 1
