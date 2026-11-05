"""`python -m fusezap serve ...` / `python -m fusezap validate ...` / `python -m fusezap generate ...` (also the `fusezap` console script)."""
import sys

def main(argv: list[str] | None = None) -> int:
  argv = list(sys.argv[1:] if argv is None else argv)
  if argv and argv[0] == "validate":
    try: from fusezap.validate.cli import main as validate_main
    except ModuleNotFoundError as e:
      if e.name not in ("fusezap.validate", "fusezap.validate.cli"): raise
      print("fusezap validate: FuseZap Validate is not installed. It is a separately licensed package "
            "(fusezap-validate, https://fusezap.ai); the open-source fusezap package provides `fusezap serve`.", file=sys.stderr)
      return 2
    return validate_main(argv[1:])
  if argv and argv[0] == "generate":   # text to image (FLUX.2 klein, I81 D6)
    from fusezap.serve.generate import main as generate_main
    return generate_main(argv[1:])
  from fusezap.serve.cli import main as serve_main
  return serve_main(argv)

if __name__ == "__main__": sys.exit(main())
