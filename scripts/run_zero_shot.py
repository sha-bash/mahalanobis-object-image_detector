#!/usr/bin/env python
import sys

from moid.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["zero-shot", *sys.argv[1:]]))
