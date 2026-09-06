from moid.cli import main


def test_cli_help():
    try:
        main(["--help"])
    except SystemExit as e:
        assert e.code == 0


def test_cli_search_help():
    try:
        main(["search", "--help"])
    except SystemExit as e:
        assert e.code == 0
