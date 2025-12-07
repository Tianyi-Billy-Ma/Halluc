import argparse
import sys
from llamafactory.cli import main


def parse_args(arg_list: list[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, default="train")
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--additional", type=str, nargs="+", default=[])
    return parser.parse_args(arg_list)


def run(args: argparse.Namespace):
    sys.argv = sys.argv.append([args.additional, args.config_path])
    main()


if __name__ == "__main__":
    arg_list = None
    args = parse_args(arg_list)
    run(args)
