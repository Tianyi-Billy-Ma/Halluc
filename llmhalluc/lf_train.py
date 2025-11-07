import argparse
from llamafactory.cli import main_with_args as lf_main_with_args


def parse_args(arg_list: list[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, default="train")
    parser.add_argument("--additional", type=str, nargs="+", default=[])
    return parser.parse_args(arg_list)


def run(args: argparse.Namespace):
    lf_main_with_args(args)


if __name__ == "__main__":
    arg_list = None
    args = parse_args(arg_list)
    run(args)
