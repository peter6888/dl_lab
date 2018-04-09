import argparse
import sys

def predict(args):
    print(args.bugid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse two arguments at command line.")
    subparsers = parser.add_subparsers()
    command_parser = subparsers.add_parser("predict", help="Predict with bugid")
    command_parser.set_defaults(func=predict)

    parser.add_argument("--bugid", help="The bug id")

    ARGS = parser.parse_args()

    if not hasattr(ARGS, 'func'):
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
