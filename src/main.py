import argparse
from tests import dev_test

def main():
    parser = argparse.ArgumentParser(description='MED: Automatic testing suit')

    #group = parser.add_mutually_exclusive_group()
    parser.add_argument('--dev_test', help='run test suit for development', type=str)
    parser.add_argument('--dataset', help='one dataset for testing', type=str)
    parser.add_argument('--test', help='run test suit', type=str)
    parser.add_argument('--plot',
                       help='plot sample clustering results',
                       type=str)
    
    args = parser.parse_args()

    if args.dev_test is not None:
        if args.dataset is not None:
            dev_test(args.dataset)
        else:
            dev_test()


if __name__ == "__main__":
    main()