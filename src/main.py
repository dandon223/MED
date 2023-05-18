import argparse
from tests import dev_test, test
import pandas as pd
import json

def run(config_file: str):
    with open(config_file) as f:
        data = json.load(f)
        print(data)
def main():
    parser = argparse.ArgumentParser(description='MED: Automatic testing suit')

    parser.add_argument('--config_file', help='config file for algorithm', type=str)
    
    args = parser.parse_args()

    if args.config_file is not None:
        run(args.config_file)

if __name__ == "__main__":
    main()
