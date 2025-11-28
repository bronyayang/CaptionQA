import json
import argparse
from pipeline.utils import load_json

def count_taxonomy(data, level=0):
    """
    Count the number of taxonomy at a certain level. 
    Higher level count will include lower level taxonomy if no taxonomy at that level.
    """
    count = 0
    if not isinstance(data, dict):
        return 0
    if level == 0:
        return len(data.keys())
    for i in data.keys():
        cur_count = count_taxonomy(data[i], level-1)
        if cur_count == 0:
            count += 1
        count += cur_count
    return count

def main():
    parser = argparse.ArgumentParser(description='Count taxonomy')
    parser.add_argument('--json', type=str, help='json file')
    parser.add_argument('--level', type=int, help='level of taxonomy')
    args = parser.parse_args()
    data = load_json(args.json)
    count = count_taxonomy(data, level=args.level)
    print(f"Number of level {args.level} taxonomy in {args.json}: {count}")
    
if __name__ == '__main__':
    main()