import pandas as pd
import os

def main(args):
    test_list = open(args.file_name, 'r').readlines()
    for i in range(len(test_list)):
        test_list[i] = int(test_list[i].split('/')[1].split('_')[0])
    test_list = list(set(test_list))
    test_list.sort()
    test_len = len(test_list)
    print (test_len)
    print (test_list)

    df = pd.read_csv(args.file_dir + 'estimated_data.csv')
    df = df[df['im_id'].isin(test_list)]

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_csv(output_dir + 'modified_data.csv', index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Strip and Modify')
    parser.add_argument('--file_name', type=str, help='input file', default='supplementary/test.txt')
    parser.add_argument('--file_dir', type=str, help='input file', default='logs/test1/')
    parser.add_argument('--output_dir', type=str, help='output file', default='logs/test3/')
    args = parser.parse_args()
    main(args)