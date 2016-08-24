import argparse
from sklearn.cross_validation import KFold
import os
import time

parser = argparse.ArgumentParser(description='Transform Movielens data set to lightsvm format.')
parser.add_argument('base_dir', type=str, help='Base directory')
parser.add_argument('file', type=str, help='Dataset file to transform')
parser.add_argument('sep', type=str, help='Field separator')
parser.add_argument('--kfold', type=str, default=False, help='Generate k-fold datasets')


def gen_k_fold(base_dir, data, n_folds=5):
    k_fold = KFold(n=len(data), n_folds=n_folds)

    n = 1
    for train_indices, test_indices in k_fold:
        train = [data[i] for i in train_indices]
        test = [data[i] for i in test_indices]
        save_file_batch(os.path.join(base_dir, "u{}".format(n)) + ".base.libfm", train)
        save_file_batch(os.path.join(base_dir, "u{}".format(n)) + ".test.libfm", test)
        n += 1


def save_file_batch(filename, data):
    with open(filename, "w") as f_out:
        for row in data:
            f_out.write(row)
    f_out.close()


if __name__ == "__main__":
    start_time = time.time()
    args = parser.parse_args()
    base_dir = args.base_dir
    filename = args.file
    run_fold = args.kfold
    separator = args.sep
    f = open(os.path.join(base_dir, filename), mode='r')
    last_i = last_u = -1
    u_id = i_id = 0
    index = -1
    data = []
    with open(os.path.join(base_dir, filename) + ".libfm", "w") as out_f:
        print "processing file " + out_f.name
        for row in f:
            user_id, item_id, rating, _ = row.split(separator)
            if user_id != last_u:
                last_u = user_id
                index += 1
                u_id = index
            if item_id != last_i:
                last_i = item_id
                index += 1
                i_id = index
            line = "{} {} {}\n".format(rating, str(u_id) + ":1", str(i_id) + ":1")
            if run_fold:
                data.append(line)
            out_f.write(line)
    out_f.close()
    f.close()
    if run_fold:
        gen_k_fold(base_dir, data)
    print "Process finished in {} seconds".format(time.time() - start_time)