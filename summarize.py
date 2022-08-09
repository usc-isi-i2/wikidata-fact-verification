import argparse
import json
import os
from collections import defaultdict

ACCURACY = 'Accuracy'

F1 = 'F1'

RECALL = 'Recall'

PRECISION = 'Precision'

EPOCH = 'Epoch'

DATASET = 'Dataset'

metric_indexes = {
    DATASET: 0,
    EPOCH: 1,
    PRECISION: 2,
    RECALL: 3,
    F1: 4,
    ACCURACY: 5
}

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--start', type=int, default=1, help='Index of starting experiment')
    arg_parser.add_argument('--end', type=int, default=14, help='Index of last experiment')
    args = arg_parser.parse_args()

    train_datasets = set()
    eval_datasets = set()
    result_header = f'{DATASET}\t{EPOCH}\t{PRECISION}\t{RECALL}\t{F1}\t{ACCURACY}'

    configs = {}
    results = defaultdict(lambda: {})
    for i in range(args.start, args.end + 1):
        with open(os.path.join('experiments/configs', f'exp_{i}.json')) as f:
            exp_config = json.load(f)
            train_datasets.update(exp_config['train_datasets'])
            eval_datasets.update(exp_config['eval_datasets'])
            configs[i] = exp_config

        with open(os.path.join('experiments/logs', f'exp_{i}.tsv')) as f:
            logs = f.readlines()
            if logs[0].strip() != result_header:
                print(f'Invalid logs for exp_{i}')
                continue

            for log in logs[1:]:
                log_list = log.strip().split('\t')
                results[(i, int(log_list[metric_indexes[EPOCH]]))][log_list[metric_indexes[DATASET]]] = float(log_list[metric_indexes[F1]])

    with open('summary.tsv', 'w') as f:
        header_list = ['Experiment'] + list(train_datasets) + [EPOCH] + list(eval_datasets)
        f.write('\t'.join(header_list))
        f.write('\n')

        for (exp, epoch), values in results.items():
            summary = [exp]
            for d in train_datasets:
                if d in configs[exp]['train_datasets']:
                    summary.append('yes')
                else:
                    summary.append('no')

            summary.append(epoch)
            for d in eval_datasets:
                if d in configs[exp]['eval_datasets']:
                    summary.append(results[(exp, epoch)][d])
                else:
                    summary.append('-')

            f.write('\t'.join([str(v) for v in summary]))
            f.write('\n')
