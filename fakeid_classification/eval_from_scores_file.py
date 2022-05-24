import os
from main import eval

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Force to use CPU
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def load_scores_from_file(scores_path):
    f = open(scores_path, 'rt')
    scores = list()
    labels = list()
    for line in f:
        score, label = line.strip('\n').split(' ')
        scores.append(float(score))
        labels.append(int(label))
    f.close()

    return scores, labels


def main():
    scores_path = '/home/dschulz/TOC/fakeid_2.0/releases/URY-2022-05/models/fakeid/all_35966_imgs/scores.txt'
    results_path = '/home/dschulz/TOC/fakeid_2.0/releases/URY-2022-05/models/fakeid/all_35966_imgs'
    threshold = 0.89227

    scores, labels = load_scores_from_file(scores_path)
    eval(labels, scores, threshold, results_path)


if __name__ == '__main__':
    main()


