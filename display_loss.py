import os
from matplotlib import pyplot as plt
import json

def plot(path):
    x = []
    y1 = []
    y2 = []
    with open(path, 'r') as file:
        for line in file:
            e, tl, vl = line.strip().split('\t')
            x.append(int(e))
            y1.append(float(tl))
            y2.append(float(vl))

    plt.plot(x, y1, 'r-', x, y2, 'b--')
    plt.xlabel('epoch')
    plt.ylabel('error per pixel per channel')
    plt.savefig(path.split('.')[0] + '.jpg')
    plt.close()


def plot_experiment(root_dir):
    dir_names = os.listdir(root_dir)
    for dir_name in dir_names:
        path = os.path.join(root_dir, dir_name, 'loss_record.txt')
        plot(path)


def compare_loss_across_lr(root_dir):
    dir_names = os.listdir(root_dir)
    y = []
    for dir_name in dir_names:
        path = os.path.join(root_dir, dir_name, 'options.json')
        json_file = open(path, 'r')
        options = json.load(json_file)
        lr = options['lr']
        print(dir_name, lr)

        path = os.path.join(root_dir, dir_name, 'loss_record.txt')
        with open(path, 'r') as file:
            lines = file.readlines()
            _, t_loss, v_loss = lines[-1].strip().split('\t')
            y.append((lr, t_loss, v_loss))

    y = sorted(y, key=lambda tup: tup[0])
    x = [tup[0] for tup in y]
    plt.plot(x, [tup[1] for tup in y], 'r', x, [tup[2] for tup in y], 'b')
    plt.xlabel('experiment')
    plt.ylabel('error per pixel per channel')
    plt.savefig(os.path.join(root_dir, 'compare.jpg'))
    plt.close()

if __name__ == '__main__':
    # plot('./loss_record.txt')
    plot_experiment('experiment/experiment3')
    # compare_loss_across_lr('experiment/experiment3')
