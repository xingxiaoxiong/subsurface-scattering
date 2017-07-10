import os
from matplotlib import pyplot as plt
import json
import numpy as np
from PIL import Image

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


def npy_render(path):
    data = np.load(path)
    data = data * 255.0
    data = data.astype('uint8')
    img = Image.fromarray(data)
    img.save(path[:-4])


def render_all_npy(root_dir):
    dir_names = os.listdir(root_dir)
    for dir_name in dir_names:
        current_dir = os.path.join(root_dir, dir_name)
        file_names = os.listdir(current_dir)
        for file_name in file_names:
            if file_name.endswith('.npy'):
                file_path = os.path.join(current_dir, file_name)
                npy_render(file_path)


class Table:

    def __init__(self, path):
        self.path = path
        dir_names = os.listdir(self.path)
        for dir_name in dir_names:
            current_dir = os.path.join(path, dir_name)
            Table.plot_loss(current_dir)
            Table.npy2img(current_dir)
        self.to_html()

    @staticmethod
    def plot_loss(current_dir):
        path = os.path.join(current_dir, 'loss_record.txt')
        with open(path, 'r') as file:
            lines = file.readlines()
            xs = []
            t_losses = []
            v_losses = []
            for line in lines:
                step, t_loss, v_loss = line.strip().split('\t')
                xs.append(step)
                t_losses.append(t_loss)
                v_losses.append(v_loss)
            plt.plot(xs, t_losses, 'r', xs, v_losses, 'b')
            plt.xlabel('epoch')
            plt.ylabel('error per pixel per channel')
            plt.savefig(os.path.join(current_dir, 'loss.jpg'))
            plt.close()

    @staticmethod
    def npy2img(current_dir):
        file_names = os.listdir(current_dir)
        for file_name in file_names:
            if file_name.endswith('.npy'):
                file_path = os.path.join(current_dir, file_name)
                npy_render(file_path)

    def append_index(self, filesets):
        index_path = os.path.join(self.path, "index.html")
        if os.path.exists(index_path):
            index = open(index_path, "a")
        else:
            index = open(index_path, "w")
            index.write("<html><body><table><tr>")
            index.write("<th>learning rate</th><th>loss</th><th>result</th></tr>")

        for fileset in filesets:
            index.write("<tr>")

            index.write("<td>%s</td>" % fileset["learning_rate"])
            index.write("<td><img src='%s'></td>" % fileset['loss'])
            index.write("<td><img src='%s'></td>" % fileset['result'])

            index.write("</tr>")
        return index_path

    def to_html(self):
        lrs = []
        dir_names = os.listdir(self.path)
        for dir_name in dir_names:
            path = os.path.join(self.path, dir_name, 'options.json')
            json_file = open(path, 'r')
            options = json.load(json_file)
            lr = options['lr']
            dict = {'learning_rate': lr,
                        'loss': os.path.join(dir_name, 'loss.jpg'),
                        'result': os.path.join(dir_name, '49.jpg')}
            lrs.append(dict)
        lrs = sorted(lrs, key=lambda tup: tup['learning_rate'])
        self.append_index(lrs)


if __name__ == '__main__':
    # plot('./loss_record.txt')
    # plot_experiment('experiment/experiment3')
    # compare_loss_across_lr('experiment/experiment3')

    # npy_render('./temp/39.jpg.npy')
    # render_all_npy('experiment/experiment5')
    Table('experiment/experiment5')
