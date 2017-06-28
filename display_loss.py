
from matplotlib import pyplot as plt

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
    plt.show()


if __name__ == '__main__':
    plot('./loss_record.txt')
