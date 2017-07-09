
from PIL import Image
import struct
import numpy as np
import os
import sys

byte_order = sys.byteorder


def read_bin(file_path):  # return an numpy array height*width*3
    with open(file_path, 'rb') as file:
        width = int.from_bytes(file.read(4), byteorder=byte_order)
        height = int.from_bytes(file.read(4), byteorder=byte_order)

        # count = width * height
        data = np.zeros((height, width, 3), np.float32)
        for h in range(height):
            for w in range(width):
                r = struct.unpack('f', file.read(4))[0]
                g = struct.unpack('f', file.read(4))[0]
                b = struct.unpack('f', file.read(4))[0]
                # if r == 0.0 and g == 0.0 and b == 0.0:
                #     count -= 1
                a = file.read(4)
                data[height - 1 - h, w, 0] = r
                data[height - 1 - h, w, 1] = g
                data[height - 1 - h, w, 2] = b
        # print(count)
    return data

def render(dir_path):
    # Image's (0, 0) is at upper left corner
    # (height, width, color channel)
    # bin data's (0, 0) is at lower left

    filenames = os.listdir(dir_path)
    for filename in filenames:
        if filename.endswith('output.bin') or filename.endswith('irradiance.bin'):
            file_path = os.path.join(dir_path, filename)
            data = read_bin(file_path)
            data = data * 255
            data = data.astype('uint8')
            img = Image.fromarray(data)
            name = filename.split('.')[0]
            save_path = os.path.join(dir_path, name + '.png')
            img.save(save_path)

    # filename = 'buddha_parallel_output.bin'
    # file_path = os.path.join(dir_path, filename)
    # data = read_bin(file_path)
    # data = data * 255
    # data = data.astype('uint8')
    # img = Image.fromarray(data)
    # # img.show(filename)
    # img.save('buddha_parallel_output.jpg')


def check_object_range():
    path = '../data/sphere/sphere_grace_back_irradiance.bin'
    data = read_bin(path)
    height, width, _ = data.shape
    min_vals = np.array([2**32, 2**32, 2**32])
    max_vals = np.array([-2**32, -2**32, -2**32])
    for h in range(height):
        for w in range(width):
            val = data[h, w]
            min_vals = [min(val[i], min_vals[i]) for i in range(3)]
            max_vals = [max(val[i], max_vals[i]) for i in range(3)]
    print(min_vals, max_vals)


def npy_render(path):
    data = np.load(path)
    data = data * 255.0
    data = data.astype('uint8')
    img = Image.fromarray(data)
    img.show()


if __name__ == '__main__':
    # render('../data/sphere')
    # check_object_range()

    # root = '../data/blend2'
    # dirs = os.listdir(root)
    # for dir in dirs:
    #     render(os.path.join(root, dir))
    npy_render('./temp/9.jpg.npy')

