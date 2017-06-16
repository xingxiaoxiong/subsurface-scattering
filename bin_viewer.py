
from PIL import Image
import struct
import numpy as np
import os
import sys

byte_order = sys.byteorder

dir_path = '../data/regular'


def read_bin(file_path):  # return an numpy array height*width*3
    with open(file_path, 'rb') as file:
        width = int.from_bytes(file.read(4), byteorder=byte_order)
        height = int.from_bytes(file.read(4), byteorder=byte_order)

        # count = width * height
        print(height, width)
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

def render():
    # Image's (0, 0) is at upper left corner
    # (height, width, color channel)
    # bin data's (0, 0) is at lower left

    # filenames = os.listdir(dir_path)
    # for filename in filenames:
    #     if filename.endswith('.bin'):
    #         file_path = os.path.join(dir_path, filename)
    #         data = read_bin(file_path)
    #         data = data * 255
    #         data = data.astype('uint8')
    #         img = Image.fromarray(data)
    #         print(filename)
    #         img.show(filename)
    #     break
    filename = 'buddha_parallel_output.bin'
    file_path = os.path.join(dir_path, filename)
    data = read_bin(file_path)
    data = data * 255
    data = data.astype('uint8')
    img = Image.fromarray(data)
    # img.show(filename)
    img.save('buddha_parallel_output.jpg')


def check_object_range():
    path = '../data/regular/buddha_backlight_front_position.bin'
    with open(path, 'rb') as file:
        width = int.from_bytes(file.read(4), byteorder=byte_order)
        height = int.from_bytes(file.read(4), byteorder=byte_order)

        data = np.zeros((height, width, 3))
        for h in range(height):
            for w in range(width):
                r = struct.unpack('f', file.read(4))[0]
                g = struct.unpack('f', file.read(4))[0]
                b = struct.unpack('f', file.read(4))[0]
                a = struct.unpack('f', file.read(4))[0]
                data[height - 1 - h, w, 0] = r
                data[height - 1 - h, w, 1] = g
                data[height - 1 - h, w, 2] = b
                print(r, g, b, a)



if __name__ == '__main__':
    render()
    # check_object_range()