
import struct
import numpy as np
import os
import sys
import random

from bin_viewer import read_bin

base_dir = '../data/regular'
save_dir = './data'


def generate_backlit_training_data():
    back_position_path = os.path.join(base_dir, 'buddha_backlight_back_position.bin')
    back_position_data = read_bin(back_position_path)

    front_position_path = os.path.join(base_dir, 'buddha_backlight_front_position.bin')
    front_position_data = read_bin(front_position_path)

    back_irradiance_path = os.path.join(base_dir, 'buddha_backlight_back_irradiance.bin')
    back_irradiance_data = read_bin(back_irradiance_path)

    front_irradiance_path = os.path.join(base_dir, 'buddha_backlight_front_irradiance.bin')
    front_irradiance_data = read_bin(front_irradiance_path)

    output_path = os.path.join(base_dir, 'buddha_backlight_output.bin')
    output_data = read_bin(output_path)

    height, width, num_channels = output_data.shape

    export_y = []
    count = 0
    for h in range(height):
        for w in range(width):
            target_point_position = front_position_data[h, w]
            if target_point_position[0] != 0.0 and target_point_position[1] != 0.0 and target_point_position[2] != 0.0:
                front_relative_position = front_position_data - target_point_position
                back_relative_position = back_position_data - target_point_position
                # print(front_position_data[h + 1, w], target_point_position, front_relative_position[h + 1, w])
                export_X = np.concatenate((front_relative_position, back_relative_position), axis=2)
                np.save(os.path.join(save_dir, '%04d.npy' % count), export_X)
                count += 1
                export_y.append(output_data[h, w])

    export_y = np.array(export_y)
    np.save(os.path.join(save_dir, 'y.npy'), export_y)
    print(export_y.shape)

    np.save(os.path.join(save_dir, 'back_irradiance.npy'), back_irradiance_data)
    np.save(os.path.join(save_dir, 'front_irradiance.npy'), front_irradiance_data)


def generate_backlit_training_data_subsample():
    random.seed(1)

    back_position_path = os.path.join(base_dir, 'buddha_backlight_back_position.bin')
    back_position_data = read_bin(back_position_path)

    front_position_path = os.path.join(base_dir, 'buddha_backlight_front_position.bin')
    front_position_data = read_bin(front_position_path)

    back_irradiance_path = os.path.join(base_dir, 'buddha_backlight_back_irradiance.bin')
    back_irradiance_data = read_bin(back_irradiance_path)

    front_irradiance_path = os.path.join(base_dir, 'buddha_backlight_front_irradiance.bin')
    front_irradiance_data = read_bin(front_irradiance_path)

    output_path = os.path.join(base_dir, 'buddha_backlight_output.bin')
    output_data = read_bin(output_path)

    height, width, num_channels = output_data.shape

    step_size = 16
    anchors_h = range(0, height, step_size)
    anchors_w = range(0, width, step_size)
    sample_number = 1
    mask = np.zeros((height, width)).astype('uint8')

    export_y = []
    count = 0
    for h in anchors_h:
        for w in anchors_w:
            for _ in range(sample_number):
                dh = random.randint(0, 15)
                dw = random.randint(0, 15)

                sample_h = h + dh
                sample_w = w + dw

                target_point_position = front_position_data[sample_h, sample_w]
                if target_point_position[0] != 0.0 and target_point_position[1] != 0.0 and target_point_position[2] != 0.0:
                    mask[sample_h, sample_w] = 255
                    front_relative_position = front_position_data - target_point_position
                    back_relative_position = back_position_data - target_point_position
                    # print(front_position_data[h + 1, w], target_point_position, front_relative_position[h + 1, w])
                    export_X = np.concatenate((front_relative_position, back_relative_position), axis=2)
                    np.save(os.path.join(save_dir, '%05d.npy' % count), export_X)
                    count += 1
                    export_y.append(output_data[h, w])

    export_y = np.array(export_y)
    np.save(os.path.join(save_dir, 'y.npy'), export_y)
    print(export_y.shape)

    np.save(os.path.join(save_dir, 'back_irradiance.npy'), back_irradiance_data)
    np.save(os.path.join(save_dir, 'front_irradiance.npy'), front_irradiance_data)

    from PIL import Image
    img = Image.fromarray(mask)
    img.save(os.path.join(save_dir, 'mask.png'))


def check_generated_data():
    file_path = './data/00000.npy'
    data = np.load(file_path)
    print(data.shape)


if __name__ == '__main__':
    # check_generated_data()
    # generate_backlit_training_data()
    generate_backlit_training_data_subsample()
