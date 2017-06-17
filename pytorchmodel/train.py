import argparse
import datetime
import json
import os
import time
import torch
import numpy as np

from cnn import CNN

from data_loader import Loader

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True, choices=["train", "test"])
parser.add_argument("--output_dir", default=None, help="where to put output files")
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.9, help="momentum term of adam")
parser.add_argument("--max_epochs", type=int, default=1000, help="number of training epochs")
parser.add_argument("--gpu", type=bool, default=False, help="enable GPU")

# unit in epochs
parser.add_argument("--save_freq", type=int, default=5, help="save model every save_freq epochs, 0 to disable")
parser.add_argument("--summary_freq", type=int, default=5, help="update summaries every summary_freq epochs")
parser.add_argument("--progress_freq", type=int, default=1, help="display progress every progress_freq epochs")
parser.add_argument("--validation_freq", type=int, default=5, help="display progress every validation_freq epochs")
parser.add_argument("--display_freq", type=int, default=10, help="write images every display_freq epochs")

a = parser.parse_args()
if not a.output_dir:
    output_prepath = './output'
    if not os.path.isdir(output_prepath):
        os.makedirs(output_prepath)
    a.output_dir = os.path.join(output_prepath, datetime.datetime.now().strftime("%I_%M%p_on_%B_%d_%Y"))
    if not os.path.isdir(a.output_dir):
        os.makedirs(a.output_dir)


def draw(cnn, save_path):
    data_dir = '../data'
    front_position = np.load(os.path.join(data_dir, 'front_position.npy'))
    back_position = np.load(os.path.join(data_dir, 'back_position.npy'))
    front_lit = np.load(os.path.join(data_dir, 'front_irradiance.npy'))
    back_lit = np.load(os.path.join(data_dir, 'back_irradiance.npy'))
    height, width, _ = front_position.shape
    image = np.zeros((height, width, 3)).astype('uint8')
    for h in range(height):
        for w in range(width):
            position = front_position[h, w]
            if position[0] == 0.0 and position[1] == 0.0 and position[2] == 0.0:
                image[h, w] = [0, 0, 0]
            else:
                front_relative_position = front_position - position
                back_relative_position = back_position - position
                X = np.concatenate((front_relative_position, back_relative_position, front_lit, back_lit), axis=2)
                X = X[None, :]
                X = np.swapaxes(X, 1, 3)
                X = np.swapaxes(X, 2, 3)
                tX = torch.from_numpy(X)
                color = cnn.predict_step(tX)
                image[h, w] = [int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)]

    from PIL import Image
    img = Image.fromarray(image)
    img.save(os.path.join(save_path))



def main():
    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    loader = Loader(a.batch_size)

    # initialize models here
    model = CNN(a)

    if a.checkpoint is not None:
        print("loading model from checkpoint")
        model.load(a.checkpoint)

    if a.mode == 'test':
        if a.checkpoint is None:
            print('need checkpoint to continue')
            return
        draw(model, os.path.join(a.output_dir, 'test_output.jpg'))
    else:
        # training
        start = time.time()
        for epoch in range(a.max_epochs):
            def should(freq):
                return freq > 0 and ((epoch + 1) % freq == 0 or epoch == a.max_epochs - 1)

            training_loss = 0

            for _ in range(loader.ntrain):
                X, y = loader.next_batch(0)
                model.step(X, y)
                training_loss += model.loss.data[0]

            training_loss /= loader.ntrain

            if should(a.validation_freq):
                print('validating model')
                validation_loss = 0
                for _ in range(loader.nval):
                    X, y = loader.next_batch(1)
                    model.validate_step(X, y)
                    validation_loss += model.loss.data[0]
                validation_loss /= loader.nval

            if should(a.summary_freq):
                print("recording summary")
                with open(os.path.join(a.output_dir, 'loss_record.txt'), "a") as loss_file:
                    loss_file.write("%s\t%s\t%s\n" % (epoch, training_loss, validation_loss))

            if should(a.progress_freq):
                rate = (epoch + 1) / (time.time() - start)
                remaining = (a.max_epochs - 1 - epoch) / rate
                print("progress  epoch %d  remaining %dh" % (epoch, remaining / 3600))
                print("training loss", training_loss)

            if should(a.display_freq):
                draw(model, os.path.join(a.output_dir, '%s.jpg' % epoch))

            if should(a.save_freq):
                print("saving model")
                model.save(os.path.join(a.output_dir, '%s.pth' % epoch))


if __name__ == '__main__':
    main()


