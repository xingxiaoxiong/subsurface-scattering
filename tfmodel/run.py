import os

cmd_str = 'python train.py --mode=train --save_freq=5 --summary_freq=5 --progress_freq=1 --validation_freq=5 --display_freq=1'
os.system(cmd_str)