Different hyperparameters can be compared based on validation loss using `Tensorboard`. To do this, simply specify the folder containing the different versions, then run:
```bash
tensorboard --logdir=/rds/user/al2175/hpc-work/project/data/ellipsoid/FBP_ConvNet/views_143/fbp_convnet/ --port=8081 --bind_all
```
On your local machine, run:
```
ssh -L 8099:gpu-q-X:8081 -fN USER@login-q-X.hpc.cam.ac.uk
```
You only have to adapt the placeholders with `X`, then you can open Tensorboard in a browser on your local machine with:
```bash
http://localhost:8099/
```