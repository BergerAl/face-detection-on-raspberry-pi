# face-detection-on-raspberry-pi

Using OpenCV2 for Face Detection. Runs on Raspberry Pi model B. But takes ~10 seconds with haarcascade.

Model was trained with train_CNN on /data. The result was loss: 0.0926 - acc: 0.9649 - val_loss: 0.0657 - val_acc: 0.9811.

By training with not filtered images /old_data. The result was loss: 0.2127 - acc: 0.9066 - val_loss: 0.4971 - val_acc: 0.8152. Because of this result a preprocessing of the taken image should be used.
