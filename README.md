# face-detection-on-raspberry-pi

Using OpenCV2 for Face Detection. Runs on Raspberry Pi model B. But takes ~10 seconds with haarcascade. LBP Cascade need ~2s only, so it is used in the following setup.

Depending on the size of the image the CNN needs for example image of 400x400 pixesl ~5s. By simplifying the network this time should be reduced.

Model was trained with train_CNN on /data. The result was loss: 0.0926 - acc: 0.9649 - val_loss: 0.0657 - val_acc: 0.9811.

By training with not filtered images /old_data. The result was loss: 0.2127 - acc: 0.9066 - val_loss: 0.4971 - val_acc: 0.8152. Because of this result a preprocessing of the taken image should be used.

But there is a problem with unknown faces. A fifth person will be classified as one of the first 4. Working on this problem atm.
