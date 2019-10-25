# -*- coding: utf-8 -*-
import numpy as np
import redis
import cv2


def getRedisDataset():
    db = redis.Redis(host='db', port=6379, db=0)
    keys = db.keys()
    x = np.zeros((len(keys), 784), np.float32)
    y = np.zeros(len(keys), np.int32)

    for ii, keyname in enumerate(keys):
        npimg = np.fromstring(db.hget(keyname, 'img'), np.uint8).astype(np.float32)
        label = int(db.hget(keyname, 'gt'))
        if label==-1: continue
	x[ii] = npimg
        y[ii] = label

    return x, y

def catDatasets(train, x, y):
    train._datasets = (np.vstack((train._datasets[0], x)), np.hstack((train._datasets[1], y)))
    train._length = train._datasets[1].shape[0]
    return train

def dumpRedisDataset(outdir="images"):
    db = redis.Redis(host='db', port=6379, db=0)
    for keyname in db.keys():
        npimg = np.fromstring(db.hget(keyname, 'img'), np.uint8).reshape((28,28))
        label = int(db.hget(keyname, 'gt'))
        filepath = "{}/{}-{}.jpg".format(outdir, keyname, label)
        cv2.imwrite(filepath, npimg)

if __name__ == "__main__":
    #x, y = getRedisDataset()
    #print(x.shape, y.shape)
    #print(y)
    dumpRedisDataset()
