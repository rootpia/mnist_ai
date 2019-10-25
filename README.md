# MNIST-app backend
Handwritten number recognition

## build
```shell
$ docker build -t ai .
```

## run
```shell
$ docker run -d ai
```

## image dump
```shell
$ docker run -v ./dump:/tmp/mnist/images ai python redisdump.py
```