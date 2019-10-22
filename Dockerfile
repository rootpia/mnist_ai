FROM chainer/chainer:v1.24.0
LABEL maintainer "rootpia"

RUN apt-get update &&\
  apt-get install git libjpeg-dev libpng-dev python-opencv -y

COPY mnist /tmp/mnist
WORKDIR /tmp/mnist
#RUN python train_mnist.py -u 100 -e 5
COPY pretrained /tmp/mnist/pretrained

# app
RUN pip install flask flask-cors redis

EXPOSE 5000
ENV RECOGNITION_NUM=1

# entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
