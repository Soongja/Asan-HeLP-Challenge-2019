FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender1

RUN pip install cython
RUN pip install SimpleITK
RUN pip install albumentations
RUN pip install tensorboardX
RUN pip install pyYAML
RUN pip install easydict
RUN pip install opencv-python
RUN pip install pandas
RUN pip install pretrainedmodels
RUN pip install efficientnet_pytorch
RUN pip install scikit-learn
RUN pip install pydicom
RUN pip install scikit-image

ENV SRC_DIR /src
COPY src $SRC_DIR
WORKDIR $SRC_DIR

RUN chmod +x ./train.sh ./inference.sh
