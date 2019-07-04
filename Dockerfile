FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# install base tools
# Install some dependencies
RUN apt-get update && apt-get install -y \
		bc \
		build-essential \
		cmake \
		curl \
		g++ \
		gfortran \
		git \
		libopenblas-dev \
		pkg-config \
		software-properties-common \
		unzip \
		vim \
		wget \
#		qt4-qmake \
#		qt5-qmake \
#		libqt4-dev \
		libjpeg-dev \
		libpng-dev \
		libtiff5-dev \	
                libsm6 \
                libxext6 \
                libxrender1 \
#		doxygen \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/* && \
# Link BLAS library to use OpenBLAS using the alternatives mechanism (https://www.scipy.org/scipylib/building/linux.html#debian-ubuntu)
	update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3

RUN wget http://download.redis.io/redis-stable.tar.gz \
        && tar xvzf redis-stable.tar.gz \
        && cd redis-stable \
        && make \
        && make install

RUN wget --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && /opt/conda/bin/conda config --set always_yes yes --set changeps1 no \
#    && rm ~/miniconda.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc 
#    && echo "conda activate base" >> ~/.bashrc

ENV PATH=/opt/conda/bin:${PATH}

RUN conda install tensorflow-gpu==1.8.0 
RUN pip install --upgrade pip
RUN pip install flask flask-httpauth flask-sqlalchemy passlib redis opencv-python Pillow

# Install keras from source
RUN git clone https://github.com/keras-team/keras.git /root/keras && \
	cd /root/keras && \
        git checkout a2d11d4724d3cf4a0b18a7fe8448723d92e1c716 && \
	python setup.py install && \
	cd /root

# create directory
RUN mkdir /root/foodlg /root/dataset /root/model-zoo /root/database

# copy configurations
#COPY ./config /foodlg/config
#COPY ./*.py /root/foodlg/
#COPY ./class_indices /root/foodlg/class_indices

# Expose Ports for flask (5000), TensorBoard (6006)
EXPOSE 5000 6006

WORKDIR /root/foodlg
ENTRYPOINT ["./docker_entrypoint.sh"]
