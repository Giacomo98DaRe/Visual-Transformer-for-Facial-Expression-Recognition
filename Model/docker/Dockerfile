FROM pytorch/pytorch:latest

RUN apt-get update
RUN apt-get -y upgrade
RUN apt install -y --no-install-recommends apt-utils

RUN apt-get update && apt-get install -y build-essential cmake git

RUN apt-get install -y libpq-dev

RUN apt-get install -y fonts-liberation

RUN apt-get install -y \
    bc wget curl \
    libfreetype6-dev \
    libzmq3-dev \
    pkg-config \
    software-properties-common \
    unzip && \
    apt-get clean && \
    apt-get autoremove

RUN  wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -u -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

ENV PATH "/opt/conda/bin:$PATH"
ENV PATH "/opt/conda/envs:$PATH"
ENV PATH "/opt/conda/envs/torch/bin:$PATH"

RUN /bin/sh -c "mkdir /.cache"
RUN /bin/sh -c "chmod 777 /.cache"
RUN /bin/sh -c "mkdir /.config"
RUN /bin/sh -c "chmod 777 /.config"

# RUN conda create -n ww_torch --clone base && \
    # echo "conda activate torch" >> ~/.bashrc -> Mi ha dato errore, aggiungo conda init bash così da forzare l'avvio

COPY requirements.yml /requirements.yml

# RUN conda create -n torch --clone base && \
    # echo ". activate torch" >> ~/.bashrc && \
    # /bin/bash -c "conda init bash"

# RUN conda run -n torch pip install facenet-pytorch

RUN conda create -n torch --clone base && \
    echo ". activate torch" >> ~/.bashrc && \
    /bin/bash -c "conda init bash && source ~/.bashrc"

# RUN /bin/bash -c ". activate torch && conda install pip && pip install git+https://github.com/mknnj/kube-apilibrary/#egg=kubernetesdlprofile"

SHELL ["conda", "run", "-n", "torch", "/bin/bash", "-c"]

RUN conda install -c conda-forge pillow pycocotools

RUN pip install facenet-pytorch

RUN conda run -n torch conda install -c conda-forge tensorboard

RUN conda run -n torch conda env update -f=/requirements.yml && conda list

RUN conda run -n torch conda install -c conda-forge numpy

# RUN apt-get install -y ffmpeg libsm6 libxext6

RUN conda run -n torch conda install -c conda-forge scikit-learn

RUN conda run -n torch conda install -c conda-forge pandas

RUN conda run -n torch conda install -c conda-forge tqdm

RUN conda run -n torch conda list



CMD ["bash"]
WORKDIR /exp