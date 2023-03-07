FROM tensorflow/tensorflow:2.7.1-gpu

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
	swig \
	libasound2-dev

RUN addgroup -q --system project && \
    adduser -q --system --home /home/project \
      --shell /bin/bash -uid 1001 --ingroup project project

USER project

WORKDIR /home/project

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


COPY --chown=project:project . .
RUN chmod +x projectRun.sh

ENTRYPOINT ["./projectRun.sh"]