# Rafay: local source if building from local. Mirrors docker/Dockerfile's
# `local_src` stage so both paths carry OUR SGLang, not upstream's.
FROM scratch AS local_src
COPY . /src

FROM ubuntu:24.04
SHELL ["/bin/bash", "-c"]

ARG SGLANG_REPO=https://github.com/sgl-project/sglang.git
ARG VER_SGLANG=main

# Rafay: "local" builds the working tree instead of cloning. Default is
# "remote", so upstream behaviour is unchanged and this is inert unless set.
#
# Without this the CPU image could only ever be built from upstream SGLang,
# which is how it came to ship an smg-grpc-servicer that does not match our
# protos -- see kubeless-me/docs/model-engine-cpu-grpc.md.
ARG BRANCH_TYPE=remote

RUN apt-get update && \
    apt-get full-upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ca-certificates \
    git \
    curl \
    wget \
    vim \
    gcc \
    g++ \
    make \
    libsqlite3-dev \
    google-perftools \
    libtbb-dev \
    libnuma-dev \
    numactl

WORKDIR /opt

ENV UV_PYTHON_INSTALL_DIR=/usr/local/share/uv/python
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /root/.local/bin/uvx /usr/local/bin/ && \
    uv venv --python 3.12

RUN echo -e '[[index]]\nname = "torch"\nurl = "https://download.pytorch.org/whl/cpu"\n\n[[index]]\nname = "torchvision"\nurl = "https://download.pytorch.org/whl/cpu"\n\n[[index]]\nname = "torchaudio"\nurl = "https://download.pytorch.org/whl/cpu"\n\n[[index]]\nname = "triton"\nurl = "https://download.pytorch.org/whl/cpu"' > .venv/uv.toml

ENV UV_CONFIG_FILE=/opt/.venv/uv.toml

WORKDIR /sgl-workspace
COPY --from=local_src /src /tmp/local_src

# Rafay: give setuptools-scm a PEP 440 version. With BRANCH_TYPE=local the
# source carries our git history and our tags look like `v0.5.16-rafay.3`,
# which setuptools-scm cannot parse. Same reasoning as docker/Dockerfile.
ARG SETUPTOOLS_SCM_PRETEND_VERSION=""
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION}

RUN source /opt/.venv/bin/activate && \
    if [ "$BRANCH_TYPE" = "local" ]; then \
        cp -r /tmp/local_src sglang; \
    else \
        git clone ${SGLANG_REPO} sglang && \
        cd sglang && git checkout ${VER_SGLANG} && cd ..; \
    fi && \
    rm -rf /tmp/local_src && \
    cd sglang/python && \
    cp pyproject_cpu.toml pyproject.toml && \
    uv pip install . && \
    cd ../sgl-kernel && \
    cp pyproject_cpu.toml pyproject.toml && \
    uv pip install .

ENV SGLANG_USE_CPU_ENGINE=1
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4:/usr/lib/x86_64-linux-gnu/libtbbmalloc.so:/opt/.venv/lib/libiomp5.so
ENV PATH="/opt/.venv/bin:$PATH"
RUN echo 'source /opt/.venv/bin/activate' >> /root/.bashrc

WORKDIR /sgl-workspace/sglang
