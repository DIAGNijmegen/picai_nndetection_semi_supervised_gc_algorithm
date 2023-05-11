FROM joeranbosma/picai_nndetection:1.3

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output /home/user/data \
    && chown algorithm:algorithm /opt/algorithm /input /output /home/user/data

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

# Install algorithm requirements
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -r requirements.txt

# Copy nnDetection results folder
# The required files for nnDetection inference are:
# results/nnDet/.../
# |-- plans_inference.pkl
# |-- fold_0/
# |---- model_best.model
# |---- model_best.model.pkl
# |-- fold_1/...
# |-- fold_2/...
# |-- fold_3/...
# |-- fold_4/...
RUN mkdir -p /opt/algorithm/results/ \
    && chown algorithm:algorithm /opt/algorithm/results/
COPY --chown=algorithm:algorithm results/ /opt/algorithm/results/

# Copy the processor to the algorithm container folder
COPY --chown=algorithm:algorithm process.py /opt/algorithm/

ENTRYPOINT python -m process $0 $@

## ALGORITHM LABELS ##

# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=picai_baseline_nndetection_semi_supervised_processor
