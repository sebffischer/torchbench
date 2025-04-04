# Use an official Miniconda3 image as the base
FROM continuumio/miniconda3:latest

# Set the working directory inside the container
WORKDIR /app

# Update conda and install some base packages (optional)
RUN conda update -n base -c defaults conda && \
    conda install -y -c conda-forge cmake=3 cudatoolkit=11.8 nvidia/label/cuda-11.8.0::cuda gcc=10 gxx=10 && \
    conda clean -afy

RUN conda install nvidia:cudnn

ENV LD_LIBRARY_PATH="/opt/conda/lib"

ENV KMP_AFFINITY=disabled

# Set the default shell to bash (optional)
SHELL ["/bin/bash", "-c"]

# Install R 4.4.2
RUN conda install -c conda-forge r-base=4.4.2 && \
    conda clean -afy


RUN conda install r-torch -c conda-forge

Run Rscript -e "torch::install_torch()"

# Define default command (optional)
CMD ["bash"]
