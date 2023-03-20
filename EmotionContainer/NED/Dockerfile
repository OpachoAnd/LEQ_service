FROM continuumio/miniconda3

COPY ./NED ./app

# Create the environment:
#COPY environment.yml .
RUN conda env create -f /app/environment.yml


# Activate the environment, and make sure it's activated:
#RUN conda activate NED
SHELL ["conda", "run", "-n", "NED", "/bin/bash", "-c"]

RUN echo "Running env..."
CMD ["bash","/app/run_ned_modules.sh","happy"]

