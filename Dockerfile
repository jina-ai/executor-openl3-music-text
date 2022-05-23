FROM jinaai/jina:3.4.4-py37-perf

# make sure the files are copied into the image
COPY . /executor_root/

WORKDIR /executor_root

RUN  apt update && apt upgrade -y && apt install -y libsndfile1 git

RUN pip install -r requirements.txt

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
