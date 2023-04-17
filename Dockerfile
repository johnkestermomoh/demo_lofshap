FROM python:3.9
WORKDIR /app

# install normal packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# install submodule packages
COPY check/ check/
RUN pip install check/ --upgrade

# copy source code
COPY ./ .

# command to run on container start
CMD ['pytest -vv']
