FROM python:3.11.5-slim

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY back app
COPY GeoVision_dataset GeoVision_dataset
COPY front front

EXPOSE 8000