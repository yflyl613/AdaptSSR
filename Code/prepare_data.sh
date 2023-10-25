#!/bin/sh

unzip ../Data/conure_data.zip
mkdir ../model_all
mkdir ../log_all
python data_preprocess.py
