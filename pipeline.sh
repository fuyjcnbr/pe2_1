#!/bin/bash
echo "creating data ..."
python3 data_creation.py
echo "done"

echo "preprocessing data ..."
python3 data_preprocessing.py
echo "done"

echo "fitting estimator ..."
python3 model_preparation.py
echo "done"

echo "testing estimator quality ..."
python3 model_testing.py
echo "done"
