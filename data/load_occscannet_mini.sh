#!/bin/bash

# run in EmbodiedOcc/data/
python load_dataset.py ./occscannet train_mini_final_sorted.txt
python load_dataset.py ./occscannet test_mini_final_sorted.txt
