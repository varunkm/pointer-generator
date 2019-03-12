#!/bin/bash
python run_summarization.py --mode=decode --data_path=./wh_data/val/val_* --vocab_path=./wh_data/wh_vocab005.txt --log_root=. --exp_name=model_trained_wh --restore_best_model=1 --single_pass=1
