#!/bin/bash
python run_summarization.py --mode=decode --data_path=./wh_data/val/val_* --vocab_path=./wh_data/wh_vocab005.txt --log_root=. --exp_name=pretrained_model --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1
