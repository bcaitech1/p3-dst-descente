"--data_root":'data'  # default='data/mwz2.1'
"--train_data":'train_dials.json'
"--dev_data":'dev_dials.json'
"--test_data":'test_dials.json'
"--ontology_data":'ontology.json'
"--slot_meta":'slot_meta.json'
# "--vocab_path":'assets/vocab.txt'
"--bert_config_path":'assets/bert_config_base_uncased.json'
"--bert_ckpt_path":'assets/bert-base-uncased-pytorch_model.bin'
"--save_dir":'outputs'

"--random_seed":42, 
"--num_workers":4, 
"--batch_size":32, 
"--enc_warmup":0.1, 
"--dec_warmup":0.1, 
"--enc_lr":4e-5, 
"--dec_lr":1e-4, 
"--n_epochs":30, 
"--eval_epoch":1, 

"--op_code":"4"
"--slot_token":"[SLOT]"
"--dropout":0.1, 
"--hidden_dropout_prob":0.1, 
"--attention_probs_dropout_prob":0.1, 
"--decoder_teacher_forcing":0.5, 
"--word_dropout":0.1, 
"--not_shuffle_state":False
"--shuffle_p":0.5, 

"--n_history":1, 
"--max_seq_length":512, 
"--msg":None
"--exclude_domain":True # False가 기본값