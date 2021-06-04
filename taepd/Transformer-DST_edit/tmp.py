parser = argparse.ArgumentParser()

    "use_cpu",   # Just for my debugging. I have not tested whether it can be used for training model.

    # Using only [CLS]
    "use_cls_only", 

    # w/o re-using dialogue
    "no_dial", 

    # Using only D_t in generation
    "use_dt_only", 

    # By default, "decoder" only attend on a specific [SLOT] position.
    # If using this option, the "decoder" can access to this group of "[SLOT] domain slot - value".
    # NEW: exclude "- value"
    "use_full_slot", 

    "only_pred_op",   # only train to predict state operation just for debugging

    "use_one_optim",   # I use one optim

    "recover_e", 0, 

    # Required parameters
    "data_root", 'data/mwz2.1', 
    "train_data", 'train_dials.json', 
    "dev_data", 'dev_dials.json', 
    "test_data", 'test_dials.json', 
    "ontology_data", 'ontology.json', 
    "vocab_path", 'assets/vocab.txt', 
    "bert_config_path", './assets/bert_config_base_uncased.json', 
    "bert_ckpt_path", './assets/bert-base-uncased-pytorch_model.bin', 
    "save_dir", 'outputs', 

    "random_seed", 42, 
    "num_workers", 0, 
    "batch_size", 32, 
    "enc_warmup", 0.1, 
    "dec_warmup", 0.1, 
    "enc_lr", 3e-5,   # my Transformer-AR uses 3e-5
    "dec_lr", 1e-4, 
    "n_epochs", 30, 
    "eval_epoch", 1, 

    "op_code", "4", 
    "slot_token", "[SLOT]", 
    "dropout", 0.1, 
    "hidden_dropout_prob", 0.1, 
    "attention_probs_dropout_prob", 0.1, 
    "decoder_teacher_forcing", 1, 
    "word_dropout", 0.1, 
    "not_shuffle_state", False, 
    "shuffle_p", 0.5, 

    "n_history", 1, 
    "max_seq_length", 256, 
    "msg", None, 
    "exclude_domain", False, 

    # generator
    'beam_size', , 1,
                        help="Beam size for searching")
    "min_len", 1, 
    'length_penalty', , 0,
                        help="Length penalty for beam search")
    'forbid_duplicate_ngrams', 
    'forbid_ignore_word', , None,
                        help="Ignore the word during forbid_duplicate_ngrams")
    'ngram_size', , 2)