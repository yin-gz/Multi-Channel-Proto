class HP:
    # datasets
    train_set = "train_wiki"
    val_set = "val_wiki"
    test_set = "val_tac"
    adv_set = None

    # N-way K-shot settings
    batch_size = 4
    trainN = 5
    N = 5
    K = 5
    Q = 5

    # Model params
    model = "multi"
    encoder = "cnn"
    hidden_size = 240
    pair = False
    dropout = 0.2
    max_length = 128

    # Multi channel encoder
    hybird_attention = True
    entity_size = 60  # context_size = hidden_size - 2*entity_size
    parse = True  # parse or not
    parse_path = 'origin_parse'
    word_emb = 50
    pos_emb = 2
    word_att = True

    # Joint learning
    n_clusters = 10
    cluster = False
    pseudo_pth = "train_wiki_and_pseudo_tac"
    feature_pth = "unlabel_features"
    M = 2  # select top 1/M, 0 denotes no selecting
    coef = 0.01

    # Training params
    train_iter = 5000
    val_iter = 1000
    val_step = 1000
    test_iter = 5000
    optim = "sgd"
    lr = 1e-1
    lr_step_size = 4000
    weight_decay = 1e-5
    adv_dis_lr = 1e-1
    adv_enc_lr = 1e-1
    warmup_step = 300

    # Save and load
    load_ckpt = None
    save_ckpt = None
    only_test = False

    # Test on the official website
    test_online = None

    # Others
    fp16 = False
    grad_iter = 1
    na_rate = 0