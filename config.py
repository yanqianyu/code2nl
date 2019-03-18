class ModelConfig:
    def __init__(self):
        self.batch_size = 64
        self.src_embedding_size = 512
        self.hidden_size = 512
        self.trg_embedding_size = 512
        # self.mode = "train"
        self.use_attention = True
        self.attention_type = "Bahdanau"

        self.x_vocab_size = 58  # sbt
        self.z_vocab_size = 5004  # text
        self.trg_vocab_size = 5004  # nl

        self.optimizer = "adam"
        self.max_grad_norm = 5.0
        self.cell_type = "lstm"
        self.num_layer = 2  # number of RNN layers per encoder and decoder
        self.use_bidir = True  # use bidirectional encoders
        self.learning_rate = 0.001
        self.learning_rate_decay = 0.95
        self.decay_step = 100
        self.keep_prob = 0.5  # dropout
        self.model_dir = "./model_ckpt-exp2"  # checkpoint
        self.model_name = "translate_ckpt"
        self.num_epoch = 10
        self.valid_freq = 250  # 每50个step/batch进行valid
        self.display_freq = 10  # 每50个step/batch打印train loss
        self.save_freq = 500 # 保存模型
        self.max_infer_step = 30  # decode最大步数 避免无限循环
        self.picture_path = "./picture"
        self.tf_serve_model_path = "./code2nl-exp2"
        self.corpus = "./minidata/data-exp2/"

        self.cutoff_size = 1
        self.dict_size = 5003
        self.vocab_path = "vocab.json"