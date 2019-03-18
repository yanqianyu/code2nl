import tensorflow as tf
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense
import utils
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops

# for tensorflow serving
# use SaveModelBuilder to save model

"""
              ---> encoder_cell ---> encoder_layer
  single_cell
              ---> decoder_cell ---> decoder_layer
                   (attention, beam search)                
"""


class Seq2SeqModel(object):
    def __init__(self, config, vocab_to_int_y):
        self.batch_size = config.batch_size
        self.src_embedding_size = config.src_embedding_size
        self.hidden_size = config.hidden_size
        self.trg_embedding_size = config.trg_embedding_size

        self.use_bidir = config.use_bidir

        self.use_attention = config.use_attention
        self.attention_type = config.attention_type

        self.x_vocab_size = config.x_vocab_size
        self.z_vocab_size = config.z_vocab_size
        self.trg_vocab_size = config.trg_vocab_size

        self.optimizer = config.optimizer
        self.max_grad_norm = config.max_grad_norm

        self.cell_type = config.cell_type
        self.num_layer = config.num_layer
        self.max_infer_step = config.max_infer_step  # inference时decoder最大循环次数

        self.learning_rate = config.learning_rate
        self.init_learning_rate = config.learning_rate
        self.decay_step = config.decay_step
        self.learning_rate_decay = config.learning_rate_decay

        self.keep_prob = config.keep_prob
        self.keep_prob_placeholder = tf.placeholder(
            tf.float32, shape=[], name='keep_prob'
        )

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(1, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(
            self.global_epoch_step, self.global_epoch_step + 1
        )
        self.vocab_to_int_y = vocab_to_int_y

        self.build_model()

    def build_model(self):
        self.init_placeholder()
        self.encoder()
        self.decoder()

        self.summary_op = tf.summary.merge_all()

    def init_placeholder(self):
        # encoder inputs
        self.data_x = tf.placeholder(
            dtype=tf.int32, shape=[None, None], name='input_data'
        )

        self.x_length= tf.placeholder(
            dtype=tf.int32, shape=(None,), name='x_length'
        )

        self.data_z = tf.placeholder(
            dtype=tf.int32, shape=[None, None], name='input_text'
        )

        self.z_length = tf.placeholder(
            dtype=tf.int32, shape=(None,), name='z_length'
        )

        # decoder train inputs
        self.data_y = tf.placeholder(
            dtype=tf.int32, shape=[None, None], name='targets'
        )

        self.y_length = tf.placeholder(
            dtype=tf.int32, shape=(None,), name='y_length'
        )

        self.max_y_length = tf.reduce_max(self.y_length)

    def rnn_cell(self, cell_size, num_layers, keep_prob=1.0):
        cells = []
        for i in range(num_layers):
            if self.cell_type.lower() == "lstm":
                cell = tf.contrib.rnn.LSTMCell(
                    cell_size, initializer=tf.random_uniform_initializer(-0.05, 0.05)
                )
            else:
                cell = tf.contrib.rnn.GRUCell(cell_size)

            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.,
                                             input_keep_prob=keep_prob)

            cells.append(cell)

        return tf.contrib.rnn.MultiRNNCell(cells)

    def encoder(self):
        self.sbt_enc_embeddings = tf.Variable(
            tf.random_uniform([self.x_vocab_size,
                               self.src_embedding_size], -0.05, 0.05),
                                dtype=tf.float32)

        sbt_enc_embeded = tf.nn.embedding_lookup(
            self.sbt_enc_embeddings, self.data_x
        )

        self.text_enc_embeddings = tf.Variable(
            tf.random_uniform([self.z_vocab_size,
                               self.src_embedding_size], -0.05, 0.05),
                                dtype=tf.float32)

        text_enc_embeded = tf.nn.embedding_lookup(
            self.text_enc_embeddings, self.data_z
        )

        with tf.variable_scope("sbt_encoder"):
            if not self.use_bidir:
                sbt_enc_cell = self.rnn_cell(self.hidden_size, self.num_layer, self.keep_prob)

                self.sbt_enc_outputs, self.sbt_enc_state = tf.nn.dynamic_rnn(
                    sbt_enc_cell, sbt_enc_embeded, self.x_length, dtype=tf.float32, time_major=False
                )

            else:  # bidirectional rnn
                sbt_enc_cell_fw = self.rnn_cell(self.hidden_size, self.num_layer, self.keep_prob)
                sbt_enc_cell_bw = self.rnn_cell(self.hidden_size, self.num_layer, self.keep_prob)

                self.sbt_enc_outputs, self.sbt_enc_state = tf.nn.bidirectional_dynamic_rnn(
                    sbt_enc_cell_fw, sbt_enc_cell_bw, sbt_enc_embeded,
                    swap_memory=True, sequence_length=self.x_length,
                    dtype=tf.float32
                )

                # concat two lstm's outputs to one tensor
                self.sbt_enc_outputs = tf.concat(self.sbt_enc_outputs, 2)

        with tf.variable_scope("text_encoder"):
            if not self.use_bidir:
                text_enc_cell = self.rnn_cell(self.hidden_size, self.num_layer, self.keep_prob)
                self.text_enc_outputs, self.text_enc_state = tf.nn.dynamic_rnn(
                    text_enc_cell, text_enc_embeded, self.z_length, dtype=tf.float32, time_major=False
                )
            else:
                text_enc_cell_fw = self.rnn_cell(self.hidden_size, self.num_layer, self.keep_prob)
                text_enc_cell_bw = self.rnn_cell(self.hidden_size, self.num_layer, self.keep_prob)

                self.text_enc_outputs, self.text_enc_state = tf.nn.bidirectional_dynamic_rnn(
                    text_enc_cell_fw, text_enc_cell_bw, text_enc_embeded,
                    swap_memory=True, sequence_length=self.z_length,
                    dtype=tf.float32
                )

                # concat two lstm's outputs to one tensor
                self.text_enc_outputs = tf.concat(self.text_enc_outputs, 2)

        # tanh(W[h1;h2])
        self.enc_outputs = tf.concat([self.sbt_enc_outputs, self.text_enc_outputs], 1)
        self.enc_outputs = tf.layers.dense(inputs=self.enc_outputs, units=self.trg_embedding_size,
                                           activation=tf.nn.tanh, trainable=True, use_bias=False)

    def process_decoding_input(self, target_data, vocab_to_int_y, batch_size):
        ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
        return tf.concat([tf.fill([batch_size, 1],
                                  vocab_to_int_y['<START>']), ending], 1)

    def decoder(self):
        data_y = self.process_decoding_input(self.data_y, self.vocab_to_int_y, self.batch_size)

        self.dec_embeddings = tf.Variable(
            tf.random_uniform([self.trg_vocab_size,
                               self.trg_embedding_size], -0.05, 0.05),
                                dtype=tf.float32)

        # trg_emb [batch_size, max_time_step + 1, embedding_size]
        dec_embedded = tf.nn.embedding_lookup(
            self.dec_embeddings, data_y
        )

        with tf.variable_scope("decoder"):
            dec_cell = self.rnn_cell(self.hidden_size, self.num_layer, self.keep_prob)

        out_layer = Dense(self.trg_vocab_size,
                            kernel_initializer=tf.truncated_normal_initializer(
                                  mean=0.0, stddev=0.1))

        attention_mechanism = BahdanauAttention(
            num_units=self.hidden_size, memory=self.enc_outputs,
            memory_sequence_length=self.x_length, normalize=False
        )
        if self.attention_type.lower() == 'luong':
            attention_mechanism = LuongAttention(
                num_units=self.hidden_size, memory=self.enc_outputs,
                memory_sequence_length=self.x_length
            )

        dec_cell = AttentionWrapper(
            dec_cell, attention_mechanism, attention_layer_size=self.hidden_size, alignment_history=True
        )

        init_state = dec_cell.zero_state(
            batch_size=self.batch_size, dtype=tf.float32
        )

        # train
        with tf.variable_scope('decoding'):
            train_helper = TrainingHelper(
                inputs=dec_embedded,
                sequence_length=self.y_length,
                time_major=False,
                name="training_helper"
            )

            train_decoder = BasicDecoder(
                cell=dec_cell,
                helper=train_helper,
                initial_state=init_state,
                output_layer=out_layer
            )

            # train_out.rnn_output [batch_size, max_time_step + 1, trg_vocab_size] if output_time_major=False
            # train_out.sample_id [batch_size], tf.int32
            train_out, final_state, final_sequence_lengths = dynamic_decode(
                decoder=train_decoder,
                output_time_major=False,
                impute_finished=True,
                maximum_iterations=self.max_y_length
            )

            self.decoder_train = train_out.rnn_output

            self.init_optimizer()

        # inference
        with tf.variable_scope('decoding', reuse=True):
            # start_tokens [batch_size,]
            start_tokens = tf.tile(
                tf.constant([self.vocab_to_int_y['<START>']], dtype=tf.int32),
                [self.batch_size]
            )
            end_token = self.vocab_to_int_y['<STOP>']

            infer_helper = GreedyEmbeddingHelper(
                embedding=self.dec_embeddings,
                start_tokens=start_tokens,
                end_token=end_token
            )

            infer_decoder = BasicDecoder(
                cell=dec_cell,
                helper=infer_helper,
                initial_state=init_state,
                output_layer=out_layer
            )

            # dec_outputs_infer.sample_ids [batch_size, max_infer_step]
            # dec_state_infer [decoder_steps, batch_size, encoder_steps]
            self.infer_out, self.infer_state, self.infer_out_length = dynamic_decode(
                decoder=infer_decoder,
                output_time_major=False,
                maximum_iterations=self.max_infer_step
            )

            self.decoder_inference = self.infer_out.sample_id
            # [decoder_steps, batch_size, encoder_steps]

            self.attention_matrix = self.infer_state[4].stack()

        tf.identity(self.decoder_train, 'decoder_train')
        tf.identity(self.decoder_inference, 'decoder_inference')

    def init_optimizer(self):
        # mask: [batch_size, max_time_step + 1]
        masks = tf.sequence_mask(
            self.y_length,
            maxlen=self.max_y_length,
            dtype=tf.float32,
            name="mask"
        )

        # computes per word average cross-entropy over a batch
        self.loss = sequence_loss(
            logits=self.decoder_train,
            targets=self.data_y,  # <STOP>
            weights=masks
            # average_across_timesteps=True,
            # average_across_batch=True
        )

        tf.summary.scalar('train loss', self.loss)

        self.learning_rate = tf.train.exponential_decay(
            self.init_learning_rate, global_step=self.global_step, decay_steps=self.decay_step, decay_rate=self.learning_rate_decay)

        # choose optimizer
        if self.optimizer.lower() == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == 'adam':
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        else:
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        grads = self.opt.compute_gradients(self.loss)

        capped_gradients = [(tf.clip_by_value(grad, -self.max_grad_norm, self.max_grad_norm), var)
                            for grad, var in grads if grad is not None]

        self.train_op = self.opt.apply_gradients(
            capped_gradients, global_step=self.global_step
        )

    def save(self, sess, path, global_step=None):
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_path=path, global_step=global_step)
        print("model saved at %s" % save_path)

    # 模型保存为可用于线上服务的文件
    # 一个.pb文件：保存模型结构等信息
    # 一个variables文件夹：保存所有的变量
    def save_serving_model(self, sess, path):
        builder = tf.saved_model.builder.SavedModelBuilder(path)
        print("exporting trained model to %s" % path)

        # 建立签名映射
        input_sentence = tf.saved_model.utils.build_tensor_info(self.data_x)
        input_sentence_length = tf.saved_model.utils.build_tensor_info(self.x_length)

        input_text = tf.saved_model.utils.build_tensor_info(self.data_z)
        input_text_length = tf.saved_model.utils.build_tensor_info(self.z_length)

        input_keep_prob = tf.saved_model.utils.build_tensor_info(self.keep_prob_placeholder)

        output_sentence = tf.saved_model.utils.build_tensor_info(self.decoder_inference)
        output_attention_matrix = tf.saved_model.utils.build_tensor_info(self.attention_matrix)

        # 定义模型的输入输出，建立调用接口和tensor签名之间的映射
        preidiction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    # "mode": mode,
                    "input_sentence": input_sentence,
                    "input_sentence_length": input_sentence_length,
                    "input_text": input_text,
                    "input_text_length": input_text_length,
                    "input_keep_prob": input_keep_prob
                },
                outputs={
                    "output_sentence": output_sentence,
                    "output_attention_matrix": output_attention_matrix
                },
                # 如果使用tensorflow_model_server部署模型
                # method_name必须为signature_constants中CLASSIFY,PREDICT,REGRESS的一种
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
        )

        # legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

        # 建立模型名称和模型签名之间的映射
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            # 保存模型的方法名，与客户端的request.model_spec.signature_name对应
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    preidiction_signature
            },
            # assets_collection=tf.get_collection()
            # legacy_init_op=legacy_init_op
        )
        builder.save()

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print("model restored from %s" % path)

    def train(self, sess, encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_inputs_length, text_inputs, text_inputs_length):
        input_feed = {}

        input_feed[self.data_x.name] = encoder_inputs
        input_feed[self.x_length.name] = encoder_inputs_length

        input_feed[self.keep_prob_placeholder.name] = self.keep_prob

        input_feed[self.data_y.name] = decoder_inputs
        input_feed[self.y_length.name] = decoder_inputs_length

        input_feed[self.data_z.name] = text_inputs
        input_feed[self.z_length.name] = text_inputs_length

        output_feed = [self.train_op, self.loss, self.summary_op]

        outputs = sess.run(output_feed, input_feed)

        return outputs[1], outputs[2]  # loss, summary

    def eval(self, sess, encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_inputs_length, text_inputs, text_inputs_length):
        # bleu
        # on validation set
        input_feed = {}

        input_feed[self.data_x.name] = encoder_inputs
        input_feed[self.x_length.name] = encoder_inputs_length

        input_feed[self.data_y.name] = decoder_inputs
        input_feed[self.y_length.name] = decoder_inputs_length

        input_feed[self.data_z.name] = text_inputs
        input_feed[self.z_length.name] = text_inputs_length

        input_feed[self.keep_prob_placeholder.name] = 1.0

        output_feed = [self.loss, self.summary_op]
        outputs = sess.run(output_feed, input_feed)

        return outputs[0], outputs[1]  # loss, summary

    def predict(self, sess, encoder_inputs, encoder_inputs_length, text_inputs, text_inputs_length):
        input_feed = {}

        input_feed[self.data_x.name] = encoder_inputs
        input_feed[self.x_length.name] = encoder_inputs_length

        input_feed[self.data_z.name] = text_inputs
        input_feed[self.z_length.name] = text_inputs_length

        input_feed[self.keep_prob_placeholder.name] = 1.0

        output_feed = [self.decoder_inference, self.attention_matrix]
        # output_feed = [self.dec_pred, self.attention_matrix]
        outputs = sess.run(output_feed, input_feed)

        # GreedyDecoder: [batch_size, max_time_step]
        # BeamSearchDecoder: [batch_size, max_time_step, beam_width]
        # attention matrix: [decoder_steps, batch_size, encoder_steps]
        return outputs[0], outputs[1]  # predict sentence, attention matrix
