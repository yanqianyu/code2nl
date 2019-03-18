import tensorflow as tf
from seq2seq_model import Seq2SeqModel
from config import ModelConfig
import os
import utils

import numpy as np

from evaluation.bleu.bleu import Bleu
from evaluation.rouge.rouge import Rouge
from evaluation.cider.cider import Cider
from evaluation.meteor.meteor import Meteor


def create_model(sess, config, vocab_to_int_y):
    model = Seq2SeqModel(config, vocab_to_int_y)

    ckpt = tf.train.get_checkpoint_state(config.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print ('Reloading model parameters..')
        model.restore(sess, ckpt.model_checkpoint_path)

    else:
        if not os.path.exists(config.model_dir):
            os.makedirs(config.model_dir)
        print ('Created new model parameters..')
        sess.run(tf.global_variables_initializer())

    return model


def train(model_config):
    corpus_dir = model_config.corpus

    train_x, train_y, train_z = utils.load_pairs("{}/train/train_sbt.json".format(corpus_dir),
                                                 "{}/train/train_nlOut.json".format(corpus_dir),
                                                 "{}/train/train_text.json".format(corpus_dir))

    val_x, val_y, val_z = utils.load_pairs("{}/valid/valid_sbt.json".format(corpus_dir),
                                           "{}/valid/valid_nlOut.json".format(corpus_dir),
                                           "{}/valid/valid_text.json".format(corpus_dir))

    pairs = zip(train_x + val_x, train_y + val_y, train_z + val_z)

    train_len = len(train_x)
    # train_len = 32

    xs, ys, zs, vocab_x, vocab_y, vocab_z, counts = utils.preprocess(pairs)

    vocab_to_int_x, int_to_vocab_x = utils.create_lookup_tables(
        vocab_x, counts, model_config.cutoff_size, model_config.dict_size)
    vocab_to_int_y, int_to_vocab_y = utils.create_lookup_tables(
        vocab_y, counts, model_config.cutoff_size, model_config.dict_size)
    vocab_to_int_z, int_to_vocab_z = utils.create_lookup_tables(
        vocab_z, counts, model_config.cutoff_size, model_config.dict_size)

    vocab_size_x = len(vocab_to_int_x)
    vocab_size_y = len(vocab_to_int_y)
    vocab_size_z = len(vocab_to_int_z)

    xs_ids = [utils.words2ids(x, vocab_to_int_x) for x in xs]
    zs_ids = [utils.words2ids(z, vocab_to_int_z) for z in zs]
    ys_ids = [utils.words2ids(y, vocab_to_int_y, vocab_to_int_y[utils.STOP]) for y in ys]

    max_seq_length = min(max([len(x) for x in xs_ids]), 200)
    xs_ids = [x[:max_seq_length] for x in xs_ids]

    max_seq_length = min(max([len(z) for z in zs_ids]), 200)
    zs_ids = [z[:max_seq_length] for z in zs_ids]

    utils.save_vocab(model_config.vocab_path, {
        "vocab_to_int_x": vocab_to_int_x,
        "vocab_to_int_y": vocab_to_int_y,
        "vocab_to_int_z": vocab_to_int_z,
        "int_to_vocab_x": int_to_vocab_x,
        "int_to_vocab_y": int_to_vocab_y,
        "int_to_vocab_z": int_to_vocab_z
    })

    pad_code = vocab_to_int_x[utils.PAD]

    train_xs = xs_ids[:train_len]
    train_ys = ys_ids[:train_len]
    train_zs = zs_ids[:train_len]

    val_xs = xs_ids[train_len:]
    val_ys = ys_ids[train_len:]
    val_zs = zs_ids[train_len:]

    print("Training model:\n"
          " number of train_pairs={}\n"
          " number of valid pairs={}\n"
          " vocab_size_x={}, \n"
          " vocab_size_y={}, \n"
          " vocab_size_z={}".format(
          len(train_xs), len(val_xs),
          vocab_size_x, vocab_size_y, vocab_size_z))

    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as ses:
        model = create_model(ses, model_config, vocab_to_int_y)
        log_writer = tf.summary.FileWriter(model_config.model_dir, graph=ses.graph)

        loss = 0.0
        for epoch_i in range(model_config.num_epoch):
            for batch_i, (x_batch, y_batch, z_batch, x_lens, y_lens, z_lens) in enumerate(
                  utils.batch_data(train_xs, train_ys, train_zs, model_config.batch_size, pad_code)):
                step_loss, summary = model.train(ses, x_batch, x_lens, y_batch, y_lens, z_batch, z_lens)
                loss += float(step_loss) / model_config.display_freq

                if model.global_step.eval() % model_config.display_freq == 0:
                    print('Epoch ', model.global_epoch_step.eval(), 'Step ', model.global_step.eval(),\
                           ': loss {0:.4f}'.format(loss))
                    loss = 0.0
                    log_writer.add_summary(summary, model.global_step.eval())

                if model.global_step.eval() % model_config.valid_freq == 0:
                    valid_loss = 0.0
                    valid_seen = 0
                    for val_i, (x_batch, y_batch, z_batch, x_lens, y_lens, z_lens) in enumerate(
                        utils.batch_data(val_xs, val_ys, val_zs, model_config.batch_size, pad_code)):
                        step_loss, summary = model.eval(ses, x_batch, x_lens, y_batch, y_lens, z_batch, z_lens)
                        valid_loss += step_loss * model_config.batch_size
                        valid_seen += model_config.batch_size

                    valid_summary = tf.Summary()
                    valid_loss /= valid_seen
                    print('valid loss: {0:.4f}'.format(valid_loss))
                    validloss = valid_summary.value.add()
                    validloss.tag = 'valid loss'
                    validloss.simple_value = valid_loss
                    log_writer.add_summary(valid_summary, model.global_step.eval())
                    log_writer.flush()

                if model.global_step.eval() % model_config.save_freq == 0:
                    print('Saving the model...')
                    checkpoint_path = os.path.join(model_config.model_dir, model_config.model_name)
                    model.save(ses, checkpoint_path, global_step=model.global_step)

            print('Epoch {0:} DONE'.format(model.global_epoch_step.eval()))
            serving_path = os.path.join(model_config.tf_serve_model_path, str(model.global_epoch_step.eval()))
            model.save_serving_model(ses, serving_path)
            model.global_epoch_step_op.eval()


def infer(model_config):
    corpus_dir = model_config.corpus

    test_x, test_y, test_z = utils.load_pairs("{}/test/test_sbt.json".format(corpus_dir),
                                              "{}/test/test_nlOut.json".format(corpus_dir),
                                              "{}/test/test_text.json".format(corpus_dir))

    params = utils.load_vocab(model_config.vocab_path)
    pad_code = params['vocab_to_int_x']['<PAD>']
    stop_code = params['vocab_to_int_x']['<STOP>']

    with tf.Session() as sess:
        gts = {}
        cands = {}
        model = load_model(sess, model_config, params['vocab_to_int_y'])
        for i in range(len(test_x)):
            ids = [i for i in utils.words2ids(utils.parse_input(test_x[i]), params['vocab_to_int_x'])]
            texts = [i for i in utils.words2ids(utils.parse_input(test_z[i]), params['vocab_to_int_z'])]
            input = [ids] * model_config.batch_size
            length = [len(ids)] * model_config.batch_size
            text = [texts] * model_config.batch_size
            text_len = [len(texts)] * model_config.batch_size
            res, att = model.predict(sess, input, length, text, text_len)
            res = res[0]
            res = " ".join([params['int_to_vocab_y'][str(i)] for i in res if i not in (pad_code, stop_code)])
            cands[i] = [res]
            gts[i] = [test_y[i]]
            print(res)

        score_Bleu, scores_Bleu = Bleu(4).compute_score(gts, cands)
        score_Meteor, scores_Meteor = Meteor().compute_score(gts, cands)
        score_Rouge, scores_Rouge = Rouge().compute_score(gts, cands)
        score_Cider, scores_Cider = Cider().compute_score(gts, cands)

        print("score_Bleu: ", score_Bleu)
        print("scores_Bleu: ", len(scores_Bleu))
        print("Bleu_1: ", np.mean(scores_Bleu[0]))
        print("Bleu_2: ", np.mean(scores_Bleu[1]))
        print("Bleu_3: ", np.mean(scores_Bleu[2]))
        print("Bleu_4: ", np.mean(scores_Bleu[3]))

        print("Meteor: ", score_Meteor)
        print("ROUGe: ", score_Rouge)
        print("Cider: ", score_Cider)


def load_model(sess, config, vocab_to_int_y):
    model = Seq2SeqModel(config, vocab_to_int_y)
    ckpt = tf.train.get_checkpoint_state(config.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reloading model parameters...")
        model.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise ValueError(
            'No such file:[{}]'.format(ckpt.model_checkpoint_path)
        )
    return model


def infer_from_model(model_config):
    with tf.Session() as sess:
        signature_key = "serving_default"
        input_sentence = "input_sentence"
        input_sentence_length = "input_sentence_length"

        input_text = "input_text"
        input_text_length = "input_text_length"

        input_keep_prob = "input_keep_prob"

        output_sentence = "output_sentence"
        output_attention_matrix = "output_attention_matrix"

        meta_graph_def = tf.saved_model.loader.load(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            model_config.tf_serve_model_path + "/2"
        )

        # 从meta_graph_def中抽取SignatureDef对象
        signature = meta_graph_def.signature_def

        # 从signature中找到具体输入输出的tensor name
        input_sentence_tensor_name = signature[signature_key].inputs[input_sentence].name
        input_sentence_length_tensor_name = signature[signature_key].inputs[input_sentence_length].name

        input_text_tensor_name = signature[signature_key].inputs[input_text].name
        input_text_length_tensor_name = signature[signature_key].inputs[input_text_length].name

        input_keep_prob_tensor_name = signature[signature_key].inputs[input_keep_prob].name

        output_sentence_tensor_name = signature[signature_key].outputs[output_sentence].name
        output_attention_matrix_tensor_name = signature[signature_key].outputs[output_attention_matrix].name

        # 获取tensor
        input_sentence_inference = sess.graph.get_tensor_by_name(input_sentence_tensor_name)
        input_sentence_length_inference = sess.graph.get_tensor_by_name(input_sentence_length_tensor_name)

        input_text_inference = sess.graph.get_tensor_by_name(input_text_tensor_name)
        input_text_length_inference = sess.graph.get_tensor_by_name(input_text_length_tensor_name)

        input_keep_prob_inference = sess.graph.get_tensor_by_name(input_keep_prob_tensor_name)

        output_sentence_inference = sess.graph.get_tensor_by_name(output_sentence_tensor_name)
        output_attention_matrix_inference = sess.graph.get_tensor_by_name(output_attention_matrix_tensor_name)

        corpus_dir = model_config.corpus

        test_x, test_y, test_z = utils.load_pairs("{}/test/test_sbt.json".format(corpus_dir),
                                                  "{}/test/test_nlOut.json".format(corpus_dir),
                                                  "{}/test/test_text.json".format(corpus_dir))

        params = utils.load_vocab(model_config.vocab_path)
        pad_code = params['vocab_to_int_x']['<PAD>']
        stop_code = params['vocab_to_int_x']['<STOP>']

        for i in range(len(test_x)):
            ids = [i for i in utils.words2ids(utils.parse_input(test_x[i]), params['vocab_to_int_x'])]
            texts = [i for i in utils.words2ids(utils.parse_input(test_z[i]), params['vocab_to_int_z'])]

            # att [infer_step, batch_size, enc_step]
            res, att = sess.run([output_sentence_inference, output_attention_matrix_inference],
                                                 feed_dict={
                                                     input_sentence_inference: [ids] * model_config.batch_size,
                                                     input_sentence_length_inference: [len(ids)] * model_config.batch_size,
                                                     input_text_inference: [texts] * model_config.batch_size,
                                                     input_text_length_inference: [len(texts)] * model_config.batch_size,
                                                     input_keep_prob_inference: 1.0,
                                                 })
            res = res[0]
            res = " ".join([params['int_to_vocab_y'][str(i)] for i in res if i != stop_code])
            att = att[0:len(res.split(' ')), 0, :]
            ids = " ".join([params['int_to_vocab_x'][str(i)] for i in ids if i != stop_code])
            heatmap_name = os.path.join(model_config.picture_path + "/attention_matrix-" + str(i) + ".png")
            utils.plot_attention_matrix(src=ids, trg=res, matrix=att, name=heatmap_name)


if __name__ == "__main__":
    model_config = ModelConfig()
    # train(model_config)
    infer(model_config)
    # infer_from_model(model_config)
