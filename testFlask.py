from flask import Flask, render_template, request, Response, jsonify
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorflow as tf
import grpc
import utils
from config import ModelConfig
import get_ast
import get_sbt
import numpy as np
import json
from datetime import datetime
from flask_cors import CORS


tf.app.flags.DEFINE_string('server', 'localhost:9000', 'PredictionService host:port')

FLAGS = tf.app.flags.FLAGS

app = Flask(__name__)
CORS(app, resources=r'/*')
app.debug = True


class mainSessRunning():
    def __init__(self):
        host_port = FLAGS.server
        MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024
        channel = grpc.insecure_channel(host_port, options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                                                   ('grpc.max_receive_message_length',
                                                                    MAX_MESSAGE_LENGTH)])
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

        self.request = predict_pb2.PredictRequest()
        # 启动tensorflow serving时配置的model_name
        self.request.model_spec.name = "code2nl"

        # 保存模型时的方法名
        # request.model_spec.signature_name = "serving_default"
        self.request.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    def inference(self, text):
        # text -> sbt -> num -> src
        ast = get_ast.get_ast(text)
        sbt, _ = get_sbt.get_sbt(ast)
        cf = ModelConfig()

        params = utils.load_vocab(cf.vocab_path)

        ids = [i for i in utils.words2ids(utils.parse_input(sbt), params['vocab_to_int_x'])]
        src = [ids] * cf.batch_size
        src_len = [len(ids)] * cf.batch_size

        self.request.inputs["input_sentence"].CopyFrom(
            tf.contrib.util.make_tensor_proto(src, dtype=tf.int32)
        )
        self.request.inputs["input_sentence_length"].CopyFrom(
            tf.contrib.util.make_tensor_proto(src_len, dtype=tf.int32)
        )

        self.request.inputs["input_keep_prob"].CopyFrom(
            tf.contrib.util.make_tensor_proto(1.0, dtype=tf.float32)
        )

        # 20.0s超时
        a = datetime.now()
        response = self.stub.Predict(self.request, 500.0)
        b = datetime.now()

        predict_ids = response.outputs["output_sentence"]
        attention_matrix = response.outputs["output_attention_matrix"]

        # [batch_size, max_infer_step, beam_width]
        predict_ids = tf.contrib.util.make_ndarray(predict_ids)
        attention_matrix = tf.contrib.util.make_ndarray(attention_matrix)

        # num2word
        predict_ids = predict_ids[0]  # [max_time_step]
        # predict_ids = np.transpose(predict_ids)

        pad_code = params['vocab_to_int_x']['<PAD>']
        stop_code = params['vocab_to_int_x']['<STOP>']

        res_hyps = ' '.join([params['int_to_vocab_y'][str(i)] for i in predict_ids if i not in (pad_code, stop_code)])

        attention_matrix = attention_matrix[0:len(res_hyps), 0, :]  # beam width * batch_size

        att_mat = attention_matrix[0: len(res_hyps.split()), :]

        return res_hyps, att_mat, str((b-a).seconds), sbt


run = mainSessRunning()


# 127.0.0.1:5000/test
@app.route('/test', methods=['GET', 'POST'])
def test():
    text = request.get_json('data')
    rawcode = text['rawcode']
    res_hyps, res_att, time, res_sbt = run.inference(rawcode)

    res_att = res_att.tolist()
    return json.dumps({"statements": res_hyps, "attention": res_att, "sbt": res_sbt})


if __name__ == "__main__":
    app.run()

'''
protected boolean runTestsOnEdt(){
    return true;
}
'''
