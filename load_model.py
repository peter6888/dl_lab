import tensorflow as tf
import numpy as np

from group_vector_model import vector_size
from group_vector_model import generate_data, vector_compare_concat, pred_op, loss_op, train_op, divide_data

if __name__ == "__main__":
    l = generate_data(1000)
    dict_l = divide_data(l)
    print(len(dict_l["train"]),len(dict_l["val"]), len(dict_l["test"]))

    tf.reset_default_graph()

    tensor_a = tf.placeholder(shape=[None, vector_size], dtype=tf.float32)
    tensor_b = tf.placeholder(shape=[None, vector_size], dtype=tf.float32)
    tensor_c = tf.placeholder(shape=[None], dtype=tf.int32)
    pred = pred_op(vector_compare_concat(tensor_a, tensor_b))
    loss = loss_op(pred, tensor_c)

    val_data = dict_l["val"]

    # Keep best model
    best_loss = 1e+10
    saver = tf.train.Saver()
    saved_model_path = "/tmp/model.ckpt"

    # Load model and use it
    with tf.Session() as predict_sess:
        #if os.path.exists(saved_model_path):
        saver.restore(predict_sess, saved_model_path)
        print("Model restored.")
        val_a, val_b, val_y = zip(*val_data)
        _c = predict_sess.run(tf.nn.softmax(pred), feed_dict={tensor_a: val_a, tensor_b: val_b})
        y_hat = np.argmax(_c, axis=1)
        print("Validation result:")
        print(np.sum(y_hat==np.array(val_y)) / len(val_y))