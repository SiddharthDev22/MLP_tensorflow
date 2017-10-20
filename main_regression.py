from datetime import datetime
import time
import tensorflow as tf
from dataset_utils import *
from model import NeuralNet
import os

now = datetime.now()
logs_path = "./graph/" + now.strftime("%Y%m%d-%H%M%S")
save_dir = './checkpoints/'


def train(X_train, Y_train,
          X_valid, Y_valid,
          num_epochs=100,
          batch_size=128,
          display=100):

    input_size = X_train.shape[1]
    # num_classes = Y_train.shape[1]

    print()
    print('Training set', X_train.shape, Y_train.shape)
    print('Validation set', X_valid.shape, Y_valid.shape)

    # Creating the alexnet model
    model = NeuralNet(input_size)
    model.inference().loss_regression().train_func()

    # Saving the best trained model (based on the validation accuracy)
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'best_validation')
    best_validation_loss = 1000

    loss_batch_all = np.array([])
    sum_count = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Initialized")
        merged = tf.summary.merge_all()
        batch_writer = tf.summary.FileWriter(logs_path + '/batch/', sess.graph)
        valid_writer = tf.summary.FileWriter(logs_path + '/valid/')
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            print('-----------------------------------------------------------------------------')
            print('Epoch: {}'.format(epoch + 1))
            X_train, Y_train = randomize(X_train, Y_train)
            step_count = int(len(X_train) / batch_size)
            for step in range(step_count):
                start = step * batch_size
                end = (step + 1) * batch_size
                X_batch, Y_batch = get_next_batch(X_train, Y_train, start, end)

                model.is_train = True
                feed_dict_batch = {model.x: X_batch, model.y: Y_batch, model.keep_prob: 0.8}
                _, loss_batch = sess.run([model.train_op, model.loss],
                                                    feed_dict=feed_dict_batch)
                loss_batch_all = np.append(loss_batch_all, loss_batch)

                if step > 0 and not (step % display):
                    mean_loss = np.mean(loss_batch_all)
                    print(
                        "Step {0}, training loss: {1:.5f}".format(step, mean_loss))
                    batch_writer.add_summary(summary_tr, sum_count * display)
                    summary_tr = tf.Summary(value=[tf.Summary.Value(tag='Loss', simple_value=mean_loss)])
                    batch_writer.add_summary(summary_tr, sum_count * display)
                    summary = sess.run(merged, feed_dict=feed_dict_batch)
                    batch_writer.add_summary(summary, sum_count * display)
                    sum_count += 1
                    acc_batch_all = loss_batch_all = np.array([])


            model.is_train = False
            feed_dict_val = {model.x: X_valid, model.y: Y_valid, model.keep_prob: 1}
            loss_valid = sess.run(model.loss, feed_dict=feed_dict_val)
            if loss_valid < best_validation_loss:
                # Update the best-known validation accuracy.
                best_validation_loss = loss_valid
                best_epoch = epoch
                # Save all variables of the TensorFlow graph to file.
                saver.save(sess=sess, save_path=save_path)
                # A string to be printed below, shows improvement found.
                improved_str = '*'
            else:
                # An empty string to be printed below.
                # Shows that no improvement was found.
                improved_str = ''
            epoch_time = time.time() - epoch_start_time
            print("Epoch {0}, run time: {1:.1f} seconds, validation loss: {2:.2f}{3}"
                  .format(epoch + 1, epoch_time, loss_valid, improved_str))


if __name__ == '__main__':
    # %% Loading the input data
    # datasetNo = 1, 2
    X_train, Y_train, X_valid, Y_valid = read_dataset('Dataset.xlsx',
                                                      datasetNo=2,
                                                      normalize=False,
                                                      type="regression")

    train(X_train, Y_train,
          X_valid, Y_valid,
          num_epochs=1500,
          batch_size=128,
          display=100)