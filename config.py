import tensorflow as tf

flags = tf.app.flags

############################
#    hyper parameters      #
############################
flags.DEFINE_float('ac_lambda0', 0.01, '\lambda in the activation function a_c, iteration 0')
flags.DEFINE_float('ac_lambda_step', 0.01,
                   'It is described that \lambda increases at each iteration with a fixed schedule, however specific super parameters is absent.')

flags.DEFINE_float('initial_learning_rate', 1e-3, '\lambda in the activation function a_c, iteration 0')

flags.DEFINE_integer('batch_size', 25, 'batch size')
flags.DEFINE_integer('num_epochs', 50, 'number of epochs')
flags.DEFINE_integer('iter_routing', 1, 'number of iterations')
flags.DEFINE_float('m_schedule', 0.2, 'the m will get to 0.9 at current epoch')
flags.DEFINE_float('epsilon', 1e-9, 'epsilon')
flags.DEFINE_float('m_plus', 0.9, 'the parameter of m plus')
flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')
flags.DEFINE_float('lambda_val', 0.5, 'down weight of the loss for absent digit classes')
flags.DEFINE_boolean('weight_reg', False, 'train with regularization of weights')
flags.DEFINE_string('norm', 'norm2', 'norm type')
################################
#    structure parameters      #
################################
flags.DEFINE_integer('A', 32, 'number of channels in output from ReLU Conv1')
flags.DEFINE_integer('B', 8, 'number of capsules in output from PrimaryCaps')
flags.DEFINE_integer('C', 16, 'number of channels in output from ConvCaps1')
flags.DEFINE_integer('D', 16, 'number of channels in output from ConvCaps2')

############################
#   environment setting    #
############################
flags.DEFINE_string('dataset', 'data/mnist', 'the path for dataset')
flags.DEFINE_boolean('use_masked', True, 'whether or not to use masked images')
flags.DEFINE_string('phase', 'train', 'train or predict phase')
flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueueing exampls')
flags.DEFINE_string('logdir', 'logs', 'logs directory')
flags.DEFINE_string('test_logdir', 'test_logs', 'test logs directory')

cfg = tf.app.flags.FLAGS

