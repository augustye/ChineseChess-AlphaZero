
def set_session_config(per_process_gpu_memory_fraction=None, allow_growth=None, device_list='0'):
    """

    :param allow_growth: When necessary, reserve memory
    :param float per_process_gpu_memory_fraction: specify GPU memory usage as 0 to 1

    :return:
    """
    import os
    import tensorflow as tf
    import keras.backend as K

    if "COLAB_TPU_ADDR" in os.environ:
        tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
        print('Init TPU session with TPU address:', tpu_address)
        sess = tf.Session(tpu_address, config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        return K.set_session(sess)

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=per_process_gpu_memory_fraction,
            allow_growth=allow_growth,
            log_device_placement=True,
            visible_device_list=device_list
        )
    )
    sess = tf.Session(config=config)
    K.set_session(sess)
