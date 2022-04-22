from tensorflow import keras
import tensorflow as tf
model = keras.models.load_model('D:\\code\\model\\lenet5_1_model.h5', compile=False)

export_path = 'D:\\code\\model\\pb_test'
model.save(export_path, save_format="tf")

saved_model_dir = 'D:\\code\\model\\pb_test'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open('D:\\code\\model\\lenet5_1_model.tflite', 'wb').write(tflite_model)