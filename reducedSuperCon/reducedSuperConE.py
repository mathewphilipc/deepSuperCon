# Some documentation:
# https://www.tensorflow.org/api_docs/python/tf/feature_column/numeric_column

# various imports
import numpy as np
import tensorflow as tf
import os
import shutil

# Shut off some annoying warning messages
# (Not as ominous as it sounds)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

SUPERCON_TRAINING = "filteredFullChemData.csv"
SUPERCON_TESTING = "filteredFullChemData.csv"

# important note, observed from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/base.py
# about how load_csv_from_header works: In the top row, entry 1 (i.e., header[0])
# gives the number of samples. Entry 2 (i.e., header[1]) gives the number of features
# Output values are given in last col

print("Let's learn when something will superconduct")
def main():

	# Load datasets

	training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
	filename=SUPERCON_TRAINING,
	target_dtype=np.int,
	features_dtype=np.float32)
	print("...\nTraining data imported succesfully")

	# For simplicity we'll use the same set for testing
	# and just monitor our free parameter count to 
	# avoid memorizing

	test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
	filename=SUPERCON_TESTING,
	target_dtype=np.int,
	features_dtype=np.float32)
	print("...\nTest data imported successfully")

	# Each data point has seven real-valued input variables
	# Note that we have 16414 data points, each with 145 feature inputs
	# Also note that 75.84% of materials in list superconduct

	feature_columns = [tf.feature_column.numeric_column("x", shape=[108])]
	print("...\nDefined feature columns")

	# Build 3 layer DNN with [10,20,10] units respectively

	classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
		hidden_units=[50,30,10],
		n_classes=2,
		model_dir="/home/mathew/deepSuperConModels/reducedSuperConModels/reduced_supercon_model_E")

	# Note that model_dir is persistent after training
	# In order to re-run training with, e.g., different hidden unit numbers,
	# you must delete model_dir. To make this easier, I changed the path
	# and stored it in the project directory (TensorFun)
	# Maybe there's a code way to delete it from in here?
	# Look into this tomorrow
	# Update: looked into it, super easy and done

	print("...\nDefined classifier")

	# Define the training inputs

	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": np.array(training_set.data)},
		y=np.array(training_set.target),
		num_epochs=None,
		shuffle=True)

	print("...\nTraining inputs defined")

	#Train mode

	steps = 1000
	multiplier = 500

	for i in range(multiplier):
		progress = ((i+1.0)/multiplier)*100
		classifier.train(input_fn=train_input_fn, steps=steps)
		print("...\nTrained {} steps ({}% complete)".format(steps,progress))

	print("...\nModel trained for {} steps".format(steps*multiplier))

	# Define test inputs

	test_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": np.array(test_set.data)},
		y=np.array(test_set.target),
		num_epochs=1,
		shuffle=False)

	print("...\nTest inputs defined")

	# evaluate accuracy

	accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
	print("...\nAccuracy calculated")
	print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

	# Define new samples to evaluate
	#new_samples = np.array([[1,0,0,1],[0,0,1,0]], dtype=np.float32)
	#predict_input_fn = tf.estimator.inputs.numpy_input_fn(
	#	x={"x":new_samples},
	#	num_epochs=1,
	#	shuffle=False)
	#predictions = list(classifier.predict(input_fn=predict_input_fn))
	#predicted_classes = [p["classes"] for p in predictions]
	#print("New Samples, Class Predictions: {}\n".format(predicted_classes[0]))
	#print("New Samples, Class Predictions: {}\n".format(predicted_classes[1]))



if __name__ == "__main__":
    main()
    # os.remove("FileToRemove.csv")
    # shutil.rmtree("supercon_model", ignore_errors=False, onerror=None)