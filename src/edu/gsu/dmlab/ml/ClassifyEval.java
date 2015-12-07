package edu.gsu.dmlab.ml;

import java.util.ArrayList;

import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.ml.CvBoost;
import org.opencv.ml.CvNormalBayesClassifier;
import org.opencv.ml.CvSVM;
import org.opencv.ml.CvStatModel;

public class ClassifyEval {

	static final int CV_ROW_SAMPLE = 1;
	static final int CV_COL_SAMPLE = 0;
	static final int modelType = 1;

	public static double eval(Mat[] labels) {

		@SuppressWarnings("unchecked")
		ArrayList<Mat>[] dataList = new ArrayList[10];
		@SuppressWarnings("unchecked")
		ArrayList<Float>[] classList = new ArrayList[10];

		// Init the array lists
		for (int i = 0; i < dataList.length; i++) {
			dataList[i] = new ArrayList<Mat>();
			classList[i] = new ArrayList<Float>();
		}

		// break the data and class labels up into N different sets
		for (int j = 0; j < labels[0].rows(); j++) {
			dataList[j % dataList.length].add(labels[0].row(j));
			double label = labels[1].get(j, 0)[0];
			classList[j % classList.length].add((float) label);
		}

		// put the sets into an array to call next method
		Mat[][] labelsArr = new Mat[dataList.length][];
		for (int k = 0; k < dataList.length; k++) {
			ArrayList<Mat> data = dataList[k];
			ArrayList<Float> classLabels = classList[k];

			// need an array to construct the mat of labels with
			float[] labelArr = new float[classLabels.size()];
			// need a mat to put the input vectors into
			Mat dataMat = new Mat(data.size(), data.get(0).cols(),
					CvType.CV_32F);
			for (int i = 0; i < labelArr.length; i++) {
				Mat tmpMat = data.get(i);
				for (int j = 0; j < tmpMat.cols(); j++)
					dataMat.put(i, j, tmpMat.get(0, j));
				labelArr[i] = classLabels.get(i);
			}

			Mat labelMat = new MatOfFloat(labelArr);
			labelsArr[k] = new Mat[] { dataMat, labelMat };
		}

		double[] accArr = ClassifyEval.eval(labelsArr);

		Mean meanCalc = new Mean();
		double mean = meanCalc.evaluate(accArr);

		dataList = null;
		classList = null;
		labelsArr = null;
		meanCalc = null;
		System.gc();

		return mean;
	}

	public static double[] eval(Mat[][] dataLabelsArr) {
		double[] accArr = new double[dataLabelsArr.length];
		for (int i = 0; i < accArr.length; i++) {
			CvStatModel model = ClassifyEval.getModel(dataLabelsArr, i,
					modelType);
			accArr[i] = ClassifyEval.calcModelAccuracy(dataLabelsArr[i], model,
					modelType);
		}
		return accArr;
	}

	private static double calcModelAccuracy(Mat[] dataLabels,
			CvStatModel model, int modelType) {
		double count = 0.0;
		double correct = 0.0;

		for (int y = 0; y < dataLabels[0].rows(); y++) {
			double predictedClass = ClassifyEval.predict(model, modelType,
					dataLabels[0].row(y));
			double actual = dataLabels[1].get(y, 0)[0];
			if (predictedClass == actual) {
				correct++;
			}
			count++;
		}

		return correct / count;
	}

	private static CvStatModel getModel(Mat[][] labelsArr, int leavOut,
			int modelType) {
		ArrayList<Float> classLabels = new ArrayList<Float>();
		ArrayList<Mat> dataList = new ArrayList<Mat>();

		// put each of the observations to train on into an arraylist to put
		// into mat at later step
		for (int i = 0; i < labelsArr.length; i++) {
			// check to see if it is the one we want to leave out
			if (i == leavOut)
				continue;
			// get current array of values
			// corresponds to month
			Mat[] labels = labelsArr[i];

			// get each input vector and class label and
			// put them into the respective arraylist
			for (int j = 0; j < labels[0].rows(); j++) {
				Mat vals = labels[0].row(j);
				dataList.add(vals);
				double label = labels[1].get(j, 0)[0];
				classLabels.add((float) label);
			}
		}

		// need an array to construct the mat of labels with
		float[] labelArr = new float[classLabels.size()];
		// need a mat to put the input vectors into
		Mat dataMat = new Mat(dataList.size(), dataList.get(0).cols(),
				CvType.CV_32F);
		for (int i = 0; i < labelArr.length; i++) {
			Mat tmpMat = dataList.get(i);
			for (int j = 0; j < tmpMat.cols(); j++)
				dataMat.put(i, j, tmpMat.get(0, j));
			labelArr[i] = classLabels.get(i);
		}

		Mat labelMat = new MatOfFloat(labelArr);
		classLabels = null;
		dataList = null;

		if (modelType == 1) {
			CvNormalBayesClassifier classifier = new CvNormalBayesClassifier();
			classifier.train(dataMat, labelMat);
			labelMat.release();
			labelMat = null;
			dataMat.release();
			dataMat = null;
			return classifier;
		} else if (modelType == 2) {
			CvSVM classifier = new CvSVM();
			classifier.train(dataMat, labelMat);
			labelMat.release();
			labelMat = null;
			dataMat.release();
			dataMat = null;
			return classifier;
		} else if (modelType == 3) {
			CvBoost classifier = new CvBoost();
			classifier.train(dataMat, CV_ROW_SAMPLE, labelMat);
			labelMat.release();
			labelMat = null;
			dataMat.release();
			dataMat = null;
			return classifier;
		} else {
			labelMat.release();
			labelMat = null;
			dataMat.release();
			dataMat = null;
			return null;
		}
	}

	private static double predict(CvStatModel model, int modelType, Mat value) {
		if (modelType == 1) {
			CvNormalBayesClassifier classifier = (CvNormalBayesClassifier) model;
			return classifier.predict(value);
		} else if (modelType == 2) {
			CvSVM classifier = (CvSVM) model;
			return classifier.predict(value);
		} else if (modelType == 3) {
			CvBoost classifier = (CvBoost) model;
			return classifier.predict(value);
		} else {
			return 0.0;
		}
	}
}
