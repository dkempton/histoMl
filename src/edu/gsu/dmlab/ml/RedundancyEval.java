package edu.gsu.dmlab.ml;

import java.util.ArrayList;

import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.opencv.core.Mat;

public class RedundancyEval {
	static PearsonsCorrelation corr = new PearsonsCorrelation();

	public static double eval(Mat[] dataAndLabels) {
		Mat labels = dataAndLabels[1];
		Mat data = dataAndLabels[0];
		@SuppressWarnings("unchecked")
		ArrayList<Double>[] sameFeatures = new ArrayList[data.cols()];
		@SuppressWarnings("unchecked")
		ArrayList<Double>[] diffFeatures = new ArrayList[data.cols()];
		for (int col = 0; col < data.cols(); col++) {
			sameFeatures[col] = new ArrayList<Double>();
			diffFeatures[col] = new ArrayList<Double>();
		}

		for (int row = 0; row < data.rows(); row++) {
			double label = labels.get(row, 0)[0];
			for (int col = 0; col < data.cols(); col++) {
				double val = data.get(row, col)[0];
				if (label == Consts.SameLabel) {
					sameFeatures[col].add(val);
				} else {
					diffFeatures[col].add(val);
				}
			}
		}

		double runningTotal = 0.0;
		if (sameFeatures.length > 1) {
			double tmp1Tot = 0.0;
			for (int i = 0; i < sameFeatures.length - 1; i++) {
				for (int j = i + 1; j < sameFeatures.length; j++) {
					tmp1Tot += RedundancyEval.eval(sameFeatures[i],
							sameFeatures[j]);
				}
			}
			runningTotal = tmp1Tot
					/ (sameFeatures.length * (sameFeatures.length + 1) / 2.0);
			double tmp2Tot = 0.0;
			for (int i = 0; i < diffFeatures.length - 1; i++) {
				for (int j = i + 1; j < diffFeatures.length; j++) {
					tmp2Tot += RedundancyEval.eval(diffFeatures[i],
							diffFeatures[j]);
				}
			}
			runningTotal += tmp2Tot
					/ (diffFeatures.length * (diffFeatures.length + 1) / 2.0);
		} else {
			double tmp1Tot = RedundancyEval.eval(sameFeatures[0],
					diffFeatures[0]);
			runningTotal = tmp1Tot;
		}

		return runningTotal;
	}

	public static double eval(Mat[][] labels) {
		return 0;
	}

	private static double eval(ArrayList<Double> feature1,
			ArrayList<Double> feature2) {
		double[] feat1Arr = new double[feature1.size()];
		double[] feat2Arr = new double[feature2.size()];
		for (int i = 0; i < feature1.size(); i++) {
			feat1Arr[i] = feature1.get(i);
			feat2Arr[i] = feature2.get(i);
		}

		return Math.abs(RedundancyEval.corr.correlation(feat1Arr, feat2Arr));
	}
}
