package edu.gsu.dmlab.ml;

import java.util.ArrayList;

import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.opencv.core.Mat;

import edu.gsu.dmlab.datatypes.interfaces.ITrack;
import edu.gsu.dmlab.ml.Consts.CompMethod;

public class TopKSearch {
	ITrack[][] trackArray;
	DbAccess dbAccess;
	ResultWriter rw;

	public TopKSearch(ITrack[][] trackArray, DbAccess dbAccess, ResultWriter rw) {
		this.trackArray = trackArray;
		this.dbAccess = dbAccess;
		this.rw = rw;
	}

	public void run() {
		ITrack[] evlTrks = trackArray[22];
		ArrayList<Double> redundancyList = new ArrayList<Double>();
		ArrayList<Double> accList = new ArrayList<Double>();

		double[] classifyResults = new double[0];
		int[] skipVals = new int[0];
		int[][] dims = this.dbAccess.getOrderedParams(skipVals,
				CompMethod.INTERSECT, 1, evlTrks[0].getFirst().getType());
		Mat[] dataAndLabels = this.dbAccess.getDataAndLabels(evlTrks, dims,
				CompMethod.BHATTACHARYYA);
		double accMean = ClassifyEval.eval(dataAndLabels);

		this.calcForCharts(redundancyList, accList, dims, dataAndLabels);

		int leavIdx = 1;
		for (int dimNum = 2; dimNum < 21; dimNum++) {
			boolean foundNext = false;
			while (!foundNext && leavIdx < 90) {
				System.out.println("----------------");
				int[][] tmpDims = this.dbAccess.getOrderedParams(skipVals,
						CompMethod.INTERSECT, dimNum, evlTrks[0].getFirst()
								.getType());
				Mat[] tmpDataAndLabels = this.dbAccess.getDataAndLabels(
						evlTrks, tmpDims, CompMethod.BHATTACHARYYA);
				double tmpAccMean = ClassifyEval.eval(tmpDataAndLabels);
				if (tmpAccMean > accMean) {
					accMean = tmpAccMean;
					dataAndLabels = tmpDataAndLabels;
					dims = tmpDims;
					leavIdx++;
					foundNext = true;
				} else {
					int[] tmpSkipVals = new int[skipVals.length + 1];
					System.out.print("Leave:");
					for (int i = 0; i < skipVals.length; i++) {
						System.out.print(skipVals[i] + ", ");
						tmpSkipVals[i] = skipVals[i];
					}
					System.out.println(leavIdx);
					tmpSkipVals[skipVals.length] = leavIdx++;
					skipVals = tmpSkipVals;
				}
			}
			if (leavIdx >= 90)
				break;
			// ////////////Eval for graphs below this\\\\\\\\\\\\\\\\\\
			classifyResults = this.calcForCharts(redundancyList, accList, dims,
					dataAndLabels);
		}

		String ext = "Search";
		String type = evlTrks[0].getFirst().getType();
		this.rw.writeFinalAccuracyFile(classifyResults, ext, type);
		this.rw.writeDimsFile(dims, ext, type);

		double[] redundancyArry = new double[redundancyList.size()];
		for (int i = 0; i < redundancyArry.length; i++)
			redundancyArry[i] = redundancyList.get(i);
		this.rw.writeRedundancyFile(redundancyArry, ext, type);

		double[] accArry = new double[accList.size()];
		for (int i = 0; i < accArry.length; i++)
			accArry[i] = accList.get(i);
		this.rw.writeAccuracyFile(accArry, ext, type);

	}

	private double[] calcForCharts(ArrayList<Double> redundancyList,
			ArrayList<Double> accList, int[][] dims, Mat[] dataAndLabels) {

		double[] classifyResults;
		redundancyList.add(RedundancyEval.eval(dataAndLabels));

		Mat[][] dataAndLabelsMatArr = new Mat[this.trackArray.length][];
		for (int i = 0; i < dataAndLabelsMatArr.length; i++) {
			ITrack[] trks = this.trackArray[i];
			Mat[] tmpDataAndLabels = this.dbAccess.getDataAndLabels(trks, dims,
					CompMethod.BHATTACHARYYA);
			dataAndLabelsMatArr[i] = tmpDataAndLabels;
		}

		classifyResults = ClassifyEval.eval(dataAndLabelsMatArr);

		Mean meanCalc = new Mean();
		double mean = meanCalc.evaluate(classifyResults);
		System.out.println("Mean Acc: " + mean);
		accList.add(mean);
		return classifyResults;
	}
}
