package edu.gsu.dmlab.ml;

import java.util.ArrayList;

import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.opencv.core.Mat;

import edu.gsu.dmlab.datatypes.interfaces.ITrack;
import edu.gsu.dmlab.ml.Consts.CompMethod;

public class TopKSelection {
	ITrack[][] trackArray;
	DbAccess dbAccess;
	ResultWriter rw;

	public TopKSelection(ITrack[][] trackArray, DbAccess dbAccess,
			ResultWriter rw) {
		this.trackArray = trackArray;
		this.dbAccess = dbAccess;
		this.rw = rw;
	}

	public void run() {
		ITrack[] evlTrks = trackArray[22];
		ArrayList<Double> redundancyList = new ArrayList<Double>();
		ArrayList<Double> accList = new ArrayList<Double>();
		Mean meanCalc = new Mean();
		double[] classifyResults = new double[0];
		int[][] dims = new int[0][];

		for (int dimNum = 1; dimNum < 21; dimNum++) {
			System.out.println("----------------");
			dims = this.dbAccess.getOrderedParams(new int[0],
					CompMethod.INTERSECT, dimNum, evlTrks[0].getFirst()
							.getType());
			Mat[] dataAndLabels = this.dbAccess.getDataAndLabels(evlTrks, dims,
					CompMethod.BHATTACHARYYA);
			redundancyList.add(RedundancyEval.eval(dataAndLabels));

			Mat[][] dataAndLabelsMatArr = new Mat[this.trackArray.length][];
			for (int i = 0; i < dataAndLabelsMatArr.length; i++) {
				ITrack[] trks = this.trackArray[i];
				Mat[] tmpDataAndLabels = this.dbAccess.getDataAndLabels(trks,
						dims, CompMethod.BHATTACHARYYA);
				dataAndLabelsMatArr[i] = tmpDataAndLabels;
			}

			classifyResults = ClassifyEval.eval(dataAndLabelsMatArr);
			double mean = meanCalc.evaluate(classifyResults);
			System.out.println("Mean Acc: " + mean);
			accList.add(mean);
		}

		String ext = "TopK";
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
}
