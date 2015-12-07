package edu.gsu.dmlab.ml;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.CvBoost;
import org.opencv.ml.CvNormalBayesClassifier;
import org.opencv.ml.CvSVM;
import org.opencv.ml.CvStatModel;

import edu.gsu.dmlab.datatypes.interfaces.IEvent;
import edu.gsu.dmlab.datatypes.interfaces.ITrack;
import edu.gsu.dmlab.imageproc.interfaces.IHistogramProducer;

public class MLRunnerSep {

	//int span = 12600;
	IHistogramProducer histoProducer;
	float sameLabel = (float) 1.0;
	float diffLabel = (float) -1.0;
	String type;

	Mat[][] labelsArr = null;
	ITrack[][] trackArray = null;
	final int CV_ROW_SAMPLE = 1;
	final int CV_COL_SAMPLE = 0;
	String extension = "";

	public MLRunnerSep(IHistogramProducer histoProducer, ITrack[][] trackArray,
			int[][] dims, int compMethod) {
		if (histoProducer == null)
			throw new IllegalArgumentException(
					"IHistogramProducer cannot be null in MLRunner constructor.");
		this.histoProducer = histoProducer;

		this.init(dims, trackArray, compMethod);
	}

	public void finalize() {
		for (Mat[] matsArr : this.labelsArr) {
			for (Mat mat : matsArr) {
				mat.release();
				mat = null;
			}
		}
	}

	private void init(int[][] dims, ITrack[][] trackArray, int compMethod) {

		this.type = trackArray[0][0].getFirst().getType();
		ExecutorService exService = Executors.newFixedThreadPool(8);

		ArrayList<Future<Mat[][]>> labelsFutures = new ArrayList<Future<Mat[][]>>();

		for (int i = 0; i < trackArray.length; i += 3) {
			final int idx = i;
			final int idx2 = i + 3;
			labelsFutures.add(exService.submit(new Callable<Mat[][]>() {
				@Override
				public Mat[][] call() throws Exception {
					Mat[][] retMat = new Mat[3][];
					int count = 0;
					for (int j = idx; j < idx2; j++) {
						ITrack[] tracks = trackArray[j];
						retMat[count++] = getLabels(tracks, dims, compMethod);
						System.out.println(j + ":Done");
					}
					return retMat;
				}
			}));

		}

		ArrayList<Mat[]> labelsValues = new ArrayList<Mat[]>();
		while (!labelsFutures.isEmpty()) {
			Future<Mat[][]> ft = labelsFutures.get(0);
			try {
				Mat[][] accVals = ft.get(3000, TimeUnit.SECONDS);
				for (int i = 0; i < accVals.length; i++) {
					labelsValues.add(accVals[i]);
				}
				labelsFutures.remove(0);
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (TimeoutException e) {
				System.out.println("Timeout on wait.");
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
		}
		System.out.println("Done with Labels.");
		this.labelsArr = new Mat[trackArray.length][];
		labelsValues.toArray(this.labelsArr);
		exService.shutdown();
	}

	public double run2(int modelType, String destFolder, String ext) {
		this.extension = ext;
		double val = this.run(modelType, destFolder, true);
		this.extension = "";
		return val;
	}

	public double run(int modelType, String destFolder) {
		return this.run(modelType, destFolder, true);
	}

	public double run(int modelType, String destFolder, boolean write) {
		// Set up labels

		ExecutorService exService = Executors.newFixedThreadPool(8);

		ArrayList<Future<double[]>> futures = new ArrayList<Future<double[]>>();
		for (int i = 0; i < this.labelsArr.length; i += 3) {
			final int idx = i;
			final int idx2 = i + 3;
			futures.add(exService.submit(new Callable<double[]>() {
				@Override
				public double[] call() throws Exception {

					double[] retVals = new double[3];
					int count = 0;
					for (int j = idx; j < idx2; j++) {
						CvStatModel model = getModel(labelsArr, j, modelType);
						double acc = calcModelAccuracy(labelsArr[j], model,
								modelType);
						System.out.println("Correct Percentage:" + acc);
						retVals[count++] = acc;
					}
					return retVals;
				}
			}));
		}

		ArrayList<Double> accValues = new ArrayList<Double>();
		while (!futures.isEmpty()) {
			Future<double[]> ft = futures.get(0);
			if (!ft.isCancelled()) {
				try {
					double[] accVals = ft.get(3000, TimeUnit.SECONDS);
					for (int i = 0; i < accVals.length; i++) {
						accValues.add(accVals[i]);
					}
					futures.remove(0);
				} catch (InterruptedException e) {
					e.printStackTrace();
				} catch (TimeoutException e) {
					System.out.println("Timeout on wait.");
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
			} else {
				if (ft.isDone()) {
					futures.remove(0);
				}
			}

		}
		exService.shutdown();

		if (write) {
			FileWriter fw = null;
			String fileName = destFolder + File.separator + type.toUpperCase();
			if (modelType == 1) {
				fileName += "Bayseacc" + this.extension + ".csv";
			} else if (modelType == 2) {
				fileName += "SVMacc" + this.extension + ".csv";
			} else if (modelType == 3) {
				fileName += "Boostacc" + this.extension + ".csv";
			}

			try {
				fw = new FileWriter(fileName);
				for (int i = 0; i < accValues.size(); i++) {
					fw.append(String.valueOf(accValues.get(i)));
					fw.append("\n");
				}
			} catch (Exception e) {
				System.out.println("Error in CsvFileWriter !!!");
				e.printStackTrace();
			} finally {
				if (fw != null) {
					try {
						fw.flush();
						fw.close();
					} catch (IOException e) {
						e.printStackTrace();
					}
				}
			}
		}

		double[] accArr = new double[accValues.size()];
		for (int i = 0; i < accArr.length; i++) {
			accArr[i] = accValues.get(i);
		}

		Mean meanCalc = new Mean();
		double mean = meanCalc.evaluate(accArr);
		return mean;
	}

	double calcModelAccuracy(Mat[] labels, CvStatModel model, int modelType) {
		double count = 0.0;
		double correct = 0.0;

		for (int y = 0; y < labels[0].rows(); y++) {
			double predictedClass = this.predict(model, modelType,
					labels[0].row(y));
			double actual = labels[1].get(y, 0)[0];
			if (predictedClass == actual) {
				correct++;
			}
			count++;
		}

		return correct / count;
	}

	

	double predict(CvStatModel model, int modelType, Mat value) {
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

	// This causes null pointers when run with dualHists true. If
	// this MLRunner is going to do that type, then this needs to be addressed.
	private CvStatModel getModel(Mat[][] labelsArr, int leavOut, int modelType) {
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

			//get each input vector and class label and
			//put them into the respective arraylist
			for (int j = 0; j < labels[0].rows(); j++) {
				Mat vals = labels[0].row(j);
				dataList.add(vals);
				double label = labels[1].get(j, 0)[0];
				classLabels.add((float) label);
			}
		}

		//need an array to construct the mat of labels with
		float[] labelArr = new float[classLabels.size()];
		//need a mat to put the input vectors into
		Mat dataMat = new Mat(dataList.size(), dataList.get(0).cols(),
				CvType.CV_32F);
		for (int i = 0; i < labelArr.length; i++) {
			Mat tmpMat = dataList.get(i);
			for (int j = 0; j < tmpMat.cols(); j++)
				dataMat.put(i, j, tmpMat.get(0, j));
			labelArr[i] = classLabels.get(i);
		}

		Mat labelMat = new MatOfFloat(labelArr);

		if (modelType == 1) {
			CvNormalBayesClassifier classifier = new CvNormalBayesClassifier();
			classifier.train(dataMat, labelMat);
			dataMat.release();
			dataMat = null;
			return classifier;
		} else if (modelType == 2) {
			CvSVM classifier = new CvSVM();
			classifier.train(dataMat, labelMat);
			dataMat.release();
			dataMat = null;
			return classifier;
		} else if (modelType == 3) {
			CvBoost classifier = new CvBoost();
			classifier.train(dataMat, CV_ROW_SAMPLE, labelMat);
			dataMat.release();
			dataMat = null;
			return classifier;
		} else {
			dataMat.release();
			dataMat = null;
			return null;
		}
	}

	private Mat[] getLabels(ITrack[] tracks, int[][] dims, int compMethod) {

		ArrayList<Float> classLabels = new ArrayList<Float>();
		ArrayList<float[]> dataList = new ArrayList<float[]>();
		Random rand = new Random();
		int gcCount = 0;
		for (int i = 0; i < tracks.length; i++) {
			IEvent[] events = tracks[i].getEvents();
			if (events.length > 3) {
				for (int j = 0; j < events.length - 1; j++) {

					ArrayList<Double> sameDimVals = new ArrayList<Double>();
					ArrayList<Double> diffDimVals = new ArrayList<Double>();

					// get a track that is not this one
					int idx = rand.nextInt(tracks.length);
					IEvent[] tmpEvents = null;
					boolean foundDiffTrack = false;
					while (!foundDiffTrack) {
						if (idx == i) {
							idx = rand.nextInt(tracks.length);
						} else {
							tmpEvents = tracks[idx].getEvents();
							if (tmpEvents.length > 1) {
								foundDiffTrack = true;
							} else {
								idx = rand.nextInt(tracks.length);
							}
						}
					}
					idx = rand.nextInt(tmpEvents.length - 1);

					// process each dim individually
					for (int dimIdx = 0; dimIdx < dims.length; dimIdx++) {
						Mat sameHist1 = new Mat();
						Mat sameHist2 = new Mat();

						Mat diffHist1 = new Mat();

						// get the two histograms from the same track
						int[][] dim = new int[1][];
						dim[0] = dims[dimIdx];
						this.histoProducer.getHist(sameHist1, events[j], dim,
								true);
						this.histoProducer.getHist(sameHist2, events[j + 1],
								dim, false);

						// get the histogram from the different track
						this.histoProducer.getHist(diffHist1, tmpEvents[idx],
								dim, false);

						// calculate the distances of the histograms and add
						// them to the list
						double sameVal1 = this.compareHistogram(sameHist1,
								sameHist2, compMethod);
						double difVal1 = this.compareHistogram(sameHist1,
								diffHist1, compMethod);
						sameDimVals.add(sameVal1);
						diffDimVals.add(difVal1);

						sameHist1.release();
						sameHist2.release();
						diffHist1.release();

						sameHist1 = null;
						sameHist2 = null;
						diffHist1 = null;
					}

					// add the values for the same track comparison
					classLabels.add(sameLabel);
					float[] sameDimValsArr = new float[sameDimVals.size()];
					for (int k = 0; k < sameDimVals.size(); k++) {
						sameDimValsArr[k] = (float) ((double) sameDimVals
								.get(k));
					}
					dataList.add(sameDimValsArr);

					// add the values for the different track comparison
					classLabels.add(diffLabel);
					float[] diffDimValsArr = new float[diffDimVals.size()];
					for (int k = 0; k < diffDimVals.size(); k++) {
						diffDimValsArr[k] = (float) ((double) diffDimVals
								.get(k));
					}
					dataList.add(diffDimValsArr);

					// just for garbage collection, nothing else
					gcCount++;
				}
			}
			if (gcCount > 500) {
				System.gc();
				gcCount = 0;
			}
		}

		// now create two Mat objects containing the data and the labels
		int numHists = dims.length;
		float[] labelArr = new float[classLabels.size()];
		Mat dataMat = new Mat(dataList.size(), numHists, CvType.CV_32F);
		for (int i = 0; i < labelArr.length; i++) {
			float[] vals = dataList.get(i);
			for (int j = 0; j < numHists; j++)
				dataMat.put(i, j, vals[j]);
			labelArr[i] = classLabels.get(i);
		}

		Mat labelMat = new MatOfFloat(labelArr);
		Mat[] retArr = { dataMat, labelMat };
		return retArr;
	}

	private double compareHistogram(Mat hist1, Mat hist2, int method) {
		double value = 0;
		switch (method) {
		case 1:
			value = Imgproc.compareHist(hist1, hist2, Imgproc.CV_COMP_CORREL);
			break;
		case 2:
			value = Imgproc.compareHist(hist1, hist2, Imgproc.CV_COMP_CHISQR);
			break;
		case 3:
			value = Imgproc
					.compareHist(hist1, hist2, Imgproc.CV_COMP_INTERSECT);
			break;
		default:
			value = Imgproc.compareHist(hist1, hist2,
					Imgproc.CV_COMP_BHATTACHARYYA);
			break;
		}

		if (Double.isNaN(value)) {
			return 0.0;
		}
		return value;
	}

}
