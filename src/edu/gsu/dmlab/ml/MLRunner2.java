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

import org.opencv.core.Core;
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

public class MLRunner2 {

	int span = 12600;
	IHistogramProducer histoProducer;
	float sameLabel = (float) 1.0;
	float diffLabel = (float) -1.0;
	String type;

	Mat[][] labelsArr = null;
	ITrack[][] trackArray = null;
	final int CV_ROW_SAMPLE = 1;
	final int CV_COL_SAMPLE = 0;

	public MLRunner2(IHistogramProducer histoProducer, ITrack[][] trackArray,
			int[][] dims, int compMethod, boolean dualHist) {
		if (histoProducer == null)
			throw new IllegalArgumentException(
					"IHistogramProducer cannot be null in MLRunner constructor.");
		this.histoProducer = histoProducer;

		this.init(dims, trackArray, compMethod, dualHist);
	}

	private void init(int[][] dims, ITrack[][] trackArray, int compMethod,
			boolean dualHist) {

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
						retMat[count++] = getLabels(tracks, dims, compMethod,
								dualHist);
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

	public void run(int modelType, String destFolder) {
		// Set up labels

		ExecutorService exService = Executors.newFixedThreadPool(8);

		ArrayList<Future<double[]>> futures = new ArrayList<Future<double[]>>();
		for (int i = 0; i < this.labelsArr.length; i += 3) {
			final int idx = i;
			final int idx2 = i + 3;
			futures.add(exService.submit(new Callable<double[]>() {
				@Override
				public double[] call() throws Exception {
					// TODO Auto-generated method stub
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

		FileWriter fw = null;
		String fileName = destFolder + File.separator + type.toUpperCase();
		if (modelType == 1) {
			fileName += "Bayseacc.csv";
		} else if (modelType == 2) {
			fileName += "SVMacc.csv";
		} else if (modelType == 3) {
			fileName += "Boostacc.csv";
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

	private CvStatModel getModel(Mat[][] labelsArr, int leavOut, int modelType) {
		ArrayList<Float> classLabels = new ArrayList<Float>();
		ArrayList<Mat> dataList = new ArrayList<Mat>();

		for (int i = 0; i < labelsArr.length; i++) {
			Mat[] labels = labelsArr[i];
			if (i == leavOut)
				continue;
			for (int j = 0; j < labels[0].rows(); j++) {
				Mat vals = labels[0].row(j);
				dataList.add(vals);
				double label = labels[1].get(j, 0)[0];
				classLabels.add((float) label);
			}
		}

		float[] labelArr = new float[classLabels.size()];
		Mat dataMat = new Mat(dataList.size(), dataList.get(0).rows(),
				CvType.CV_32F);
		for (int i = 0; i < labelArr.length; i++) {
			Mat tmpMat = dataList.get(i);
			for (int j = 0; j < tmpMat.rows(); j++)
			dataMat.put(i, j, tmpMat.get(0, j));
			labelArr[i] = classLabels.get(i);
		}

		Mat labelMat = new MatOfFloat(labelArr);

		if (modelType == 1) {
			CvNormalBayesClassifier classifier = new CvNormalBayesClassifier();
			classifier.train(dataMat, labelMat);
			return classifier;
		} else if (modelType == 2) {
			CvSVM classifier = new CvSVM();
			classifier.train(dataMat, labelMat);
			return classifier;
		} else if (modelType == 3) {
			CvBoost classifier = new CvBoost();
			classifier.train(dataMat, CV_ROW_SAMPLE, labelMat);
			return classifier;
		} else {
			return null;
		}
	}

	private Mat[] getLabels(ITrack[] tracks, int[][] dims, int compMethod,
			boolean dualHist) {

		ArrayList<Float> classLabels = new ArrayList<Float>();
		ArrayList<float[]> dataList = new ArrayList<float[]>();
		Random rand = new Random();
		int gcCount = 0;
		for (int i = 0; i < tracks.length; i++) {
			IEvent[] events = tracks[i].getEvents();
			if (events.length > 3) {
				for (int j = 0; j < events.length - 3; j++) {
					Mat sameHist1 = new Mat();
					Mat sameHist2 = new Mat();
					Mat sameHist3 = new Mat();
					Mat sameHist4 = new Mat();
					Mat diffHist1 = new Mat();
					Mat diffHist2 = new Mat();

					// get the two histograms from the same track
					this.histoProducer
							.getHist(sameHist1, events[j], dims, true);
					this.histoProducer.getHist(sameHist2, events[j + 1], dims,
							true);
					this.histoProducer.getHist(sameHist3, events[j + 2], dims,
							false);
					this.histoProducer.getHist(sameHist4, events[j + 3], dims,
							false);

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
					// use it to get a histogram of a different event
					idx = rand.nextInt(tmpEvents.length - 1);
					this.histoProducer.getHist(diffHist1, tmpEvents[idx], dims,
							false);
					this.histoProducer.getHist(diffHist2, tmpEvents[idx + 1],
							dims, false);

					// calculate the distances of the histograms and add them to
					// the
					// list
					double sameVal1 = this.compareHistogram(sameHist2,
							sameHist3, compMethod);
					Mat absDiffMat1 = new Mat();
					Mat absDiffMat2 = new Mat();
					Mat absDiffMat3 = new Mat();
					Core.absdiff(sameHist1, sameHist2, absDiffMat1);
					Core.absdiff(sameHist3, sameHist4, absDiffMat2);
					Core.absdiff(diffHist1, diffHist2, absDiffMat3);
					double sameVal2 = this.compareHistogram(absDiffMat1,
							absDiffMat2, compMethod);

					classLabels.add(sameLabel);
					dataList.add(new float[] { (float) sameVal1,
							(float) sameVal2 });
					double difVal1 = this.compareHistogram(sameHist1,
							diffHist2, compMethod);
					double difVal2 = this.compareHistogram(absDiffMat1,
							absDiffMat3, compMethod);

					classLabels.add(diffLabel);

					dataList.add(new float[] { (float) difVal1, (float) difVal2 });
					gcCount++;
				}
			}
			if (gcCount > 500) {
				System.gc();
				gcCount = 0;
			}
		}
		// now create two Mat objects containing the data and the labels
		// float[][] dataArr = new float[dataList.size()][];
		int numHists;
		if (dualHist) {
			numHists = 2;
		} else {
			numHists = 1;
		}
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

	// private File[] getFiles(File dir, String type) {
	// ArrayList<File> files = new ArrayList<File>();
	//
	// if (dir.isDirectory()) {
	// String[] children = dir.list();
	// for (int i = 0; children != null && i < children.length; i++) {
	// File[] tmpFiles = this.getFiles(new File(dir, children[i]),
	// type);
	// for (int j = 0; j < tmpFiles.length; j++) {
	// files.add(tmpFiles[j]);
	// }
	// }
	// }
	// if (dir.isFile()) {
	// if (dir.getName()
	// .endsWith(type.toUpperCase() + "DustinTracked.txt")) {
	// files.add(dir);
	// }
	// }
	//
	// File[] returnFiles = new File[files.size()];
	// files.toArray(returnFiles);
	// return returnFiles;
	// }
}
