package edu.gsu.dmlab.ml;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import javax.sql.DataSource;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.imgproc.Imgproc;

import edu.gsu.dmlab.datatypes.interfaces.IEvent;
import edu.gsu.dmlab.datatypes.interfaces.ITrack;
import edu.gsu.dmlab.imageproc.interfaces.IHistogramProducer;
import edu.gsu.dmlab.ml.Consts.CompMethod;

public class DbAccess {

	IHistogramProducer histoProducer;
	DataSource dsourc;

	public DbAccess(DataSource dsourc, IHistogramProducer histoProducer) {
		this.dsourc = dsourc;
		this.histoProducer = histoProducer;
	}

	public Mat[] getDataAndLabels(ITrack[] tracks, int[][] dims,
			CompMethod compMethod) {

		ArrayList<Float> classLabels = new ArrayList<Float>();
		ArrayList<float[]> dataList = new ArrayList<float[]>();
		Random rand = new Random();
		class DataPoint {
			public float classLabel;
			public float[] data;
		}

		ArrayList<Future<DataPoint[]>> labelsFutures = new ArrayList<Future<DataPoint[]>>();
		ExecutorService exService = Executors.newFixedThreadPool(8);

		for (int i = 0; i < tracks.length; i++) {
			IEvent[] events = tracks[i].getEvents();
			if (events.length > 3) {
				for (int j = 0; j < events.length - 1; j++) {

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

					final IEvent ev1 = events[j];
					final IEvent ev2 = events[j + 1];
					final IEvent diffEv = tmpEvents[idx];

					labelsFutures.add(exService
							.submit(new Callable<DataPoint[]>() {
								@Override
								public DataPoint[] call() throws Exception {
									ArrayList<Double> sameDimVals = new ArrayList<Double>();
									ArrayList<Double> diffDimVals = new ArrayList<Double>();
									// process each dim individually
									for (int dimIdx = 0; dimIdx < dims.length; dimIdx++) {
										Mat sameHist1 = new Mat();
										Mat sameHist2 = new Mat();

										Mat diffHist1 = new Mat();

										// get the two histograms from the same
										// track
										int[][] dim = new int[1][];
										dim[0] = dims[dimIdx];
										histoProducer.getHist(sameHist1, ev1,
												dim, true);
										histoProducer.getHist(sameHist2, ev2,
												dim, false);

										// get the histogram from the different
										// track
										histoProducer.getHist(diffHist1,
												diffEv, dim, false);

										// calculate the distances of the
										// histograms and
										// add
										// them to the list
										double sameVal1 = compareHistogram(
												sameHist1, sameHist2,
												compMethod);
										double difVal1 = compareHistogram(
												sameHist1, diffHist1,
												compMethod);
										sameDimVals.add(sameVal1);
										diffDimVals.add(difVal1);

										sameHist1.release();
										sameHist2.release();
										diffHist1.release();

										sameHist1 = null;
										sameHist2 = null;
										diffHist1 = null;
									}
									float[] sameDimValsArr = new float[sameDimVals
											.size()];
									for (int k = 0; k < sameDimVals.size(); k++) {
										sameDimValsArr[k] = (float) ((double) sameDimVals
												.get(k));
									}

									float[] diffDimValsArr = new float[diffDimVals
											.size()];
									for (int k = 0; k < diffDimVals.size(); k++) {
										diffDimValsArr[k] = (float) ((double) diffDimVals
												.get(k));
									}

									DataPoint sameDataPoint = new DataPoint();
									sameDataPoint.data = sameDimValsArr;
									sameDataPoint.classLabel = Consts.SameLabel;

									DataPoint diffDataPoint = new DataPoint();
									diffDataPoint.data = diffDimValsArr;
									diffDataPoint.classLabel = Consts.DiffLabel;
									return new DataPoint[] { sameDataPoint,
											diffDataPoint };
								}
							}));
				}
			}
		}

		while (!labelsFutures.isEmpty()) {
			Future<DataPoint[]> ft = labelsFutures.get(0);
			try {
				DataPoint[] compVals = ft.get(3000, TimeUnit.SECONDS);
				classLabels.add(compVals[0].classLabel);
				dataList.add(compVals[0].data);
				classLabels.add(compVals[1].classLabel);
				dataList.add(compVals[1].data);
				labelsFutures.remove(0);
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (TimeoutException e) {
				System.out.println("Timeout on wait.");
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
		}
		exService.shutdown();

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

	private double compareHistogram(Mat hist1, Mat hist2, CompMethod method) {
		double value = 0;
		switch (method) {
		case CORREL:
			value = Imgproc.compareHist(hist1, hist2, Imgproc.CV_COMP_CORREL);
			break;
		case CHISQR:
			value = Imgproc.compareHist(hist1, hist2, Imgproc.CV_COMP_CHISQR);
			break;
		case INTERSECT:
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

	public int[][] getOrderedParams(int[] leaveOut, CompMethod measure,
			int returnNumber, String eventType) {
		Connection con = null;
		int[][] returnVals = null;
		try {
			con = this.dsourc.getConnection();
			con.setAutoCommit(true);

			String selString = "SELECT wave1, param1 FROM (SELECT * FROM trackingdata."
					+ eventType.toLowerCase()
					+ "_hist_vals "
					+ "INNER JOIN trackingdata.param_combos "
					+ "ON trackingdata."
					+ eventType.toLowerCase()
					+ "_hist_vals.param_combo_id = trackingdata.param_combos.id "
					+ "and trackingdata.param_combos.wave2 IS null) as t1 "
					+ "WHERE measure = "
					+ (measure.ordinal() + 1)
					+ " ORDER by f_val1 DESC LIMIT ?;";
			//System.out.println(selString);
			PreparedStatement selParamsStmt = con.prepareStatement(selString);
			if (leaveOut.length < 1) {
				selParamsStmt.setInt(1, returnNumber);
			} else {
				selParamsStmt.setInt(1, returnNumber + leaveOut.length);
			}
			ResultSet rs = selParamsStmt.executeQuery();
			ArrayList<int[]> paramAndwaveList = new ArrayList<int[]>();
			while (rs.next()) {
				int[] tmpVals = new int[2];
				for (int i = 1; i < tmpVals.length + 1; i++) {
					tmpVals[i - 1] = rs.getInt(i);
				}
				paramAndwaveList.add(tmpVals);
			}

			selParamsStmt.close();
			selParamsStmt = null;

			returnVals = new int[returnNumber][];
			int idx = 0;
			for (int i = 0; i < paramAndwaveList.size(); i++) {
				if ((!this.contains(leaveOut, i)) && idx < returnVals.length) {
					returnVals[idx++] = paramAndwaveList.get(i);
					System.out.println("(Wave, Param): "
							+ paramAndwaveList.get(i)[0] + ", "
							+ paramAndwaveList.get(i)[1]);
				}
			}

		} catch (SQLException ex) {
			System.out.println(ex);
		} finally {
			if (con != null) {
				try {
					con.close();
				} catch (SQLException e) {
					e.printStackTrace();
				}
			}
		}
		return returnVals;
	}

	private boolean contains(final int[] array, final int key) {
		for (final int i : array) {
			if (i == key) {
				return true;
			}
		}
		return false;
	}
}
