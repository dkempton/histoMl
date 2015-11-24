package edu.gsu.dmlab.ml;

import javax.sql.DataSource;

import edu.gsu.dmlab.datatypes.interfaces.ITrack;
import edu.gsu.dmlab.imageproc.interfaces.IHistogramProducer;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;

public class SearchRunner {
	String type;
	DataSource dsourc;
	IHistogramProducer histoProducer;
	ITrack[][] trackArray;

	public SearchRunner(DataSource dsourc, IHistogramProducer histoProducer,
			ITrack[][] trackArray) {
		this.type = trackArray[0][0].getFirst().getType();
		this.dsourc = dsourc;
		this.trackArray = trackArray;
		this.histoProducer = histoProducer;
	}

	public void run(String destFolder) {

		for (int compRank = 2; compRank < 5; compRank++) {
			for (int compMeasure = 1; compMeasure < 5; compMeasure++) {
				int[] skipVals = new int[0];
				int dimCount = 1;
				int[][] dims = this.query(skipVals, compRank, dimCount++);
				MLRunner2 runner = new MLRunner2(this.histoProducer,
						this.trackArray, dims, compMeasure, false);
				double accMean = runner.run(1, "", false);
				System.out.println("Mean:" + accMean);
				int leavIdx = 1;
				while (dimCount < 5 && leavIdx < 15) {
					int[][] dims2 = this.query(skipVals, compRank, dimCount);
					MLRunner2 runner2 = new MLRunner2(this.histoProducer,
							this.trackArray, dims2, compMeasure, false);
					double tmpMean = runner2.run(1, "", false);
					System.out.println("Mean:" + tmpMean);
					if (tmpMean > accMean) {
						accMean = tmpMean;
						runner = runner2;
						dims = dims2;
						dimCount++;
						leavIdx++;
					} else {
						int[] tmpSkipVals = new int[skipVals.length + 1];
						for (int i = 0; i < skipVals.length; i++) {
							tmpSkipVals[i] = skipVals.length;
						}
						tmpSkipVals[skipVals.length] = leavIdx++;
						skipVals = tmpSkipVals;
					}
				}
				String ext = "" + compRank + "_" + compMeasure;
				this.writeFile(dims, destFolder, ext);
				for (int modelType = 1; modelType < 4; modelType++) {
					runner.run2(modelType, destFolder, ext);

				}
			}
		}
	}

	private void writeFile(int[][] dims, String destFolder, String extension) {

		FileWriter fw = null;
		String fileName = destFolder + File.separator + this.type.toUpperCase();

		fileName += "Dims" + extension + ".csv";

		try {
			fw = new FileWriter(fileName);
			for (int i = 0; i < dims.length; i++) {
				fw.append(String.valueOf(dims[i][0]) + ", "
						+ String.valueOf(dims[i][1]));
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

	private int[][] query(int[] leaveOut, int measure, int num) {
		Connection con = null;
		int[][] returnVals = null;
		try {
			con = this.dsourc.getConnection();
			con.setAutoCommit(true);

			String selString = "SELECT wave1, param1 FROM (SELECT * FROM trackingdata.ar_hist_vals "
					+ "INNER JOIN trackingdata.param_combos "
					+ "ON trackingdata."
					+ this.type.toLowerCase()
					+ "_hist_vals.param_combo_id = trackingdata.param_combos.id "
					+ "and trackingdata.param_combos.wave2 IS null) as t1 "
					+ "WHERE measure = "
					+ measure
					+ " ORDER by f_val1 DESC LIMIT "
					+ (leaveOut.length + num)
					+ ";";
			// System.out.println(selString);
			PreparedStatement selParamsStmt = con.prepareStatement(selString);

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

			returnVals = new int[num][];
			int idx = 0;
			for (int i = 0; i < paramAndwaveList.size(); i++) {
				if (!this.contains(leaveOut, i) && idx < returnVals.length) {
					returnVals[idx++] = paramAndwaveList.get(i);
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

	public boolean contains(final int[] array, final int key) {
		for (final int i : array) {
			if (i == key) {
				return true;
			}
		}
		return false;
	}
}
