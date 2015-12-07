package edu.gsu.dmlab.ml;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class ResultWriter {
	String folder;

	public ResultWriter(String folder) {
		this.folder = folder;
	}

	public void writeDimsFile(int[][] dims, String extension, String eventType) {

		FileWriter fw = null;
		String fileName = this.folder + File.separator
				+ eventType.toUpperCase();

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

	public void writeRedundancyFile(double[] accVals, String extension,
			String eventType) {
		FileWriter fw = null;
		String fileName = this.folder + File.separator
				+ eventType.toUpperCase();

		fileName += "Redundancy" + extension + ".csv";

		try {
			fw = new FileWriter(fileName);
			for (int i = 0; i < accVals.length; i++) {
				fw.append(String.valueOf(i + 1) + ", "
						+ String.valueOf(accVals[i]));
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

	public void writeAccuracyFile(double[] accVals, String extension,
			String eventType) {
		FileWriter fw = null;
		String fileName = this.folder + File.separator
				+ eventType.toUpperCase();

		fileName += "Acc" + extension + ".csv";

		try {
			fw = new FileWriter(fileName);
			for (int i = 0; i < accVals.length; i++) {
				fw.append(String.valueOf(i + 1) + ", "
						+ String.valueOf(accVals[i]));
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

	public void writeFinalAccuracyFile(double[] accVals, String extension,
			String eventType) {
		FileWriter fw = null;
		String fileName = this.folder + File.separator
				+ eventType.toUpperCase();

		fileName += "AccFinal" + extension + ".csv";

		try {
			fw = new FileWriter(fileName);
			for (int i = 0; i < accVals.length; i++) {
				fw.append(String.valueOf(accVals[i]));
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
}
