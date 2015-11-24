package edu.gsu.dmlab.ml;

import java.io.File;

import javax.sql.DataSource;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.opencv.core.Core;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import snaq.db.DBPoolDataSource;
import edu.gsu.dmlab.ObjectFactory;
import edu.gsu.dmlab.databases.interfaces.IImageDBConnection;
import edu.gsu.dmlab.datatypes.interfaces.ITrack;
import edu.gsu.dmlab.exceptions.InvalidConfigException;
import edu.gsu.dmlab.imageproc.interfaces.IHistogramProducer;

public class HistoCompML {

	DataSource imageDBPoolSourc;
	IImageDBConnection imageDBConnect;
	DataSource trackingDBPoolSourc;
	IHistogramProducer histProd;

	int imageCacheSize;

	int[] wavelengths = { 94, 131, 171, 193, 211, 304, 335, 1600, 1700 };
	String sourceFile = "F:\\files\\exp3\\";

	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		HistoCompML ml = new HistoCompML();
		ml.run();
	}

	public HistoCompML() {
		String configFile = "config";

		try {
			this.config(configFile);
			this.imageDBConnect = ObjectFactory.getImageDBConnection(
					this.imageDBPoolSourc, this.imageCacheSize);
			this.histProd = ObjectFactory.getHistoProducer(this.imageDBConnect,
					this.wavelengths);

		} catch (InvalidConfigException e) {
			e.printStackTrace();
		}
	}

	public void run2() {
		String mainDestFolder = "C:\\Users\\Dustin\\Google Drive\\Lab Research\\Appearance Models\\spamsWork\\Data\\";
		ITrack[][] arTracks = TrackReader.getTracks(this.sourceFile, "AR");
		ITrack[][] chTracks = TrackReader.getTracks(this.sourceFile, "CH");
		{
			String destFolder1 = mainDestFolder + "OldHist";
			this.runCrossValSingle(destFolder1, arTracks, chTracks, false);
		}
	}

	public void run() {
		String mainDestFolder = "C:\\Users\\Dustin\\Google Drive\\Lab Research\\Summer15\\Data\\";
		ITrack[][] arTracks = TrackReader.getTracks(this.sourceFile, "AR");
		ITrack[][] chTracks = TrackReader.getTracks(this.sourceFile, "CH");
		// {
		// String destFolder1 = mainDestFolder + "Best1";
		// String destFolder2 = mainDestFolder + "Best2";
		// this.runBest(destFolder1, arTracks, chTracks, false);
		// //this.runBest(destFolder2, arTracks, chTracks, true);
		// }
		// {
		// String destFolder1 = mainDestFolder + "Mid1";
		// String destFolder2 = mainDestFolder + "Mid2";
		// this.runMid(destFolder1, arTracks, chTracks, false);
		// //this.runMid(destFolder2, arTracks, chTracks, true);
		// }
		// {
		// String destFolder1 = mainDestFolder + "Worst1";
		// String destFolder2 = mainDestFolder + "Worst2";
		// this.runWorst(destFolder1, arTracks, chTracks, false);
		// // this.runWorst(destFolder2, arTracks, chTracks, true);
		// }

		// {
		// String destFolder1 = mainDestFolder + "TopK";
		// this.runTopK(destFolder1, arTracks, chTracks, false);
		// // this.runWorst(destFolder2, arTracks, chTracks, true);
		// }

		{
			String destFolder = mainDestFolder + "TopK+";
			{
				SearchRunner runner = new SearchRunner(
						this.trackingDBPoolSourc, this.histProd, arTracks);
				runner.run(destFolder);
			}
			{
				SearchRunner runner = new SearchRunner(
						this.trackingDBPoolSourc, this.histProd, chTracks);
				runner.run(destFolder);
			}
		}

	}

	void runCrossValSingle(String destFolder, ITrack[][] arTracks,
			ITrack[][] chTracks, boolean dualHist) {

		// AR
		{
			int[][] dims = { { 6, 3 }, { 6, 7 }, { 6, 9 }, { 6, 10 } };// AR
			int histMeasure = 4;
			MLRunner2 runner = new MLRunner2(this.histProd, arTracks, dims,
					histMeasure, dualHist);
			int modelType = 1;
			// for (int modelType = 1; modelType < 4; modelType++) {
			runner.run(modelType, destFolder);
			// }
		}
		// CH
		{
			int[][] dims = { { 6, 2 }, { 6, 3 }, { 6, 6 }, { 6, 9 } };// CH

			int histMeasure = 4;
			MLRunner2 runner = new MLRunner2(this.histProd, chTracks, dims,
					histMeasure, dualHist);
			int modelType = 1;
			// for (int modelType = 1; modelType < 4; modelType++) {
			runner.run(modelType, destFolder);
			// }
		}

	}

	void runTopK(String destFolder, ITrack[][] arTracks, ITrack[][] chTracks,
			boolean dualHist) {

		// AR
		{

			int[][][] dimsArr = { { { 1, 6 }, { 7, 6 }, { 8, 2 }, { 9, 2 } },// With
																				// 1
					{ { 1, 6 }, { 6, 4 }, { 5, 3 }, { 5, 1 } },// With 2
					{ { 8, 2 }, { 9, 3 }, { 8, 3 }, { 5, 2 } },// With 3
					{ { 8, 3 }, { 5, 1 }, { 3, 2 }, { 7, 3 } } };// With 4

			for (int i = 0; i < 4; i++) {
				for (int histMeasure = 1; histMeasure < 5; histMeasure++) {
					MLRunner2 runner = new MLRunner2(this.histProd, arTracks,
							dimsArr[i], histMeasure, dualHist);
					for (int modelType = 1; modelType < 4; modelType++) {
						String ext = "" + (i + 1) + "_" + histMeasure;
						runner.run2(modelType, destFolder, ext);
					}
				}
			}
		}
		// CH
		{
			int[][][] dimsArr = { { { 7, 7 }, { 5, 2 }, { 2, 2 }, { 1, 2 } },// With
					// 1
					{ { 7, 6 }, { 1, 6 }, { 7, 7 }, { 2, 2 } },// With 2
					{ { 8, 2 }, { 9, 2 }, { 9, 3 }, { 5, 2 } },// With 3
					{ { 3, 3 }, { 3, 1 }, { 3, 2 }, { 2, 1 } } };// With 4
			for (int i = 0; i < 4; i++) {
				for (int histMeasure = 1; histMeasure < 5; histMeasure++) {

					MLRunner2 runner = new MLRunner2(this.histProd, chTracks,
							dimsArr[i], histMeasure, dualHist);
					for (int modelType = 1; modelType < 4; modelType++) {
						String ext = "" + (i + 1) + "_" + histMeasure;
						runner.run2(modelType, destFolder, ext);
					}
				}
			}
		}

	}

	void runBest(String destFolder, ITrack[][] arTracks, ITrack[][] chTracks,
			boolean dualHist) {

		// AR
		{
			int[][] dims = { { 1, 6 }, { 3, 1 }, { 4, 1 }, { 7, 9 } };// Best//
																		// 1//AR
			int histMeasure = 1;
			MLRunner2 runner = new MLRunner2(this.histProd, arTracks, dims,
					histMeasure, dualHist);
			for (int modelType = 1; modelType < 4; modelType++) {
				runner.run(modelType, destFolder);
			}
		}
		// CH
		{
			int[][] dims = { { 1, 10 }, { 7, 7 }, { 9, 1 } };// 1//CH//Best

			int histMeasure = 1;
			MLRunner2 runner = new MLRunner2(this.histProd, chTracks, dims,
					histMeasure, dualHist);
			for (int modelType = 1; modelType < 4; modelType++) {
				runner.run(modelType, destFolder);
			}
		}

	}

	void runMid(String destFolder, ITrack[][] arTracks, ITrack[][] chTracks,
			boolean dualHist) {

		// AR
		{
			int[][] dims = { { 1, 7 }, { 4, 6 }, { 6, 6 }, { 8, 5 } };// 1//AR//Mid

			int histMeasure = 1;
			MLRunner2 runner = new MLRunner2(this.histProd, arTracks, dims,
					histMeasure, dualHist);
			for (int modelType = 1; modelType < 4; modelType++) {
				runner.run(modelType, destFolder);
			}
		}
		// CH
		{
			int[][] dims = { { 1, 6 }, { 3, 5 }, { 7, 2 }, { 8, 4 } };// 3//CH//Mid

			int histMeasure = 3;
			MLRunner2 runner = new MLRunner2(this.histProd, chTracks, dims,
					histMeasure, dualHist);
			for (int modelType = 1; modelType < 4; modelType++) {
				runner.run(modelType, destFolder);
			}
		}

	}

	void runWorst(String destFolder, ITrack[][] arTracks, ITrack[][] chTracks,
			boolean dualHist) {

		// AR
		{
			int[][] dims = { { 6, 8 } };// 2//Worst

			int histMeasure = 2;
			MLRunner2 runner = new MLRunner2(this.histProd, arTracks, dims,
					histMeasure, dualHist);
			for (int modelType = 1; modelType < 4; modelType++) {
				runner.run(modelType, destFolder);
			}
		}
		// CH
		{
			int[][] dims = { { 3, 10 }, { 4, 10 }, { 8, 10 } };// 2//Worst

			int histMeasure = 2;
			MLRunner2 runner = new MLRunner2(this.histProd, chTracks, dims,
					histMeasure, dualHist);
			for (int modelType = 1; modelType < 4; modelType++) {
				runner.run(modelType, destFolder);
			}
		}

	}

	// /////////////////////////////////////////////////////////////////////////////////
	// Start of configuration methods
	// ////////////////////////////////////////////////////////////////////////////////
	public void config(String folderLocation) throws InvalidConfigException {
		try {
			DocumentBuilderFactory fctry = DocumentBuilderFactory.newInstance();
			Document doc;
			String fileLoc = folderLocation + File.separator
					+ "histocomp.cfg.xml";
			DocumentBuilder bldr = fctry.newDocumentBuilder();
			doc = bldr.parse(new File(fileLoc));
			doc.getDocumentElement().normalize();

			Element root = doc.getDocumentElement();
			NodeList ndLst = root.getChildNodes();
			for (int i = 0; i < ndLst.getLength(); i++) {
				Node nde = ndLst.item(i);
				if (nde.getNodeType() == Node.ELEMENT_NODE) {
					String ndName = nde.getNodeName();
					if (ndName.compareTo("imagepool") == 0) {
						this.imageDBPoolSourc = this.getPoolSourc(nde
								.getChildNodes());
					} else if (ndName.compareTo("trackingpool") == 0) {
						this.trackingDBPoolSourc = this.getPoolSourc(nde
								.getChildNodes());
					}
				}
			}

			this.getRestConfig(ndLst);

		} catch (Exception e) {
			throw new InvalidConfigException("Config failed with: "
					+ e.getMessage());
		}

	}

	private void getRestConfig(NodeList ndLst) {

		for (int i = 0; i < ndLst.getLength(); i++) {
			Node nde = ndLst.item(i);
			if (nde.getNodeType() == Node.ELEMENT_NODE) {
				String ndName = nde.getNodeName();
				switch (ndName) {
				case "imagedbcache":
					this.imageCacheSize = Integer.parseInt(this.getAttrib(nde,
							"max"));
					break;
				}
			}
		}
	}

	private DBPoolDataSource getPoolSourc(NodeList ndLst) {
		DBPoolDataSource dbPoolSourc = null;
		dbPoolSourc = new DBPoolDataSource();
		for (int i = 0; i < ndLst.getLength(); i++) {
			Node nde = ndLst.item(i);
			if (nde.getNodeType() == Node.ELEMENT_NODE) {
				String ndName = nde.getNodeName();
				switch (ndName) {
				case "poolname":
					dbPoolSourc.setName(this.getAttrib(nde, "value"));
					break;
				case "description":
					dbPoolSourc.setDescription(this.getAttrib(nde, "value"));
					break;
				case "ideltimeout":
					String idlStr = this.getAttrib(nde, "value");
					dbPoolSourc.setIdleTimeout(Integer.parseInt(idlStr));
					break;
				case "minpool":
					String minStr = this.getAttrib(nde, "value");
					dbPoolSourc.setMinPool(Integer.parseInt(minStr));
					break;
				case "maxpool":
					String maxStr = this.getAttrib(nde, "value");
					dbPoolSourc.setMaxPool(Integer.parseInt(maxStr));
					break;
				case "maxsize":
					String maxszStr = this.getAttrib(nde, "value");
					dbPoolSourc.setMaxSize(Integer.parseInt(maxszStr));
					break;
				case "username":
					dbPoolSourc.setUser(this.getAttrib(nde, "value"));
					break;
				case "password":
					dbPoolSourc.setPassword(this.getAttrib(nde, "value"));
					break;
				case "validationquery":
					dbPoolSourc
							.setValidationQuery(this.getAttrib(nde, "value"));
					break;
				case "driverclass":
					dbPoolSourc
							.setDriverClassName(this.getAttrib(nde, "value"));
					break;
				case "url":
					dbPoolSourc.setUrl(this.getAttrib(nde, "value"));
					break;
				default:
					System.out.print("Unknown Element: ");
					System.out.println(ndName);
				}
			}
		}
		return dbPoolSourc;
	}

	private String getAttrib(Node prntNde, String attName) {
		StringBuffer buf = new StringBuffer("");
		boolean isSet = false;
		if (prntNde.hasAttributes()) {
			NamedNodeMap ndeMp = prntNde.getAttributes();
			for (int i = 0; i < ndeMp.getLength(); i++) {
				Node nde = ndeMp.item(i);
				if (nde.getNodeName().compareTo(attName) == 0) {
					buf.append(nde.getNodeValue());
					isSet = true;
					break;
				}
			}
		}

		if (!isSet) {
			return "";
		} else {
			return buf.toString();
		}
	}
}
