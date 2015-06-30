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
import edu.gsu.dmlab.exceptions.InvalidConfigException;
import edu.gsu.dmlab.imageproc.interfaces.IHistogramProducer;

public class HistoCompML {

	DataSource imageDBPoolSourc;
	IImageDBConnection imageDBConnect;
	DataSource trackingDBPoolSourc;
	IHistogramProducer histProd;

	int imageCacheSize;

	int[] wavelengths = { 94, 131, 171, 193, 211, 304, 335, 1600, 1700 };

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

	public void run() {
		String sourceFile = "F:\\files\\exp3\\";
		String destFolder = "F:\\files\\tmp1";
		// AR
		{
			 //int[][] dims = { { 1, 6 }, { 3, 1 }, { 4, 1 }, { 7, 9 } };//Best
			// 1//AR
			int[][] dims = { { 1, 7 }, { 4, 6 }, { 6, 6 }, { 8, 5 } };// 1//AR//Mid
			String type = "AR";
			int histMeasure = 1;
			MLRunner runner = new MLRunner(this.histProd, type, sourceFile,
					dims, histMeasure);
			for (int modelType = 1; modelType < 4; modelType++) {
				runner.run(modelType, destFolder);
			}
		}
		// CH
		{
			// int[][] dims = { { 1, 10 }, { 7, 7 }, { 9, 1 } };// 1//CH//Best
			int[][] dims = { { 1, 6 }, { 3, 5 }, { 7, 2 }, { 8, 4 } };// 3//CH//Mid
			String type = "CH";
			int histMeasure = 3;
			MLRunner runner = new MLRunner(this.histProd, type, sourceFile,
					dims, histMeasure);
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
