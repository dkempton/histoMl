package edu.gsu.dmlab.ml;

import java.io.File;
import java.util.ArrayList;

import edu.gsu.dmlab.ObjectFactory;
import edu.gsu.dmlab.datatypes.interfaces.ITrack;

public class TrackReader {

	static int span = 12600;

	static public ITrack[][] getTracks(String sourceFolder, String type) {
		File[] files = getFiles(new File(sourceFolder), type);
		ArrayList<ITrack[]> trackList = new ArrayList<ITrack[]>();
		for (int i = 0; i < files.length; i++) {
			ITrack[] tracks = ObjectFactory.getTrackedResults(
					type.toUpperCase(), files[i].getParent(), span);
			trackList.add(tracks);

		}

		ITrack[][] retArr = new ITrack[trackList.size()][];
		trackList.toArray(retArr);
		
		//ITrack[][] retArr = new ITrack[3][];
		//retArr[0] = trackList.get(0);
		//retArr[1] = trackList.get(1);
		//retArr[2] = trackList.get(2);
		
		return retArr;
	}

	static private File[] getFiles(File dir, String type) {
		ArrayList<File> files = new ArrayList<File>();

		if (dir.isDirectory()) {
			String[] children = dir.list();
			for (int i = 0; children != null && i < children.length; i++) {
				File[] tmpFiles = getFiles(new File(dir, children[i]), type);
				for (int j = 0; j < tmpFiles.length; j++) {
					files.add(tmpFiles[j]);
				}
			}
		}
		if (dir.isFile()) {
			if (dir.getName()
					.endsWith(type.toUpperCase() + "DustinTracked.txt")) {
				files.add(dir);
			}
		}

		File[] returnFiles = new File[files.size()];
		files.toArray(returnFiles);
		return returnFiles;
	}
}
