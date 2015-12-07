package edu.gsu.dmlab.ml;

public class Consts {

	public static final float SameLabel = (float) 1.0;
	public static final float DiffLabel = (float) -1.0;
	
	public static enum CompMethod{
		CORREL, CHISQR, INTERSECT, BHATTACHARYYA
	}
}
