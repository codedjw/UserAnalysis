package sjtu.ist.main;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

import sjtu.ist.similarity.SimilarityFactory;
import cern.colt.matrix.DoubleMatrix2D;

public class UserCluster {

	public static void main(String[] args) {
		List<String> sequences = new ArrayList<String>() {
			{
				add("AFD");
				add("ABCD");
			}
		};
		String seqLogName = "simlarityMatrix.log";
		PrintStream sysout = System.out; // always console output
		// --- BEGIN --- (redirect log output to file)
		PrintStream printStream = null;
		File logfile = new File(seqLogName);
		try {
			logfile.createNewFile();
			FileOutputStream fileOutputStream = new FileOutputStream(logfile);
			printStream = new PrintStream(fileOutputStream);
			System.setOut(printStream);
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		// --- END ---
		SimilarityFactory sf = new SimilarityFactory();
		DoubleMatrix2D matrix = sf.doSimCalculation(sequences);
		
		System.setOut(sysout);
		if (matrix != null) {
			System.out.println(matrix);
		}
	}

}
