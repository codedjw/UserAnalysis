package sjtu.ist.similarity;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import cern.colt.function.DoubleDoubleFunction;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;

public class SimilarityFactory {
	
	Map<SimilarityCalculator, Double> simCals = new HashMap<SimilarityCalculator, Double>();
	
	public SimilarityFactory() {
		simCals.put(new SeqSimCalculator(), 0.5);
		simCals.put(new FreqSimCalculator(), 0.5);
	}
	
	public DoubleMatrix2D doSimCalculation(List<String> sequences) {
		System.out.println("---------- Calculate Similarity Matrix Begin ----------");
		DoubleMatrix2D matrix = null;
		if (sequences != null && !sequences.isEmpty()) {
			int seqLength = sequences.size();
			matrix = new DenseDoubleMatrix2D(seqLength, seqLength);
			for (SimilarityCalculator sc : simCals.keySet()) {
				double prop = simCals.get(sc);
				DoubleMatrix2D calMatrix = sc.calculateSimilarity(sequences);
				matrix.assign(calMatrix, cern.jet.math.PlusMult.plusMult(prop));
			}
		}
		if (matrix != null) {
			System.out.println(matrix);
		}
		System.out.println("********** Calculate Similarity Matrix End **********");
		return matrix;
	}

}
