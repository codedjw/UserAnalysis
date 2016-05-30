package sjtu.ist.similarity;

import java.util.List;

import cern.colt.matrix.DoubleMatrix2D;

public interface SimilarityCalculator {

	public DoubleMatrix2D calculateSimilarity(List<String> sequences);
	
}
