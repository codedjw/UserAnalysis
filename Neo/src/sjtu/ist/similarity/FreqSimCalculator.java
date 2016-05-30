package sjtu.ist.similarity;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import neobio.alignment.NeedlemanWunsch;
import neobio.alignment.PairwiseAlignmentAlgorithm;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;

/**
 * <p>Description: 计算序列间出现频率相似度 </p>
 * @author dujiawei
 * @version 1.0
 */

public class FreqSimCalculator implements SimilarityCalculator {
	public DoubleMatrix2D calculateSimilarity(List<String> sequences) {
		System.out.println("---------- Calculate Frequency Similarity Begin ----------");
		DoubleMatrix2D matrix = null;
		
		// 计算序列结构频率相似性
		if (sequences != null && !sequences.isEmpty()) {
			// 初始化
			int seqLength = sequences.size();
			matrix = new DenseDoubleMatrix2D(seqLength, seqLength);
			PairwiseAlignmentAlgorithm algorithm = new NeedlemanWunsch();
			
			for (int i=0; i<seqLength; i++) {
				matrix.setQuick(i, i, 1);
				for (int j=i; j<seqLength; j++) {
					String left = sequences.get(i);
					String right = sequences.get(j);
					// 计算两个序列（left, right）的相似度（FreqSim[i][j]） Sim[i][j] = 1
					// 输入两个序列
					// left union right
					Set<Character> units = new HashSet<Character>();
					for (int k=0; k<left.length(); k++) {
						units.add(left.charAt(k));
					}
					for (int k=0; k<right.length(); k++) {
						units.add(right.charAt(k));
					}
					int union = units.size();
					// left intersect right
					int intersect = 0;
					for (Character c : units) {
						if (left.indexOf(c) != -1 && right.indexOf(c) != -1) {
							intersect ++;
						}
					}
					System.out.println(left+" vs. "+right+": "+union+" union, "+intersect+" intersect.");
					double similarity = (intersect == 0 || union == 0) ? (double)0 : (double)intersect/(double)union;
					matrix.setQuick(i, j, similarity);
					matrix.setQuick(j, i, similarity);
				}
			}
		}
		
		if (matrix != null) {
			System.out.println(matrix);
		}
		System.out.println("********** Calculate Frequency Similarity End **********");
		return matrix;
	}
}
