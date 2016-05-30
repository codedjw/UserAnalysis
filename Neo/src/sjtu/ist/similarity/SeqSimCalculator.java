package sjtu.ist.similarity;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

import neobio.alignment.BasicScoringScheme;
import neobio.alignment.IncompatibleScoringSchemeException;
import neobio.alignment.InvalidSequenceException;
import neobio.alignment.NeedlemanWunsch;
import neobio.alignment.PairwiseAlignment;
import neobio.alignment.PairwiseAlignmentAlgorithm;
import cern.colt.function.DoubleDoubleFunction;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;

/***
 * <p>Description: 计算序列间结构相似度 </p>
 * @author dujiawei
 * @version 1.0
 */

public class SeqSimCalculator implements SimilarityCalculator {
	public DoubleMatrix2D calculateSimilarity(List<String> sequences) {
		System.out.println("---------- Calculate Sequence Similarity Begin ----------");
		DoubleMatrix2D matrix = null;
		
		// 计算序列结构相似性
		if (sequences != null && !sequences.isEmpty()) {
			// 初始化
			int seqLength = sequences.size();
			matrix = new DenseDoubleMatrix2D(seqLength, seqLength);
			PairwiseAlignmentAlgorithm algorithm = new NeedlemanWunsch();
			
			for (int i=0; i<seqLength; i++) {
				for (int j=i; j<seqLength; j++) {
					String left = sequences.get(i);
					String right = sequences.get(j);
					// 计算两个序列（left, right）的相似度（SeqSim[i][j]） Sim[i][j] = 1
					// 输入两个序列
					try {
						algorithm.loadSequences(left, right);
						//设置三个参数，我也不知道是什么参数，就用默认值了
						algorithm.setScoringScheme(new BasicScoringScheme (1, -1, -1));
						//计算两个序列
						PairwiseAlignment alignment = algorithm.getPairwiseAlignment();
						//显示两个序列的分数
//						System.out.println(alignment.getScore());
						//显示2个序列的对齐
						System.out.println(alignment.toString());
						//按比例归一化到0~1.0
						Integer longer = (left.length() > right.length()) ? left.length() : right.length();
						double similarity = (double)alignment.getScore()/(double)longer;
						matrix.setQuick(i, j, similarity);
						matrix.setQuick(j, i, similarity);
					} catch (InvalidSequenceException | IncompatibleScoringSchemeException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
			}
			// Sum(A,B)
//			DoubleDoubleFunction plus = new DoubleDoubleFunction() {
//			    public double apply(double a, double b) { return a+b; }
//			}; 
//			matrix.assign(Algebra.DEFAULT.transpose(matrix), plus);
		}
		
		if (matrix != null) {
			System.out.println(matrix);
		}
		System.out.println("********** Calculate Sequence Similarity End **********");
		return matrix;
	}
	
}
