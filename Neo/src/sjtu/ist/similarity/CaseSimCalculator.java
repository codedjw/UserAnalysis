package sjtu.ist.similarity;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import neobio.alignment.BasicScoringScheme;
import neobio.alignment.IncompatibleScoringSchemeException;
import neobio.alignment.InvalidSequenceException;
import neobio.alignment.NeedlemanWunsch;
import neobio.alignment.PairwiseAlignment;
import neobio.alignment.PairwiseAlignmentAlgorithm;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import db2xes.util.Case;
import db2xes.util.Event;

public class CaseSimCalculator {
	
	public static DoubleMatrix2D calculateSimilarity(List<Case> cases) {
		System.out.println("---------- Calculate Case Similarity Begin ----------");
		DoubleMatrix2D matrix = null;
		if (cases != null && !cases.isEmpty()) {
			// 初始化
			int caseLength = cases.size();
			matrix = new DenseDoubleMatrix2D(caseLength, caseLength);
			PairwiseAlignmentAlgorithm algorithm = new NeedlemanWunsch();
			
			for (int i=0; i<caseLength; i++) {
				for (int j=i; j<caseLength; j++) {
					Case left_case = cases.get(i);
					Case right_case = cases.get(j);
					String left = "";
					String right = "";
					Map<String, String> event_alphabet = new HashMap<String, String>();
					char c = 'A';
					int index = 0;
					for (Event e : left_case.getEvents()) {
						String activity = e.getActivity();
						if (!event_alphabet.containsKey(activity)) {
							event_alphabet.put(activity, ""+(char)(c+index));
							index++;
						}	
					}
					
					for (Event e : right_case.getEvents()) {
						String activity = e.getActivity();
						if (!event_alphabet.containsKey(activity)) {
							event_alphabet.put(activity, ""+(char)(c+index));
							index++;
						}	
					}
					for (Event e : left_case.getEvents()) {
						String activity = e.getActivity();
						left += event_alphabet.get(activity);	
					}
					
					for (Event e : right_case.getEvents()) {
						String activity = e.getActivity();
						right += event_alphabet.get(activity);	
					}
//					System.out.println(left+" vs. "+right);
//					System.out.println(event_alphabet.size());
					if (event_alphabet.size() <= 26) {
						// 计算两个序列（left, right）的相似度（CaseSim[i][j]）CaseSim[i][j] = 1
						// 输入两个序列
						try {
							algorithm.loadSequences(left, right);
							//设置三个参数，我也不知道是什么参数，就用默认值了
//							algorithm.setScoringScheme(new BasicScoringScheme (1, -1, -1));
							algorithm.setScoringScheme(new BasicScoringScheme (1, -1, 0));
							//计算两个序列
							PairwiseAlignment alignment = algorithm.getPairwiseAlignment();
							//显示Sample CaseID
							System.out.println(left_case.getCase_id()+" vs. "+right_case.getCase_id());
							//显示两个序列的分数
//							System.out.println(alignment.getScore());
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
					} else {
						System.out.println(left_case.getCase_id()+" vs. "+right_case.getCase_id());
						System.out.println(left+" vs. "+right);
						System.out.println("too long: "+event_alphabet.size());
					}
				}
			}
		}
		System.out.println("********** Calculate Case Similarity End **********");
		return matrix;
	}
	
}
