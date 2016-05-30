package neobio.gui;

import neobio.alignment.BasicScoringScheme;
import neobio.alignment.IncompatibleScoringSchemeException;
import neobio.alignment.InvalidSequenceException;
import neobio.alignment.NeedlemanWunsch;
import neobio.alignment.PairwiseAlignment;
import neobio.alignment.PairwiseAlignmentAlgorithm;


public class Test {
	

	public static void main(String[] args) throws InvalidSequenceException, IncompatibleScoringSchemeException {

		//我只重写了NeedlemanWunsch方法，其他的什么SmithWaterman、Smawk等等不支持直接的字符序列输入。
		PairwiseAlignmentAlgorithm algorithm = new NeedlemanWunsch();
		
		String s1="TEMSWCRAQMFNGKDTINYQGAQYYVTFLWAWGWQTSIKQANYMYNEVSVVQSVTFPYNA EKMENQVGRDMFYCYCEDWTPFPHWAPICRKQCEVMWVMC";
		
		String s2="TEMSWCRAQMFNGKDFDTINYQGAQYYVTFLWAWGWQTSIKQANYMYNEVSVVSDQSVTFPYNA EKMENQVGRDMFYCYCEDWTPFPHWAPICRKQCEVDFMWVMC";
		
		//输入两个序列
		algorithm.loadSequences(s1,s2);

		//设置三个参数，我也不知道是什么参数，就用默认值了
		algorithm.setScoringScheme(new BasicScoringScheme (1, -1, -1));
		
		//计算两个序列
		PairwiseAlignment alignment = algorithm.getPairwiseAlignment();
		

		
		//显示两个序列的分数
		System.out.println(alignment.getScore());
		
		//显示2个序列的对齐
		System.out.println(alignment.toString());
			
		//按比例归一化到0~1.0
		Integer longer=0;
		
		if (s1.length()>s2.length()) {
			longer=s1.length();
		}
		else {
			longer=s2.length();
		}
		System.out.println("Similarity:"+ (double)alignment.getScore()/(double)longer);
	}

}
