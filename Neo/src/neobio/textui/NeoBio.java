/*
 * NeoBio.java
 *
 * Copyright 2003 Sergio Anibal de Carvalho Junior
 *
 * This file is part of NeoBio.
 *
 * NeoBio is free software; you can redistribute it and/or modify it under the terms of
 * the GNU General Public License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * NeoBio is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with NeoBio;
 * if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 * Proper attribution of the author as the source of the software would be appreciated.
 *
 * Sergio Anibal de Carvalho Junior		mailto:sergioanibaljr@users.sourceforge.net
 * Department of Computer Science		http://www.dcs.kcl.ac.uk
 * King's College London, UK			http://www.kcl.ac.uk
 *
 * Please visit http://neobio.sourceforge.net
 *
 * This project was supervised by Professor Maxime Crochemore.
 *
 */

package neobio.textui;

import neobio.alignment.*;
import java.io.FileReader;
import java.io.IOException;

/**
 * This class is a simple command line based utility for computing pairwise sequence
 * alignments using one of the the algorithms provided in the {@link neobio.alignment}
 * package.
 *
 * <P>The main method takes the follwing parameters from the command line:
 *
 * <CODE><BLOCKQUOTE>
 * NeoBio &lt;alg&gt; &lt;S1&gt; &lt;S2&gt; [M &lt;matrix&gt; | S &lt;match&gt;
 * &lt;mismatch&gt; &lt;gap&gt;]
 * </BLOCKQUOTE></CODE>
 *
 * <UL>
 * <LI><B><CODE>&lt;alg&gt;</CODE></B> is either <B><CODE>NW</CODE></B> for {@linkplain
 * neobio.alignment.NeedlemanWunsch Needleman & Wunsch} (global alignment),
 * <B><CODE>SW</CODE></B> for {@linkplain neobio.alignment.SmithWaterman Smith & Waterman}
 * (local alignment), <B><CODE>CLZG</CODE></B> for {@linkplain
 * neobio.alignment.CrochemoreLandauZivUkelsonGlobalAlignment Crochemore, Landau &
 * Ziv-Ukelson global alignment} or <B><CODE>CLZL</CODE></B> for {@linkplain
 * neobio.alignment.CrochemoreLandauZivUkelsonLocalAlignment Crochemore, Landau &
 * Ziv-Ukelson local alignment};
 *
 * <LI><B><CODE>&lt;S1&gt;</CODE></B> is the first sequence file;
 *
 * <LI><B><CODE>&lt;S2&gt;</CODE></B> is the second sequence file;
 *
 * <LI><B><CODE>M &lt;matrix&gt;</CODE></B> is for using a scoring matrix file;
 *
 * <LI><B><CODE>S &lt;match&gt; &lt;mismatch&gt; &lt;gap&gt;</CODE></B> is for using a
 * simple scoring scheme, where <B><CODE>&lt;match&gt;</CODE></B> is the match reward
 * value, <B><CODE>&lt;mismatch&gt;</CODE></B> is the mismatch penalty value and
 * <B><CODE>&lt;gap&gt;</CODE></B> is the cost of a gap (linear gap cost function).
 * </UL>
 *
 * @author Sergio A. de Carvalho Jr.
 */
public class NeoBio
{
	/**
	 * The main method takes parameters from the command line to compute a pairwise
	 * sequence alignment. See the class description for details.
	 *
	 * @param args command line arguments
	 */
	public static void main (String args[])
	{
		PairwiseAlignmentAlgorithm	algorithm;
		FileReader					seq1, seq2;
		ScoringScheme				scoring;
		PairwiseAlignment			alignment;
		String						algo, file1, file2, scoring_type;
		long						start, elapsed;
		int							match, mismatch, gap;

		try
		{
			// create an instance of the
			// requested algorithm
			algo = args[0];

			if (algo.equalsIgnoreCase("nw"))
				algorithm = new NeedlemanWunsch();
			else if (algo.equalsIgnoreCase("sw"))
				algorithm = new SmithWaterman();
			else if (algo.equalsIgnoreCase("clzg"))
				algorithm = new CrochemoreLandauZivUkelsonGlobalAlignment();
			else if (algo.equalsIgnoreCase("clzl"))
				algorithm = new CrochemoreLandauZivUkelsonLocalAlignment();
			else
			{
				usage();
				System.exit(1);
				return;
			}

			// sequences file names
			file1 = args[1];
			file2 = args[2];
		}
		catch (ArrayIndexOutOfBoundsException e)
		{
			usage();
			System.exit(1);
			return;
		}

		try
		{
			// scoring scheme type
			scoring_type = args[3];

			try
			{
				if (scoring_type.equalsIgnoreCase("M"))
				{
					// use scoring matrix
					scoring = new ScoringMatrix (new FileReader(args[4]));
				}
				else if (scoring_type.equalsIgnoreCase("S"))
				{
					// use basic scoring scheme
					match = Integer.parseInt(args[4]);
					mismatch = Integer.parseInt(args[5]);
					gap = Integer.parseInt(args[6]);

					scoring = new BasicScoringScheme (match, mismatch, gap);
				}
				else
				{
					usage();
					System.exit(1);
					return;
				}
			}
			catch (NumberFormatException e)
			{
				usage();
				System.exit(1);
				return;
			}
			catch (ArrayIndexOutOfBoundsException e)
			{
				usage();
				System.exit(1);
				return;
			}
			catch (InvalidScoringMatrixException e)
			{
				System.err.println(e.getMessage());
				System.exit(2);
				return;
			}
			catch (IOException e)
			{
				System.err.println(e.getMessage());
				System.exit(2);
				return;
			}
		}
		catch (ArrayIndexOutOfBoundsException e)
		{
			// not specified: use default scoring scheme
			scoring = new BasicScoringScheme (1, -1, -1);
		}

		// set scoring scheme
		algorithm.setScoringScheme(scoring);

		try
		{
			// load sequences
			System.err.println("\nLoading sequences...");

			seq1 = new FileReader(file1);
			seq2 = new FileReader(file2);

			start = System.currentTimeMillis();
			algorithm.loadSequences(seq1, seq2);
			elapsed = System.currentTimeMillis() - start;

			// close files
			seq1.close();
			seq2.close();

			System.err.println("[ Elapsed time: " + elapsed + " milliseconds ]\n");

			/*
			// compute score only
			System.err.println("\nComputing score...");

			start = System.currentTimeMillis();
			score = algorithm.getScore();
			elapsed = System.currentTimeMillis() - start;

			System.out.println("Score: " + score);
			System.err.println("[ Elapsed time: " + elapsed + " milliseconds ]");
			//*/

			// compute alignment
			System.err.println("Computing alignment...");

			start = System.currentTimeMillis();
			alignment = algorithm.getPairwiseAlignment();
			elapsed = System.currentTimeMillis() - start;

			System.err.println("[ Elapsed time: " + elapsed + " milliseconds ]\n");

			System.out.println("Alignment:\n" + alignment);
		}
		catch (InvalidSequenceException e)
		{
			System.err.println("Invalid sequence file.");
			System.exit(2);
			return;
		}
		catch (IncompatibleScoringSchemeException e)
		{
			System.err.println("Incompatible scoring scheme.");
			System.exit(2);
			return;
		}
		catch (IOException e)
		{
			System.err.println(e.getMessage());
			System.exit(2);
			return;
		}

		// print scoring scheme
		//System.out.println(scoring);

		System.exit(0);
	}

	/**
	 * Prints command line usage.
	 */
	public static void usage ()
	{
		System.err.println(
		"\nUsage: NeoBio <alg> <S1> <S2> [M <matrix> | S <match> <mismatch> <gap>]\n\n" +
		"where:\n\n" +
		"   <alg> = NW   for Needleman & Wunsch (global alignment)\n" +
		"        or SW   for Smith & Waterman (local alignment)\n" +
		"        or CLZG for Crochemore, Landau & Ziv-Ukelson global alignment\n" +
		"        or CLZL for Crochemore, Landau & Ziv-Ukelson local alignment\n\n" +
		"   <S1> = first sequence file\n\n" +
		"   <S2> = second sequence file\n\n" +
		"   M <matrix> for using a scoring matrix file\n\n" +
		"or\n\n" +
		"   S <match> <mismatch> <gap> for using a simple scoring scheme\n" +
		"     where <match> = match reward value\n" +
		"           <mismatch> = mismatch penalty value\n" +
		"           <gap> = cost of a gap (linear gap cost function)"
		);
	}
}
