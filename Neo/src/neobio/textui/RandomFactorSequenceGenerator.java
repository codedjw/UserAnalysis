/*
 * RandomFactorSequenceGenerator.java
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

import java.io.BufferedWriter;
import java.io.Writer;
import java.io.FileWriter;
import java.io.OutputStreamWriter;
import java.io.IOException;

/**
 * This class is a simple command line based utility for generating random sequences with
 * optimal LZ78 factorisation.
 *
 * <P>The main method takes three parameters from the command line to generate a
 * sequence: <CODE>type</CODE>, <CODE>size</CODE> and <CODE>file</CODE>, where:
 * <UL>
 * <LI><B><CODE>type</CODE></B> is either <CODE>DNA</CODE> for DNA sequences or
 * <CODE>PROT</CODE> for protein sequences.
 * <LI><B><CODE>size</CODE></B> is the number os characters.
 * <LI><B><CODE>file</CODE></B> (optional) is the name of a file (if ommited, sequence
 * is written to standard output).
 * </UL>
 * </P>
 *
 * @author Sergio A. de Carvalho Jr.
 */
public class RandomFactorSequenceGenerator
{
	/**
	 * Character set for DNA sequences.
	 */
	private static final char[] DNA_CHARS = {'A', 'C', 'G', 'T'};

	/**
	 * Character set for protein sequences.
	 */
	private static final char[] PROT_CHARS = {'A','R','N','D','C','Q','E','G','H','I',
								'L','K','M','F','P','S','T','W','Y','V','B','Z','X'};

	/**
	 * The main method takes three parameters from the command line to generate a
	 * sequence. See the class description for details.
	 *
	 * @param args command line arguments
	 */
	public static void main (String[] args)
	{
		Writer		output;
		String		seq_type, filename;
		int			size, random;
		char[]		charset;
		int[]		qty;
		int[]		factor;

		try
		{
			// get 1st argument (required): file type
			seq_type = args[0];

			// get 2nd argument (required): number of characters
			size = Integer.parseInt(args[1]);
		}
		catch (ArrayIndexOutOfBoundsException e)
		{
			usage();
			System.exit(1);
			return;
		}
		catch (NumberFormatException e)
		{
			usage();
			System.exit(1);
			return;
		}

		// validate character set
		if (seq_type.equalsIgnoreCase("DNA"))
			charset = DNA_CHARS;
		else if (seq_type.equalsIgnoreCase("PROT"))
			charset = PROT_CHARS;
		else
		{
			// no such option
			usage();
			System.exit(1);
			return;
		}

		// validate size
		if (size <= 3)
		{
			System.err.println ("Error: size must be greater than 3.");
			System.exit(1);
			return;
		}

		try
		{
			// get 3rd argument (optional): file name
			filename = args[2];

			try
			{
				// open file for writing
				output = new BufferedWriter (new FileWriter (filename));
			}
			catch (IOException e)
			{
				System.err.println ("Error: couldn't open " + filename + " for writing.");
				e.printStackTrace();
				System.exit(2);
				return;
			}
		}
		catch (ArrayIndexOutOfBoundsException e)
		{
			// file name was ommited, use standard output
			filename = null;
			output = new OutputStreamWriter (System.out);
		}

		// alocate an of characters statistics
		qty = new int[charset.length];

		// alocate an array to store the growing factor
		// its size will be no greather than half sequence size
		// (in fact, it's much less than that!)
		factor = new int [size / 2];

		try
		{
			int s = 0, i, f_size = 0;

			// write sequence
			while (s < size)
			{
				// copy previous factor
				for (i = 0; i < f_size && s < size; i++)
				{
					output.write(charset[factor[i]]);

					s++;

					// keep track of how many characters
					// have been writen of each type
					qty[factor[i]]++;
				}

				if (s < size)
				{

					// choose a character index randomly
					random = (int) (Math.random() * charset.length);

					// extend factor with the random char index
					factor[f_size++] = random;

					// keep track of how many characters
					// have been writen of each type
					qty[random]++;

					output.write(charset[random]);

					s++;
				}
			}

			output.flush();

			if (filename != null) output.close();
		}
		catch (IOException e)
		{
			System.err.println ("Error: failed to write sequence.");
			e.printStackTrace();
			System.exit(2);
			return;
		}

		// print character distribution
		System.out.println("\nCharacter distribution:");
		for (int i = 0; i < charset.length; i++)
			System.err.println(charset[i] + ": " + qty[i]);

		System.exit(0);
	}

	/**
	 * Prints command line usage.
	 */
	private static void usage ()
	{
		System.err.println(
		"\nUsage: RandomFactorSequenceGenerator <type> <size> [<file>]\n\n" +
		"where:\n\n" +
		"   <type> = DNA for nucleotide sequences\n" +
		"         or PROT for protein sequences\n\n" +
		"   <size> = number os characters\n\n" +
		"   <file> = name of a file to where the sequence is to be written\n" +
		"            (if ommited, sequence is written to standard output)"
		);
	}
}
