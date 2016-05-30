/*
 * About.java
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

package neobio.gui;

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import java.net.URL;

/**
 * About screen.
 *
 * @author Sergio A. de Carvalho Jr.
 */
public class AboutDialog extends JDialog
{
	private JLabel image_label;

	/**
	 * Creates a new instance of the About screen.
	 *
	 * @param parent the parent frame
	 */
    public AboutDialog (Frame parent)
    {
        super (parent, true);
        initComponents ();
        pack ();
    }

    private void initComponents ()
    {
		URL	image_filename;

        setTitle ("About");
        setResizable (false);
        setDefaultCloseOperation (WindowConstants.DISPOSE_ON_CLOSE);

        addWindowListener (new WindowAdapter ()
        {
            public void windowClosing (WindowEvent e)
            {
                closeDialog (e);
            }
        });

		image_filename = getClass().getResource("icons/about.jpg");
		if (image_filename != null)
		{
			image_label = new JLabel ();
        	image_label.setIcon(new ImageIcon(image_filename));
        	getContentPane().add(image_label, BorderLayout.CENTER);
		}
    }

    private void closeDialog(WindowEvent e)
    {
        setVisible (false);
        dispose ();
    }
}