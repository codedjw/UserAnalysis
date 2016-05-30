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

package neobio.gui;

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;
import java.applet.*;
import java.util.*;
import java.net.URL;
import java.beans.PropertyVetoException;

/**
 * This class is a simple GUI utility for computing pairwise sequence alignments using one
 * of the the algorithms provided in the {@link neobio.alignment} package.
 *
 * @author Sergio A. de Carvalho Jr.
 */
public class NeoBio extends JFrame
{
    private JMenu			file_menu, help_menu;
    private JMenuBar		menu_bar;
    private JMenuItem		new_alignment_item, exit_menuitem, about_menuitem;
    private JSeparator		mid_separator;
    private JToolBar		file_toolbar;
    private JPanel 			toolbar_panel;
    private JButton			alignment_button;

    private JDesktopPane	desktop_pane;

	/**
	 * Creates a new instance of a graphical interface.
	 */
    public NeoBio()
    {
		super();

        setTitle("NeoBio");
        setDefaultCloseOperation (WindowConstants.DISPOSE_ON_CLOSE);
        initComponents();

        show();

		// always open pairwise alignment internal frame
        pairwiseAlignment();
    }

    private void initComponents()
    {
		URL icon;

		// window closing event
        addWindowListener(new WindowAdapter()
        {
            public void windowClosing(WindowEvent e)
            {
                exitForm();
            }
        });

		Container content_pane = getContentPane();

		desktop_pane = new JDesktopPane();

		content_pane.add (desktop_pane, BorderLayout.CENTER);

        new_alignment_item = new JMenuItem("Pairwise Alignment");
        new_alignment_item.setMnemonic('p');
        new_alignment_item.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                newAlignmentActionPerformed(e);
            }
        });
        icon = getClass().getResource("icons/alignment.gif");
        if (icon != null) new_alignment_item.setIcon(new ImageIcon(icon));

        mid_separator = new JSeparator();

        exit_menuitem = new JMenuItem("Exit");
        exit_menuitem.setMnemonic('x');
        exit_menuitem.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                exitMenuItemActionPerformed(e);
            }
        });

        file_menu = new JMenu("File");
        file_menu.setMnemonic('f');
        file_menu.add(new_alignment_item);
        file_menu.add(mid_separator);
        file_menu.add(exit_menuitem);

        about_menuitem = new JMenuItem("About");
        about_menuitem.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                aboutMenuItemActionPerformed(e);
            }
        });
        icon = getClass().getResource("icons/help.gif");
        if (icon != null) about_menuitem.setIcon(new ImageIcon(icon));

        help_menu = new JMenu("Help");
        help_menu.add(about_menuitem);

        menu_bar = new JMenuBar();
        //menu_bar.setFont(getFont());
        menu_bar.add(file_menu);
        menu_bar.add(help_menu);

        setJMenuBar(menu_bar);

        alignment_button = new JButton();
        alignment_button.setMnemonic('p');
        alignment_button.setToolTipText("Pairwise Alignment...");
        alignment_button.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                newAlignmentActionPerformed(e);
            }
        });
        icon = getClass().getResource("icons/alignment.gif");
        if (icon != null) alignment_button.setIcon(new ImageIcon(icon));

        file_toolbar = new JToolBar();
        file_toolbar.setRollover(true);
        file_toolbar.add(alignment_button);

        toolbar_panel = new JPanel();
		toolbar_panel.setLayout(new FlowLayout(FlowLayout.LEFT, 0, 0));
        toolbar_panel.setBorder(new EtchedBorder());
        toolbar_panel.add(file_toolbar);

        content_pane.add(toolbar_panel, BorderLayout.NORTH);

		// set frame size
        Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
        setSize((screen.width * 2) / 3, (screen.height * 7) / 8);
		setLocation(screen.width / 6, screen.height / 16);
    }

    private void aboutMenuItemActionPerformed (ActionEvent e)
    {
        (new AboutDialog(this)).show();
    }

    private void exitMenuItemActionPerformed (ActionEvent e)
    {
        exitForm();
    }

    private void exitForm ()
    {
        System.exit(0);
    }

    private void newAlignmentActionPerformed (ActionEvent e)
    {
		pairwiseAlignment();
    }

    private void pairwiseAlignment ()
    {
	    PairwiseAlignmentFrame alignment_frame = new PairwiseAlignmentFrame (this);

        desktop_pane.add (alignment_frame);

        alignment_frame.setBounds(0, 0, 500, 500);
        alignment_frame.show();
        alignment_frame.toFront();

		try
		{
			alignment_frame.setMaximum (true);
		}
		catch (PropertyVetoException e) {}
	}

	/**
	 * Create and run a new interface.  The main method takes no parameter from the
	 * command line.
	 *
	 * @param args command line arguments
	 */
    public static void main(String args[])
    {
        new NeoBio();
    }
}
