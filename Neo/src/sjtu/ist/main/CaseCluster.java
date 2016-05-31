package sjtu.ist.main;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import cern.colt.matrix.DoubleMatrix2D;
import sjtu.ist.similarity.CaseSimCalculator;
import sjtu.ist.util.SQLConnFactory;
import db2xes.util.Case;
import db2xes.util.Event;

public class CaseCluster {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
			Connection con = SQLConnFactory.getCon("qyw");
			Statement stmt = con.createStatement();
			String query = "SELECT CASE_ID, USER_ID AS USER_RESOURCE, VISIT_MEAN AS ACTIVITY, VISIT_TIME AS TIMESTAMP FROM qyw_7th_yy_succ_all_selected ORDER BY USER_ID, VISIT_TIME ASC;";
			long beginTime = System.currentTimeMillis();
			ResultSet rs = stmt.executeQuery(query);
			long finishTime = System.currentTimeMillis();
			System.out.println(query+" 运行时间： "+(finishTime-beginTime)/1000/60+"min, "+(finishTime-beginTime)%(1000*60)/1000+"s, "+(finishTime-beginTime)%(1000*60)%1000+"ms");
			
			String seqLogName = "simlarityMatrix.log";
			PrintStream sysout = System.out; // always console output
			// --- BEGIN --- (redirect log output to file)
			PrintStream printStream = null;
			File logfile = new File(seqLogName);
			try {
				logfile.createNewFile();
				FileOutputStream fileOutputStream = new FileOutputStream(logfile);
				printStream = new PrintStream(fileOutputStream);
				System.setOut(printStream);
			} catch (IOException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
			// --- END ---
			
			Map<String, List<Event>> case_events = new HashMap<String, List<Event>>();
			SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
			List<Case> sample_cases = new ArrayList<Case>();
			List<Case> real_cases = new ArrayList<Case>();
			while (rs.next()) {
				String activity_time = sdf.format(rs.getTimestamp("TIMESTAMP"));
				String activity = rs.getString("ACTIVITY");
				String user_id = rs.getString("USER_RESOURCE");
				String case_id = rs.getString("CASE_ID"); 
				List<Event> events = null;
				if (!case_events.containsKey(case_id)) {
					case_events.put(case_id, new ArrayList<Event>());
				}
				events = case_events.get(case_id);
				events.add(new Event(activity, activity_time, user_id, case_id));
				case_events.put(case_id, events);
			}
			int i = 0;
			for (String cid : case_events.keySet()) {
				List<Event> eves = case_events.get(cid);
				real_cases.add(new Case(cid, eves));
				Case ca = new Case(String.valueOf(i), eves);
				boolean found = false;
				for (Case c : sample_cases) {
					if (c.structureEquals(ca)) {
						found = true;
						break;
					}
				}
				if (!found) {
					sample_cases.add(ca);
					i++;
				}
			}
			
			for (Case c : sample_cases) {
				System.out.print(c.getCase_id()+"\t");
				for (int k=0; k<c.getEvents().size(); k++) {
					Event e = c.getEvents().get(k);
					System.out.print(e.getActivity());
					if (k != c.getEvents().size()-1) {
						System.out.print("->");
					}
				}
				System.out.println();
			}
			
			DoubleMatrix2D matrix = CaseSimCalculator.calculateSimilarity(sample_cases);
			File case_sim_file = new File("/Users/dujiawei/git/UserAnalysis/case_sim.txt");
			File case_file = new File("/Users/dujiawei/git/UserAnalysis/case.txt");
			if (!case_sim_file.exists()) {
				case_sim_file.createNewFile();
			}
			if (!case_file.exists()) {
				case_file.createNewFile();
			}
			BufferedWriter bw = new BufferedWriter(new FileWriter(case_sim_file, false));
			DecimalFormat df = new DecimalFormat("0.00");
			for (int j=0; j<matrix.rows(); j++) {
				String row = "";
				for (int k=0; k<matrix.columns(); k++) {
					row += df.format(matrix.get(j, k));
					if (k < matrix.columns()-1) {
						row += "\t";
					}
				}
				bw.write(row);
				if (j < matrix.rows()-1) {
					bw.newLine();
				}
			}
//			bw.write(matrix.toString());
			bw.close();
			
			bw = new BufferedWriter(new FileWriter(case_file, false));
			for (int j=0; j<sample_cases.size(); j++) {
				String row = "";
				Case c = sample_cases.get(j);
				String case_id = c.getCase_id();
				row += case_id;
				List<Event> es = c.getEvents();
				if (es != null && !es.isEmpty()) {
					row += "\t";
					for (int k=0; k<es.size(); k++) {
						row += es.get(k).getActivity();
						if (k < es.size()-1) {
							row += "\t";
						}
					}
				}
				bw.write(row);
				if (j < sample_cases.size()-1) {
					bw.newLine();
				}
			}
			bw.close();
			
			System.out.println(matrix);
			
			System.setOut(sysout);
			
			System.out.println("Case Similarity Calculation Over");
//			System.out.println("sample_cases: "+sample_cases.size());
//			System.out.println("real_cases: "+real_cases.size());
			con.close();
		} catch (InstantiationException | IllegalAccessException
				| ClassNotFoundException | SQLException | IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
