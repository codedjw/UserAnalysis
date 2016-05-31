package sjtu.ist.main;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import sjtu.ist.similarity.CaseSimCalculator;
import sjtu.ist.similarity.SimilarityFactory;
import sjtu.ist.util.SQLConnFactory;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import db2xes.util.Case;
import db2xes.util.Event;

public class UserCluster {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
			Connection con = SQLConnFactory.getCon("qyw");
			Statement stmt = con.createStatement();
			String query = "SELECT CASE_ID, USER_ID AS USER_RESOURCE, VISIT_MEAN AS ACTIVITY, VISIT_TIME AS TIMESTAMP FROM qyw_7th_yy_succ_all_selected ORDER BY USER_ID, VISIT_TIME ASC;";
			long beginTime = System.currentTimeMillis();
			ResultSet rs = stmt.executeQuery(query);
			long finishTime = System.currentTimeMillis();
			System.out.println(query + " 运行时间： " + (finishTime - beginTime)
					/ 1000 / 60 + "min, " + (finishTime - beginTime)
					% (1000 * 60) / 1000 + "s, " + (finishTime - beginTime)
					% (1000 * 60) % 1000 + "ms");

			Map<String, List<Event>> case_events = new HashMap<String, List<Event>>();
			SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
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
			Map<String, List<Case>> user_cases = new HashMap<String, List<Case>>();
			for (String cid : case_events.keySet()) {
				List<Event> eves = case_events.get(cid);
				Case ca = new Case(cid, eves);
				if (eves != null && !eves.isEmpty()) {
					String user_id = eves.get(0).getResource();
					List<Case> cas = null;
					if (!user_cases.containsKey(user_id)) {
						user_cases.put(user_id, new ArrayList<Case>());
					}
					cas = user_cases.get(user_id);
					cas.add(ca);
					user_cases.put(user_id, cas);
				}
			}
			
			// 读入预处理后的sample case cluster （case.txt, case_cluster.txt，一一对应）
			FileReader reader = new FileReader("/Users/dujiawei/git/UserAnalysis/case.txt");
            BufferedReader br = new BufferedReader(reader);
            List<Case> sample_cases = new ArrayList<Case>();
            String str;
            while ((str = br.readLine()) != null) {
                String[] fields = str.split("\t");
                if (fields.length >= 2) {
                	String case_id = fields[0];
                	List<Event> eves = new ArrayList<Event>();
                    for (int i=1; i<fields.length; i++) {
                    	Event e = new Event(fields[i], null, null, case_id);
                    	eves.add(e);
                    }
                    Case c = new Case(case_id, eves);
                    sample_cases.add(c);
                }
            }
            br.close();
            reader.close();

            reader = new FileReader("/Users/dujiawei/git/UserAnalysis/case_cluster.txt");
            br = new BufferedReader(reader);
            Map<Integer, Integer> case_clusters = new HashMap<Integer, Integer>();
            int i = 0;
            int max_cluster = 0; // min_cluster == 1 (must)
            while ((str = br.readLine()) != null) {
            	int cluster = (int) (double) (Double.valueOf(str));
            	max_cluster = (max_cluster < cluster) ? cluster : max_cluster;
            	case_clusters.put(i, cluster);
            	i++;
            }
            br.close();
            reader.close();
            
            // 用户与Case维度映射
            DoubleMatrix2D matrix = new SparseDoubleMatrix2D(user_cases.keySet().size(), max_cluster);
            List<String> uids = new ArrayList<String>();
            int k = 0;
            for (String uid : user_cases.keySet()) {
            	uids.add(uid);
            	List<Case> cas = user_cases.get(uid);
            	for (Case c : cas) {
            		for (Case s : sample_cases) {
            			if (c.structureEquals(s)) {
            				int clid = Integer.valueOf(s.getCase_id());
            				matrix.setQuick(k, case_clusters.get(clid), matrix.getQuick(k, case_clusters.get(clid))+1);
            				break;
            			}
            		}
            	}
            	k++;
            }
            
            // 写入文件
            File user_dim_file = new File("/Users/dujiawei/git/UserAnalysis/user_dim.txt");
			File user_file = new File("/Users/dujiawei/git/UserAnalysis/user.txt");
			if (!user_dim_file.exists()) {
				user_dim_file.createNewFile();
			}
			if (!user_file.exists()) {
				user_file.createNewFile();
			}
			BufferedWriter bw = new BufferedWriter(new FileWriter(user_dim_file, false));
			for (int j=0; j<matrix.rows(); j++) {
				String row = "";
				for (int l=0; l<matrix.columns(); l++) {
					row += matrix.get(j, l);
					if (l < matrix.columns()-1) {
						row += "\t";
					}
				}
				bw.write(row);
				if (j < matrix.rows()-1) {
					bw.newLine();
				}
			}
			bw.close();		
//            System.out.println(matrix);
			
			bw = new BufferedWriter(new FileWriter(user_file, false));
			for (int j=0; j<uids.size(); j++) {
				String row = uids.get(j);
				bw.write(row);
				if (j < sample_cases.size()-1) {
					bw.newLine();
				}
			}
			bw.close();
            
			System.out.println("User Dimension Retrievation Over");
			// System.out.println("sample_cases: "+sample_cases.size());
			// System.out.println("real_cases: "+real_cases.size());
			con.close();
		} catch (InstantiationException | IllegalAccessException
				| ClassNotFoundException | SQLException | IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
}
