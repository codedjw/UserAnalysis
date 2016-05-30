package sjtu.ist.input;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;

import org.dom4j.Document;
import org.dom4j.DocumentException;
import org.dom4j.Element;
import org.dom4j.Namespace;
import org.dom4j.QName;
import org.dom4j.io.SAXReader;

/**
 * <p> 读取XES文件到数据库中 </p>
 * @author dujiawei
 * @version 1.0
 */
public class EventXES2DB {
	
	public static void eventXes2DB(String filename, String sqlname) throws DocumentException, IOException {
		File sqlfile = new File(sqlname);
		if (!sqlfile.exists()) {
			sqlfile.createNewFile();
		}
		BufferedWriter bw = new BufferedWriter(new FileWriter(sqlfile, false));
		SAXReader reader = new SAXReader();
		Document document = reader.read(new File(filename));
		Element log = document.getRootElement();
		List traces = log.elements("trace");
		for (Iterator it = traces.iterator(); it.hasNext();) {
			Element trace = (Element) it.next();
			List info = trace.elements("string");
			String case_id = ((Element)info.get(0)).attributeValue("value");
			case_id = case_id.substring(4);
			List events = trace.elements("event");
			for (Iterator eit = events.iterator(); eit.hasNext();) {
				Element event = (Element) eit.next();
				List einfo1 = event.elements("string");
				List einfo2 = event.elements("date");
				String activity = ((Element)einfo1.get(0)).attributeValue("value");
				String user_id = ((Element)einfo1.get(1)).attributeValue("value");
				String timestamp = ((Element)einfo2.get(0)).attributeValue("value");
				timestamp = timestamp.substring(0, timestamp.indexOf('+')).replaceAll("T", " ");
				String query = "INSERT INTO qyw_7th_yy_succ_all(CASE_ID, USER_ID, VISIT_TIME, VISIT_MEAN) VALUES (\'" + case_id + "\', \'" + user_id + "\', \'" + timestamp + "\', \'" + activity + "\');";
				bw.write(query);
				bw.newLine();
			}
		}
		bw.close();
	}
	
	public static void main(String[] args) {
		try {
			String filepath = "/Users/dujiawei/Desktop/流程挖掘案例/趣医网/趣医网-第七阶段/XES/";
//			String name = "趣医网第七次日志_新用户_预约业务（成功）";
			String name = "趣医网第七次日志_老用户_预约业务（成功）";
			EventXES2DB.eventXes2DB(filepath+name+".xes",filepath+name+".sql");
			System.out.println("Over");
		} catch (DocumentException | IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
}
