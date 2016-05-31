package db2xes.util;

import java.sql.Timestamp;
import java.text.DateFormat;
import java.text.SimpleDateFormat;

public class TimestampConverter {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		String s=convert("2014-07-31 14:12:48");
		System.out.println(s);
	}
	
	public static String convert(String inputs){
		Timestamp ts = new Timestamp(System.currentTimeMillis());  
        //String tsStr = "2011-05-09 13:49:45";  
		String tsStr = inputs;
        String s="";
        //System.out.println(tsStr);
        tsStr=tsStr.replaceAll("/", "-");
        DateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss"); 
       // DateFormat sdf2 = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss"); 
        try {  
            ts = Timestamp.valueOf(tsStr);  
            s=sdf.format(ts); 
           // System.out.println(ts);  
        } catch (Exception e) {
            e.printStackTrace();  
        }
		return s.replaceAll(" ", "T"); 
	}

}
