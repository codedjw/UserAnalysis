package sjtu.ist.util;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class SQLConnFactory {
	
	public static Connection getCon(String dbname) throws InstantiationException, IllegalAccessException, ClassNotFoundException, SQLException{
		Class.forName("com.mysql.jdbc.Driver");
		Class.forName("com.mysql.jdbc.Driver").newInstance();
//		String url="jdbc:mysql://localhost:3306/cop?user=root&password=110211";
		String url="jdbc:mysql://localhost:3306/"+dbname;
		Connection con = DriverManager.getConnection(url, "root", "");
		return con;
	}

}
