package db2xes.util;

public class Event {
	String activity;
	String timestamp;
	String resource;
	String caseid;
	
	public Event(String activity1, String timestamp1, String resource1, String caseid1 ){

		activity=activity1;
		timestamp=(timestamp1 != null) ? TimestampConverter.convert(timestamp1) : null;
		resource=resource1;
		caseid=caseid1;
		
	}
	
	public String getActivity() {
		return activity;
	}


	public void setActivity(String activity) {
		this.activity = activity;
	}


	public String getTimestamp() {
		return timestamp;
	}


	public void setTimestamp(String timestamp) {
		this.timestamp = timestamp;
	}


	public String getResource() {
		return resource;
	}


	public void setResource(String resource) {
		this.resource = resource;
	}


	public String getCaseid() {
		return caseid;
	}


	public void setCaseid(String caseid) {
		this.caseid = caseid;
	}


	public void print(){
		System.out.println("activity:"+activity+" timestamp:"+timestamp+" resource:"+resource+" caseID:"+caseid);
	}
	

}
