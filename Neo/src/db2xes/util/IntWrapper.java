package db2xes.util;

public class IntWrapper {
	
	private int intValue;
	
	public IntWrapper() {
		
	}
	
	public IntWrapper(int intValue) {
		this.intValue = intValue;
	}
	
	public int incr() {
		this.intValue++;
		return this.intValue;
	}
	
	public int decr() {
		this.intValue--;
		return this.intValue;
	}

}
