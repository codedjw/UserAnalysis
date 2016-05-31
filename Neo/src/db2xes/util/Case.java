package db2xes.util;

import java.util.ArrayList;
import java.util.List;

public class Case {
	
	String case_id;
	List<Event> events = new ArrayList<Event>();
	
	public Case() {
		
	}
	
	public Case(String case_id, List<Event> events) {
		this.case_id = case_id;
		this.events = events;
	}

	public String getCase_id() {
		return case_id;
	}

	public void setCase_id(String case_id) {
		this.case_id = case_id;
	}

	public List<Event> getEvents() {
		return events;
	}

	public void setEvents(List<Event> events) {
		this.events = events;
	}

	@Override
	public String toString() {
		return "Case [case_id=" + case_id + ", events=" + events + "]";
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((case_id == null) ? 0 : case_id.hashCode());
		result = prime * result + ((events == null) ? 0 : events.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Case other = (Case) obj;
		if (case_id == null) {
			if (other.case_id != null)
				return false;
		} else if (!case_id.equals(other.case_id))
			return false;
		if (events == null) {
			if (other.events != null)
				return false;
		} else if (!events.equals(other.events))
			return false;
		return true;
	}
	
	public boolean structureEquals(Case other) {
		List<Event> this_events = this.events;
		List<Event> other_events = other.events;
		if (this_events == null) {
			if (other.events != null) {
				return false;
			}
		} else {
			if (other.events == null) {
				return false;
			} else {
				if (this_events.size() != other_events.size()) {
					return false;
				} else {
					for (int i=0; i<this_events.size(); i++) {
						Event te = this_events.get(i);
						Event oe = other_events.get(i);
						if (!te.getActivity().equals(oe.getActivity())) {
							return false;
						}
					}
				}
			}
		}
		return true;
	}

}
