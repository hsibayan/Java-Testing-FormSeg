package forms;
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;

public class Form {

	// Image
	protected Mat image;
	
	// Guides & Major Segments
	int guideCount;
	private int[] guideMatch;
	ArrayList<MatOfPoint> guideContours;
	ArrayList<MatOfPoint> fieldContours;
	
	// Major Field Types
	private int fieldCount;
	private int[] fieldTypes;
	
	// Total First Layer External Contours
	int totalContours;
	
	// Element Count for each Major Field
	private int[] elementCount;
	
	private List<List<Object>> fields;

	public Mat getImage() {
		return image;
	}

	public void setImage(Mat image) {
		this.image = image;
	}

	public int getFieldCount() {
		return fieldCount;
	}

	public void setFieldCount(int fieldCount) {
		this.fieldCount = fieldCount;
	}

	public int[] getFieldTypes() {
		return fieldTypes;
	}

	public void setFieldTypes(int[] fieldTypes) {
		this.fieldTypes = fieldTypes;
	}

	public List<List<Object>> getFields() {
		return fields;
	}

	public void setFields() {
		this.fields = new ArrayList<>();
	}

	public void addField(List<Object> field) {
		this.fields.add(field);
	}

	public int[] getElementCount() {
		return elementCount;
	}

	public void setElementCount(int[] elementCount) {
		this.elementCount = elementCount;
	}

	public int[] getGuideMatch() {
		return guideMatch;
	}

	public void setGuideMatch(int[] guideMatch) {
		this.guideMatch = guideMatch;
	}
	
}
