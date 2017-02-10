import java.awt.List;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;

import javax.imageio.ImageIO;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

public class Main {

	private static ComputerVision cv = ComputerVision.getInstance();
	private final int FIELDTYPE_TEXT = 1;
	private final int FIELDTYPE_MARK = 2;
	private final int FIELDTYPE_BLOB = 3;
	
	public static void main(String args[]) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		Main m = new Main();
		int[] form1 = {1, 1, 1, 1, 3, 3, 2, 2, 3};
		m.start(form1, "form7.png");

		System.out.println("dan");
	}
	
	private void start(int[] field_types, String img_filename) {
		
		String parent_folder = "segmented-1";

		createFolder(parent_folder);
		
		// PREPROCESS
		Mat orig = Imgcodecs.imread(img_filename);
		Mat gray = cv.grayscale(orig);
		Mat bw = cv.threshold(gray, true);
		
		// PART 1
		ArrayList<Mat> contours_1 = cv.contour2(bw, true, 5, 200);
		int length = contours_1.size()-1;
		
		// IF actual major segments == extracted major segments
		if(field_types.length == length) {
			System.out.println("No of Actual Major Segments & Extracted Major Segments --> Equal");
			createFolder(parent_folder);
			saveContours(parent_folder, contours_1);
			for(int i = 0; i < length; i++) {
				String temp_in = parent_folder + File.separator + i + ".png";
				String temp_out = parent_folder + File.separator + i;
				
				switch(field_types[i]) {
					case FIELDTYPE_TEXT: crop_text(contours_1.get(i), temp_out);break;
					case FIELDTYPE_MARK: crop_marks(contours_1.get(i), temp_out); break;
					case FIELDTYPE_BLOB: crop_blob(contours_1.get(i), temp_out);	
				}
			}
		}
		else {
			System.out.println("No of Actual Major Segments & Extracted Major Segments --> Not Equal");
			System.out.println("con > " + contours_1.size());
			System.out.println("passed > " + field_types.length);
		}
		
	}
		
	private void crop_text(Mat img, String out_filename) {
		ArrayList<Mat> contours = cv.contour2(img, false, 0, 200);
		save(out_filename, contours);
		
		for(int i = 0; i < contours.size()-1; i++) {
			String temp_out = out_filename + File.separator + i;
			crop_letterboxes(contours.get(i), temp_out);
		}
	}
	
	private void crop_letterboxes(Mat img, String out_filename) {
		img = cv.threshold(img, true);
		ArrayList<Mat> contours = cv.contour2(img, true, 1, 0);
		save(out_filename, contours);
	}
	
	private void crop_marks(Mat img, String out_filename) {
		ArrayList<Mat> contours = cv.contour2(img, false, 0, 150);
		save(out_filename, contours);
	}

	private void crop_blob(Mat img, String out_filename) {
		img = cv.threshold(img, true);
		createFolder(out_filename);
		Imgcodecs.imwrite(out_filename + File.separator + "0.png", img);
	}
	
	private void createFolder(String filename) {
		File folder = new File(filename);
		folder.mkdir();
	}
	
	private void saveContours(String out_filename, ArrayList<Mat> contours) {
		String temp_filename;
		for(int i = 0; i < contours.size(); i++) {
			temp_filename = out_filename + File.separator + i + ".png";
			Imgcodecs.imwrite(temp_filename, contours.get(i));
		}
	}
	
	private void save(String out_filename, ArrayList<Mat> contours) {
		createFolder(out_filename);
		saveContours(out_filename, contours);
	}
	
}
