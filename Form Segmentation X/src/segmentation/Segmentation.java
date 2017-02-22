package segmentation;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import fields.Text;
import filesaving.FileSave;
import forms.Form;
import helpers.ComputerVision;
import helpers.Sorting;

public class Segmentation {

	private static ComputerVision cv = ComputerVision.getInstance();
	private static Sorting sort = Sorting.getInstance();
	private static FileSave fs = FileSave.getInstance();
	private static TextSegmentation textSeg = TextSegmentation.getInstance();
	
	// Field Type Variables
	private final int FIELDTYPE_TEXT = 1;
	private final int FIELDTYPE_MARK = 2;
	private final int FIELDTYPE_BLOB = 3;
	
	// Contour Modification Variables
    public final boolean SORT_ASC = true;
    public final boolean SORT_DESC = false;
	
	// Image Modification Variables
	public final int BORDER_THICKNESS_PAPER = 10;
    public final int BORDER_THICKNESS_FIELDS = 20;
    
    // Image
//    Mat binaryImage;
	
    private static Segmentation segmenter = new Segmentation();
    public static Segmentation getInstance() { return segmenter; }
    private Segmentation() { }
	
    public void segment(Form form) {
    	List<MatOfPoint> groupContours;
    	Mat paperImage = form.getImage();
    	
    	// PREPROCESS
		paperImage = cropBorder(paperImage, BORDER_THICKNESS_PAPER);
		cv.preprocess(paperImage);
//		cv.morph(paperImage, Imgproc.MORPH_OPEN, Imgproc.MORPH_ELLIPSE, 3);
		
		groupContours = cv.findContours(paperImage.clone(), Imgproc.RETR_EXTERNAL);
		groupContours = segmentMajorFields(groupContours, form.getFieldCount());
		
		int[] groupTypes = form.getFieldTypes();
		
		System.out.println("[OK] SEGMENTATION: Major Fields Good");

		Mat sampleImage = paperImage.clone();
		cv.invert(sampleImage);
		
		List<Mat> groupImages = borderRemoval(groupContours, sampleImage.clone());
		
		int size = groupImages.size();
				
		for(int i = 0; i < size; i++) {
			List<MatOfPoint> elementContours = cv.findContours(groupImages.get(i).clone(), Imgproc.RETR_EXTERNAL);
			
			switch(groupTypes[i]) {
				case FIELDTYPE_TEXT: 
					textSeg.segment(groupImages.get(i).clone(), elementContours, form, i);
					break;
				case FIELDTYPE_MARK: 
					break;
				case FIELDTYPE_BLOB:
			}
		}
	}
    
    public List<Mat> borderRemoval(List<MatOfPoint> borderContours, Mat image) {
    	List<Mat> resultImages = new ArrayList<>();
    	int size = borderContours.size();
    	Mat filled = new Mat(image.rows(), image.cols(), CvType.CV_8UC1, new Scalar(255));
    	Scalar black = new Scalar(0);
    	MatOfPoint border;
    	Rect rect;
    	int midX, midY;
    	Mat matA, matB, result;
    	
    	for(int i = 0; i < size; i++) {
    		border = borderContours.get(i);
    		rect = Imgproc.boundingRect(border);
    		
    		// border
            Mat inv = image.submat(rect);
            List<MatOfPoint> invCont = cv.findContours(inv.clone(), Imgproc.RETR_EXTERNAL);
            invCont = getLargestN(invCont, 1);

            // fill
            filled = new Mat(rect.height, rect.width, CvType.CV_8UC1, new Scalar(255));
    		Imgproc.drawContours(filled, invCont, 0, black);
    		midX = rect.width/2;
    		midY = rect.height/2;
            Imgproc.floodFill(filled, new Mat(), new Point(midX, midY), black);
            
            // submats
        	matB = filled;
        	matA = image.submat(rect);
        	
        	// subtract - removal of border
    		result = new Mat(matA.rows(), matA.cols(), CvType.CV_8UC1);
            
            Core.add(matA, matB, result);

//            Imgcodecs.imwrite("w" + i + "_0filled.png", matB);
//            Imgcodecs.imwrite("w" + i + "_1normal.png", matA);
//            Imgcodecs.imwrite("w" + i + "_2result.png", result);
            
        	cv.invert(result);
            resultImages.add(result);
    	}
    	
    	return resultImages;
    	
    }
    
    public List<MatOfPoint> segmentMajorFields(List<MatOfPoint> contours, int elementCount) {
    	List<MatOfPoint> contours2 = new ArrayList<>(contours);
    	contours2 = getLargestN(contours2, elementCount);
    	contours2 = sort.contourPositions(contours2);
    	return contours2;
    }
    
    public List<MatOfPoint> getLargestN(List<MatOfPoint> contours, int elementCount) {
    	contours = sort.contourAreas(contours, SORT_DESC);
    	return contours.subList(0, elementCount);
    }
            
    public Mat cropBorder(Mat image, int thickness) {
        int rowStart 	= thickness;
        int rowEnd 		= image.rows() - thickness;
        int colStart 	= thickness;
        int colEnd 		= image.cols() - thickness;
        
        return image.submat(rowStart, rowEnd, colStart, colEnd);
    }
}
