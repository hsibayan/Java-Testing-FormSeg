package segmentation;

import java.io.File;
import java.util.ArrayList;
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

public class TextSegmentation {

	private static ComputerVision cv = ComputerVision.getInstance();
	private static Sorting sort = Sorting.getInstance();
	private static FileSave fs = FileSave.getInstance();

    public final int MIDDLE_OFFSET_Y = 10;

    private static TextSegmentation textSeg = new TextSegmentation();
    public static TextSegmentation getInstance() { return textSeg; }
    private TextSegmentation() { }
    
    public void segment(Mat elementImage, List<MatOfPoint> elementContours, Form form, int parentIndex) {
		List<List<Object>> fields = form.getFields();
		
		List<MatOfPoint> textContours = segmentTextboxes(elementContours, form.getElementCount()[parentIndex]);
		System.out.println(" > " + textContours.size());
		List<Mat> textElements = getElements(elementImage, textContours);
		
		
		for(int textIndex = 0; textIndex < textElements.size(); textIndex++) {
			Mat orig = textElements.get(textIndex).clone();
			Mat inverted = orig.clone();
			Imgproc.cvtColor(orig, orig, Imgproc.COLOR_GRAY2BGR);
//	    	Imgcodecs.imwrite("_" + i + "_" + j + "_orig.png", orig);

			cv.invert(inverted);
			cv.morph(inverted, Imgproc.MORPH_ERODE, Imgproc.MORPH_RECT, 5);

			List<MatOfPoint> letterContours = cv.findContours(inverted, Imgproc.RETR_EXTERNAL);
			Text text = ((Text)fields.get(parentIndex).get(textIndex));
			System.out.println("    1. " + letterContours.size());
			letterContours = 
					getMidYContours(letterContours, text.getLetterCount(), inverted.rows()/2, inverted, "_" + parentIndex + "_" + textIndex);	
			
			Mat newMat = Mat.zeros(inverted.rows(), inverted.cols(), CvType.CV_8UC1);
	    	
	    	for(int k = 0; k < letterContours.size(); k++)
	    		Imgproc.drawContours(newMat, letterContours, k, new Scalar(255, 255, 255));
	    	
			cv.morph(newMat, Imgproc.MORPH_DILATE, Imgproc.MORPH_RECT, 5);
	    	
	    	letterContours = cv.findContours(newMat, Imgproc.RETR_EXTERNAL);
			
			System.out.println("    2. " + letterContours.size());
			letterContours = segmentLetterboxes(letterContours, text.getLetterCount());		
	    	
//		    	for(int k = 0; k < letterContours.size(); k++)
//		    		Imgproc.drawContours(orig, letterContours, k, new Scalar(255, 255, 0), 1);
	    	

//			Imgcodecs.imwrite("_" + parentIndex + "_" + textIndex + "_lettersOnOrig.png", orig);
//			
//			fs.writeContours("_" + parentIndex + "_" + textIndex + "_lettersOnly.png", 
//					letterContours, textElements.get(textIndex), new Scalar(255, 255, 255));
			
			System.out.println("    3. " + letterContours.size() + "\n");
			
			extractLetters(letterContours, orig);
		}
    }
    
    public List<MatOfPoint> segmentTextboxes(List<MatOfPoint> contours, int elementCount) {
    	List<MatOfPoint> contours2 = new ArrayList<>(contours);
    	contours2 = getLargestContours(contours2, elementCount);
    	contours2 = sort.contourPositions(contours2);
    	
     	return contours2;
    }
    
    public List<MatOfPoint> segmentLetterboxes(List<MatOfPoint> contours, int elementCount) {
    	List<MatOfPoint> contours2 = new ArrayList<>(contours);
    	contours2 = getLargestContours(contours2, elementCount);
    	contours2 = sort.contourPositions(contours2);
    	return contours2;
    }
    
    public List<MatOfPoint> getLargestContours(List<MatOfPoint> contours, int elementCount) {
    	contours = sort.contourAreas(contours, sort.SORT_DESC);
    	return contours.subList(0, elementCount);
    }
    
    public List<Mat> getElements(Mat image, List<MatOfPoint> elementContours) {
    	List<Mat> elements = new ArrayList<>();
    	int size = elementContours.size();
    	for(int i = 0; i < size; i++) {
    		elements.add(getSubImage(image, elementContours.get(i)));
    	}
    	return elements;
    }

    public Mat getSubImage(Mat image, MatOfPoint contour) {
    	Rect contourRect = Imgproc.boundingRect(contour);
    	return image.submat(contourRect);
    }
    
    public List<MatOfPoint> getMidYContours(List<MatOfPoint> contours, int elementCount, int midY, Mat subImage, String name) {
    	List<MatOfPoint> contours2 = new ArrayList<>();
    	Rect rect;
    	int size = contours.size();
    	int lowerBound = midY - MIDDLE_OFFSET_Y;
    	int upperBound = midY + MIDDLE_OFFSET_Y;
    	int mid;

    	Mat newMat = Mat.zeros(subImage.rows(), subImage.cols(), CvType.CV_8UC3);
    	Imgproc.line(newMat, new Point(0, midY), new Point(newMat.width(), midY), new Scalar(0, 255, 0));
    	Imgproc.line(newMat, new Point(0, lowerBound), new Point(newMat.width(), lowerBound), new Scalar(255, 255, 0));
    	Imgproc.line(newMat, new Point(0, upperBound), new Point(newMat.width(), upperBound), new Scalar(255, 255, 0));
    	
    	for(int i = 0; i < size; i++) {
    		rect = Imgproc.boundingRect(contours.get(i));
    		mid = (int) ( rect.y + (rect.height / 2) );
    		int mid1 = (int) ( rect.x + (rect.width / 2) );
    		int mid2 = (int) ( rect.y + (rect.height / 2) );
    		if(mid >= lowerBound && mid <= upperBound)
    			contours2.add(contours.get(i));
    		else {
//    			System.out.println(" ---> " + midY);
//    			System.out.println(" ---< " + mid);
    		}
        	Imgproc.circle(newMat, new Point(mid1, mid2), 5, new Scalar(255, 255, 255));
    	}

    	return contours2;
    }
    
    public List<Mat> extractLetters(List<MatOfPoint> borderContours, Mat orig) {
    	List<Mat> extracted = new ArrayList<>();
    	Scalar black = new Scalar(0);
    	int size = borderContours.size();
    	List<MatOfPoint> letter;
    	Mat ext;

    	Imgproc.cvtColor(orig, orig, Imgproc.COLOR_BGR2GRAY);
    	
    	List<Mat> noborder = borderRemoval(borderContours, orig);
    	Mat temp;
    	
    	for(int i = 0; i < size; i++) {
    		temp = noborder.get(i);
            
            // largest contour -> the letter
        	letter = cv.findContours(temp, Imgproc.RETR_CCOMP);
        	letter = getLargestContours(letter, 1);
        	
        	// drawing on white background
        	ext = new Mat(temp.height(), temp.width(), CvType.CV_8UC1, new Scalar(255));
        	Imgproc.drawContours(ext, letter, 0, black, -1);
        	
    		Imgcodecs.imwrite("asd" + File.separator  + "_ltr_" + i + ".png", ext);
    	}
    	
    	return extracted;
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
    		// fill
    		Imgproc.drawContours(filled, borderContours, i, black);
    		border = borderContours.get(i);
    		rect = Imgproc.boundingRect(border);
    		midX = rect.x + rect.width/2;
    		midY = rect.y + rect.height/2;
            Imgproc.floodFill(filled, new Mat(), new Point(midX, midY), black);

            // submats
        	matB = filled.submat(rect);
        	matA = image.submat(rect);
        	
        	// subtract - removal of border
    		result = new Mat(image.rows(), image.cols(), CvType.CV_8UC1);
            
            Core.subtract(matA, matB, result);
            
            resultImages.add(result);
    	}
    	
    	return resultImages;
    	
    }
    
}
