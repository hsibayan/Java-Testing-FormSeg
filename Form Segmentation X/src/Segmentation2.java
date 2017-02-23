
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

public class Segmentation2 {

	private static ComputerVision cv = ComputerVision.getInstance();
	private static FileSave fs = FileSave.getInstance();
	
	// Field Type Variables
	private final int FIELDTYPE_TEXT = 1;
	private final int FIELDTYPE_MARK = 2;
	private final int FIELDTYPE_BLOB = 3;
	
	// Filter Variables
    public final int MIDDLE_OFFSET_Y = 10;
	
	// Checker Variables
    public final int GUIDE_OFFSET = 10;
    
    // Sorter Variables
    public final static int TEXTBOX_OFFSET = 20;
	
	// Contour Modification Variables
    public final int NOISE_VALUE1 = 50;
    public final boolean SORT_ASC = true;
    public final boolean SORT_DESC = false;
    public final int SORT_X = 0;
    public final int SORT_Y = 1;
    public final int SORT_XY = 2;
	
	// Image Modification Variables
	public final int BORDER_THICKNESS_PAPER = 10;
    public final int BORDER_THICKNESS_FIELDS = 20;
    
    // Image
    Mat bwImage;
	
    private static Segmentation2 segmenter = new Segmentation2();
    public static Segmentation2 getInstance() { return segmenter; }
    private Segmentation2() { }
	
    public void segment(Form form) {
    	List<MatOfPoint> contours, fieldContours, guideContours;
    	Mat image = form.getImage();
    	
    	// PREPROCESS
		cv.preprocess(image);
//		Imgcodecs.imwrite("_Preprocess.png", image);
		image = removeBorder(image, BORDER_THICKNESS_PAPER);
    	bwImage = image.clone();
		
		contours = cv.findContours(image, Imgproc.RETR_EXTERNAL);
		contours = cleanContours(contours, NOISE_VALUE1);
		fieldContours = new ArrayList<>(contours);
		guideContours = new ArrayList<>(contours);
		
		fieldContours = segmentMajorFields(fieldContours, form.getGroupCount());
//		guideContours = segmentGuides(guideContours, form.guideCount);
		
		int[] fieldTypes = form.getGroupTypes();
		List<List<Object>> fields = form.getGroups();
		
		// IF major fields are aligned based on guide boxes
//		if(isAlignedY(form, guideContours, fieldContours)) {
			System.out.println("[OK] SEGMENTATION: Major Fields Good");
			
			List<Mat> elements = getElements(bwImage.clone(), fieldContours);
			elements = removeBorders(elements, BORDER_THICKNESS_FIELDS);
			int size = elements.size();
					
			for(int i = 0; i < size; i++) {
				List<MatOfPoint> elementContours = cv.findContours(elements.get(i).clone(), Imgproc.RETR_EXTERNAL);
//		    	writeTempImage2("_" + i + "_elementContours.png", elementContours, elements.get(i), new Scalar(255, 255, 255));
				
				switch(fieldTypes[i]) {
					case FIELDTYPE_TEXT: 
						List<MatOfPoint> textContours = segmentTextboxes(elementContours, form.getElementCount()[i], elements.get(i), i+"");
						System.out.println(" > " + textContours.size());
						List<Mat> textElements = getElements(elements.get(i).clone(), textContours);
						for(int j = 0; j < textElements.size(); j++) {
							Mat orig = textElements.get(j).clone();
							Mat inverted = orig.clone();
//							orig.convertTo(orig, CvType.CV_8U);
							Imgproc.cvtColor(orig, orig, Imgproc.COLOR_GRAY2BGR);
					    	Imgcodecs.imwrite("_" + i + "_" + j + "_orig.png", orig);

							cv.invert(inverted);
							cv.morph(inverted, Imgproc.MORPH_ERODE, Imgproc.MORPH_RECT, 5);

							List<MatOfPoint> letterContours = cv.findContours(inverted, Imgproc.RETR_EXTERNAL);
							Text text = ((Text)fields.get(i).get(j));
							System.out.println("    1. " + letterContours.size());
							letterContours = 
									getMidYContours(letterContours, text.getLetterCount(), inverted.rows()/2, inverted, "_" + i + "_" + j);	
							
							Mat newMat = Mat.zeros(inverted.rows(), inverted.cols(), CvType.CV_8UC1);
					    	
					    	for(int k = 0; k < letterContours.size(); k++)
					    		Imgproc.drawContours(newMat, letterContours, k, new Scalar(255, 255, 255));
					    	
							cv.morph(newMat, Imgproc.MORPH_DILATE, Imgproc.MORPH_RECT, 5);
					    	
					    	letterContours = cv.findContours(newMat, Imgproc.RETR_EXTERNAL);
							
							System.out.println("    2. " + letterContours.size());
							letterContours = segmentLetterboxes(letterContours, text.getLetterCount());		
					    	
//					    	for(int k = 0; k < letterContours.size(); k++)
//					    		Imgproc.drawContours(orig, letterContours, k, new Scalar(255, 255, 0), 1);
					    	

							Imgcodecs.imwrite("_" + i + "_" + j + "_lettersOnOrig.png", orig);
							
							fs.writeContours("_" + i + "_" + j + "_lettersOnly.png", 
									letterContours, textElements.get(j), new Scalar(255, 255, 255));
							
							System.out.println("    3. " + letterContours.size() + "\n");
							
							extractLetters(letterContours, orig);
						}
						
//						writeTempImage2("_" + i + "_letterContours.png", letterContours, textElements.get(i), new Scalar(255, 255, 255));
						
//						segmentLetterField(letterContours);
						
						break;
					case FIELDTYPE_MARK: 
						break;
					case FIELDTYPE_BLOB:
				}
			}
//		}
//		else {
//			System.out.println("[ERROR] SEGMENTATION: Guide boxes did not match segmented major fields");
//		}
	}
    
    // EXTRACTOR
    
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
    
    // SEGMENTERS
    
    public List<MatOfPoint> segmentMajorFields(List<MatOfPoint> contours, int elementCount) {
    	List<MatOfPoint> contours2 = new ArrayList<>(contours);
    	contours2 = getLargestContours(contours2, elementCount);
    	contours2 = sortContourPositions(contours2, SORT_XY);
    	return contours2;
    }
    
//    public List<MatOfPoint> segmentGuides(List<MatOfPoint> contours, int elementCount) {
//    	List<MatOfPoint> contours2 = new ArrayList<>(contours);
//    	contours2 = sortContourPositions(contours2, SORT_X);
//    	contours2 = contours2.subList(0, elementCount);
//    	contours2 = sortContourPositions(contours2, SORT_Y);
//    	return contours2;
//    }
    
    public List<MatOfPoint> segmentTextboxes(List<MatOfPoint> contours, int elementCount, Mat image, String name) {
    	List<MatOfPoint> contours2 = new ArrayList<>(contours);
    	contours2 = getLargestContours(contours2, elementCount);
    	contours2 = sortContourPositions(contours2, SORT_XY);
		
    	// testing
    	Mat newMat = Mat.zeros(image.rows(), image.cols(), CvType.CV_8UC3);
    	for(int i = 0; i < contours2.size(); i++) {
    		Imgproc.drawContours(newMat, contours2, i, new Scalar(255, 255, 255));
    		Rect r = Imgproc.boundingRect(contours2.get(i));
    		Imgproc.putText(newMat, ""+i, new Point(r.x, r.y), Core.FONT_HERSHEY_PLAIN, 1.5, new Scalar(0, 255, 0));
    	}
    	Imgcodecs.imwrite("_text" +name+ "_temp.png", newMat);
    	
 
    	
//    	for(int i = 0; i < contours.size(); i++) {
//    		System.out.println(" > " + Imgproc.contourArea(contours.get(i)));
//    	}
    	

    	return contours2;
    }
    
    public List<MatOfPoint> segmentLetterboxes(List<MatOfPoint> contours, int elementCount) {
    	List<MatOfPoint> contours2 = new ArrayList<>(contours);
    	contours2 = getLargestContours(contours2, elementCount);
    	contours2 = sortContourPositions(contours2, SORT_XY);
    	return contours2;
    }
    
    // FILTERING
    
    public List<MatOfPoint> getLargestContours(List<MatOfPoint> contours, int elementCount) {
    	contours = sortContourAreas(contours, SORT_DESC);
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
    	
    	Imgcodecs.imwrite("_" + name + "_midz.png", newMat);
    	
    	return contours2;
    }
    
    // CHECKERS
    
//    public boolean isAlignedY(Form form, List<MatOfPoint> guideContours, List<MatOfPoint> fieldContours) {
//    	boolean isAligned = true;
//    	int size = fieldContours.size();
//    	int[] guideMatch = form.getGuideMatch();
//    	
//    	for(int i = 0; i < size; i++) {
//    		if(!inBound(guideContours.get(guideMatch[i]-1), fieldContours.get(i))) {
//    			isAligned = false;
//    		}
//    	}
//    	return isAligned;
//    }
    
    public boolean inBound(MatOfPoint guide, MatOfPoint segment) {
    	boolean isMatched = false;
    	int yGuide = Imgproc.boundingRect(guide).y;
    	int ySegment = Imgproc.boundingRect(segment).y;
    	int lowerBound = yGuide - GUIDE_OFFSET;
    	int upperBound = yGuide + GUIDE_OFFSET;
    	
    	if(ySegment >= lowerBound && ySegment <= upperBound)
    		isMatched = true;
    	else {
    		System.out.println("g > " + yGuide);
    		System.out.println("y > " + ySegment);
    	}
    	return isMatched;
    }
    
    // SORTERS
    
    // Position Sorting
    
    public List<MatOfPoint> sortContourPositions(List<MatOfPoint> contours, int sortType) {
    	List<MatOfPoint> contours2 = new ArrayList<>(contours);
    	switch(sortType) {
    	case SORT_X: Collections.sort(contours2, positionSorterX); break;
    	case SORT_Y: Collections.sort(contours2, positionSorterY); break;
    	case SORT_XY: Collections.sort(contours2, positionSorter);
    	}
    	return contours2;
    }
    
    public static Comparator<MatOfPoint> positionSorter = new Comparator<MatOfPoint>() {
        @Override
        public int compare(MatOfPoint c1, MatOfPoint c2) {
        	Rect r1 = Imgproc.boundingRect(c1);
        	Rect r2 = Imgproc.boundingRect(c2);

        	int n;
        	int diff = r1.y - r2.y;
        	
        	// IF y values are too close to each other (probably at the same level) then compare x values instead
            if ( diff >= -TEXTBOX_OFFSET && diff <= TEXTBOX_OFFSET )		
            	n = Double.compare(r1.x, r2.x);
            else
            	n = Double.compare(r1.y, r2.y);
            
            return n;
        }
    };
    
    public static Comparator<MatOfPoint> positionSorterX = new Comparator<MatOfPoint>() {
        @Override
        public int compare(MatOfPoint c1, MatOfPoint c2) {
        	Rect r1 = Imgproc.boundingRect(c1);
        	Rect r2 = Imgproc.boundingRect(c2);

        	return Double.compare(r1.x, r2.x);
        }
    };
    
    public static Comparator<MatOfPoint> positionSorterY = new Comparator<MatOfPoint>() {
        @Override
        public int compare(MatOfPoint c1, MatOfPoint c2) {
        	Rect r1 = Imgproc.boundingRect(c1);
        	Rect r2 = Imgproc.boundingRect(c2);

        	return Double.compare(r1.y, r2.y);
        }
    };
    
    // Contour Sorting
    
    public List<MatOfPoint> sortContourAreas(List<MatOfPoint> contours, boolean isAsc) {
    	List<MatOfPoint> contours2 = new ArrayList<>(contours);
    	if(isAsc)	Collections.sort(contours2, areaSorterAsc);
    	else		Collections.sort(contours2, areaSorterDesc);
    	return contours2;
    }
    
    public static Comparator<MatOfPoint> areaSorterDesc = new Comparator<MatOfPoint>() {
        @Override
        public int compare(MatOfPoint c1, MatOfPoint c2) {
        	int area1 = (int) Imgproc.contourArea(c1);
            int area2 = (int) Imgproc.contourArea(c2);
            return area2-area1;
        }
    };
    
    public static Comparator<MatOfPoint> areaSorterAsc = new Comparator<MatOfPoint>() {
        @Override
        public int compare(MatOfPoint c1, MatOfPoint c2) {
        	int area1 = (int) Imgproc.contourArea(c1);
            int area2 = (int) Imgproc.contourArea(c2);
            return area2+area1;
        }
    };
    
    // CONTOUR MODIFICATION
    
    public List<MatOfPoint> cleanContours(List<MatOfPoint> contours, int noiseValue) {
        
    	List<MatOfPoint> newContours = new ArrayList<>();
    	int size = contours.size();
    	MatOfPoint tempContour;
    	
    	for(int i = 0; i < size; i++) {
    		tempContour = contours.get(i);
    		if(tempContour.total() > noiseValue)
    			newContours.add(contours.get(i));
    	}
    	
    	return newContours;
    	
    }
    
    // IMAGE MODIFICATION
    
    public Mat getSubImage(Mat image, MatOfPoint contour) {
    	Rect contourRect = Imgproc.boundingRect(contour);
    	return image.submat(contourRect);
    }
    public List<Mat> removeBorders(List<Mat> images, int thickness) {
    	List<Mat> images2 = new ArrayList<>();
    	int size = images.size();
    	for(int i = 0; i < size; i++)
    		images2.add(removeBorder(images.get(i), thickness));
    	return images2;
    }
    
    public Mat removeBorder(Mat image, int thickness) {
        int rowStart 	= thickness;
        int rowEnd 		= image.rows() - thickness;
        int colStart 	= thickness;
        int colEnd 		= image.cols() - thickness;
        
        return image.submat(rowStart, rowEnd, colStart, colEnd);
    }
    
    // WRITING - for testing only
        
//    public void writeContours(String filename, List<MatOfPoint> contours, Mat subImage, Scalar color) {
//    	Mat newMat = Mat.zeros(subImage.rows(), subImage.cols(), CvType.CV_8UC3);
//    	
//    	for(int i = 0; i < contours.size(); i++)
//    		Imgproc.drawContours(newMat, contours, i, color);
//    	
//    	Imgcodecs.imwrite(filename, newMat);
//    }
}
