
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import forms.Form;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Created by Hannah on 1/27/2017.
 */
public class ComputerVision2 {

    private static ComputerVision2 cv = new ComputerVision2();
    public static ComputerVision2 getInstance() { return cv; }
    private ComputerVision2() { }

    public final int NOISE_VALUE1 = 50;
    public final int NOISE_VALUE2 = 300;
    public final int GUIDE_VALUE = 200;
    public final int GUIDE_POSITION_X = 110;
    public final int GUIDE_OFFSET = 10;
    public final int BORDER_THICKNESS_PAPER = 10;
    public final int BORDER_THICKNESS_FIELDS = 8;

    public Mat grayscale(Mat fromMat) {
        Mat toMat = new Mat();
        Imgproc.cvtColor(fromMat, toMat, Imgproc.COLOR_BGR2GRAY);
        return toMat;
    }

    public Mat threshold(Mat fromMat, boolean isInverted) {
        Mat toMat = new Mat();
//        Imgproc.threshold(fromMat, toMat, 0, 255, Imgproc.THRESH_OTSU);
//        Imgproc.threshold(fromMat, toMat, 130, 255, Imgproc.THRESH_BINARY);
//        Imgproc.threshold(fromMat, toMat, 130, 255, Imgproc.THRESH_BINARY);
        Imgproc.adaptiveThreshold(fromMat, toMat, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 15, 6);
        if(isInverted)
        	Imgproc.threshold(toMat, toMat, 100, 255, Imgproc.THRESH_BINARY_INV);
        return toMat;
    }
    
    public Mat change(Mat fromMat) {
        Mat toMat = new Mat();
        fromMat.convertTo(toMat, -1, 2, 50);
        return toMat;
    }

    public Mat resize(Mat fromMat) {
        Mat toMat = new Mat();
        int a = fromMat.cols() * 2;
        int b = fromMat.rows() * 2;
        Imgproc.resize(fromMat, toMat, new Size(a, b));
        return toMat;
    }
    
    public Mat canny(Mat fromMat) {
        Mat toMat = new Mat();
        Imgproc.Canny(fromMat, toMat, 100, 255);
        return toMat;
    }
   
    public Mat hist(Mat fromMat) {
        Mat toMat = new Mat();
        Imgproc.equalizeHist(fromMat, toMat);
        return toMat;
    }
   
    public Mat morphClose(Mat fromMat) {
        Mat toMat = new Mat();
        Imgproc.morphologyEx(fromMat, toMat, Imgproc.MORPH_CLOSE, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(2,2)));
        return toMat;
    }
    
    public Mat preprocess(Mat mat) {
        mat = grayscale(mat);
        mat = threshold(mat, true);
        return mat;
    }
    
    public ArrayList<Mat> contour(Mat img, boolean removeBorder, int borderWidth, int minRectArea) {
    	Mat img2 = img.clone();
    	
        ArrayList<Mat> allMats = new ArrayList<>();

        Mat newMat = Mat.zeros(img.rows(), img.cols(), CvType.CV_8UC3);
        allMats.add(newMat);

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(img, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        
        Random rand = new Random();

        for(int i = 0; i < contours.size(); i++) {
            Imgproc.drawContours(newMat, contours, i, new Scalar(255, 255, 255));

            Rect temp = Imgproc.boundingRect(contours.get(i));
            int midX = temp.x + temp.width/2;
            int midY = temp.y + temp.height/2;
            int a = rand.nextInt(200) + 40;
            int b = rand.nextInt(200) + 40;
            int c = rand.nextInt(200) + 40;
            
            int n = borderWidth;
            int rowStart 	= temp.y + n;
            int rowEnd 		= temp.y + temp.height - n;
            int colStart 	= temp.x + n;
            int colEnd 		= temp.x + temp.width - n;
            Point p1 = new Point(colStart, rowStart);
            Point p2 = new Point(colEnd, rowEnd);
            
//            Imgproc.rectangle(origMat, p1, p2, new Scalar(255, 0, 0));
            
            // IF contour is a rect
            if(isMinRect(temp.area(), minRectArea)) {
                Imgproc.floodFill(newMat, new Mat(),
                        new Point(midX, midY), new Scalar(a, b, c, 30));
                
                Mat subMat = new Mat();
                
                if(removeBorder) 
                    subMat = img2.submat(rowStart, rowEnd, colStart, colEnd);
                else
                	subMat = img2.submat(temp); 
            	allMats.add(subMat);
            }
        }

        Collections.reverse(allMats);
        return allMats;
    }
    
    public int findSmallest(ArrayList<int[]> list) {
    	int smallestIndex = 0;
    	int size = list.size();
    	Integer smallestX = list.get(0)[0];
    	Integer tempX;
    	
    	for(int i = 1; i < size; i++) {
    		tempX = list.get(i)[0];
    		if(tempX < smallestX) {
    			smallestX = list.get(i)[0];
    			smallestIndex = i;
    		}
    	}
    	
    	return smallestIndex;
    }
    
    public ArrayList<MatOfPoint> getGuides2(ArrayList<MatOfPoint> contours, int guideCount) {
    	ArrayList<int[]> list = new ArrayList<>();
    	
    	for(int i = 0; i < contours.size(); i++) {
    		Rect rect = Imgproc.boundingRect(contours.get(i));
    		int[] temp = {rect.x, i};
    		list.add(temp);
    	}
    	
    	ArrayList<MatOfPoint> guides = new ArrayList<>();
    	for(int i = 0; i < guideCount; i++) {
    		int smallest = findSmallest(list);
    		MatOfPoint guide = contours.get(list.get(smallest)[1]);
    		list.remove(smallest);
//    		contours.remove(smallest);
    		guides.add(guide);
    		System.out.println(" > " + Imgproc.boundingRect(guides.get(i)).y);
    	}
    	
    	/*
    	
    	ArrayList<Integer> xList = new ArrayList<>();
    	for(int i = 0; i < contours.size(); i++) {
    		Rect rect = Imgproc.boundingRect(contours.get(i));
    		xList.add(rect.x);
    	}
    	
    	ArrayList<MatOfPoint> guides = new ArrayList<>();
    	for(int i = 0; i < guideCount; i++) {
    		int smallest = findSmallest(xList);

    		MatOfPoint guide = contours.get(smallest);
    		contours.remove(smallest);
    		xList.remove(smallest);
    		guides.add(guide);
    	}
    	
    	Collections.reverse(guides);
    	
    	*/
    	
    	ArrayList<Integer> guides2 = new ArrayList<>();
    	for(int i = 0; i < guides.size(); i++) {
    		Rect rect = Imgproc.boundingRect(guides.get(i));
    		System.out.println(rect.x + ", " + rect.y);
    	}
    	
    	
    	
    	return guides;
    }
    
    public Mat removeBorder(Mat image, int BORDER_THICKNESS) {
    	
        int rowStart 	= BORDER_THICKNESS;
        int rowEnd 		= image.rows() - BORDER_THICKNESS;
        int colStart 	= BORDER_THICKNESS;
        int colEnd 		= image.cols() - BORDER_THICKNESS;
        
//        System.out.println(colStart + "," + rowStart + " ~ " + colEnd + "," + rowEnd);
        
        Mat subMat = image.submat(rowStart, rowEnd, colStart, colEnd);
        
        return subMat;
    }
    
    public Mat getSubImage(Mat image, MatOfPoint contour) {
    	Rect contourRect = Imgproc.boundingRect(contour);
    	return image.submat(contourRect);
    }
    
//    public boolean isAligned(Form form) {
//    	boolean isCorrect = true;
//    	int size = form.fieldContours.size();
//    	int[] guideMatch = form.getGuideMatch();
//    	ArrayList<MatOfPoint> guideContours = form.guideContours;
//    	ArrayList<MatOfPoint> fieldContours = form.fieldContours;
//    	
//    	for(int i = 0; i < size; i++) {
//    		if(!matches(guideContours.get(guideMatch[i]-1), fieldContours.get(i))) {
//    			isCorrect = false;
//    		}
//    	}
//    	
//    	return isCorrect;
//    }
    
    public boolean matches(MatOfPoint guide, MatOfPoint segment) {
    	boolean isMatch = false;
    	int yGuide = Imgproc.boundingRect(guide).y;
    	int ySegment = Imgproc.boundingRect(segment).y;
    	int lowerBound = yGuide - GUIDE_OFFSET;
    	int upperBound = yGuide + GUIDE_OFFSET;
    	
    	if(ySegment >= lowerBound && ySegment <= upperBound)
    		isMatch = true;
    	
    	return isMatch;
    }
    
    public boolean isRightGuideCount(ArrayList<MatOfPoint> guideContours, int guideCount) {
    	if(guideContours.size() == guideCount)
    		return true;
    	else return false;
    }
    
    public boolean isRightFieldCount(ArrayList<MatOfPoint> fieldContours, int fieldCount) {
    	if(fieldContours.size() == fieldCount)
    		return true;
    	else return false;
    }
    
    public ArrayList<MatOfPoint> getMajorFields(ArrayList<MatOfPoint> contours) {
    	return removeContours(contours, GUIDE_VALUE);
    }
    
    public ArrayList<MatOfPoint> getGuides(ArrayList<MatOfPoint> contours) {

    	ArrayList<MatOfPoint> guideContours = new ArrayList<>();
    	int size = contours.size();
    	MatOfPoint tempContour;
    	Rect tempRect;
    	
    	for(int i = 0; i < size; i++) {
    		tempContour = contours.get(i);
    		tempRect = Imgproc.boundingRect(tempContour);
    		if(tempRect.x <= GUIDE_POSITION_X) {
    			guideContours.add(tempContour);
    		}
    	}
    	
    	return guideContours;
    }

    public ArrayList<MatOfPoint> removeContours(ArrayList<MatOfPoint> contours, int noiseValue) {
    
    	ArrayList<MatOfPoint> newContours = new ArrayList<>();
    	int size = contours.size();
    	MatOfPoint tempContour;
    	
    	for(int i = 0; i < size; i++) {
    		tempContour = contours.get(i);
    		if(tempContour.total() > noiseValue)
    			newContours.add(contours.get(i));
    	}
    	
    	return newContours;
    	
    }
    
    public ArrayList<MatOfPoint> findContours(Mat img, int heirarchy) {
        ArrayList<MatOfPoint> contoursFound = new ArrayList<>();
//        Imgproc.findContours(img, contoursFound, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        Imgproc.findContours(img, contoursFound, new Mat(), heirarchy, Imgproc.CHAIN_APPROX_NONE);
        Collections.reverse(contoursFound);
        return contoursFound;
    }
    
    // unsure
    public ArrayList<MatOfPoint> getMinAreaContour(ArrayList<MatOfPoint> contours, int minArea) {
    	ArrayList<MatOfPoint> contours2 = new ArrayList<>();
    	int size = contours.size();
    	Rect rect;
    	
    	for(int i = 0; i < size; i++) {
    		rect = Imgproc.boundingRect(contours.get(i));
    		if(rect.area() >= minArea)
    			contours2.add(contours.get(i));
    	}
    	
    	return contours2;
    }
    
    public ArrayList<MatOfPoint> removeBorderContour(ArrayList<MatOfPoint> contours) {
    	ArrayList<Double> areas = new ArrayList<>();
    	for(int i = 0; i < contours.size(); i++) {
        	Rect rect = Imgproc.boundingRect(contours.get(i));	
            areas.add(rect.area());
    	}
    	int largestContour = findLargestContour(areas);
    	contours.remove(largestContour);
    	return contours;
    }
    
    public int findLargestContour(ArrayList<Double> areas) {
    	int largestIndex = 0;
    	int size = areas.size();
    	Double largestArea = areas.get(0);
    	Double tempArea;
    	
    	for(int i = 1; i < size; i++) {
    		tempArea = areas.get(i);
    		if(tempArea > largestArea) {
    			largestArea = areas.get(i);
    			largestIndex = i;
    		}
    	}
    	
    	return largestIndex;
    }
    
    public ArrayList<MatOfPoint> getRectContours(ArrayList<MatOfPoint> contours) {
    	ArrayList<MatOfPoint> rectContours = new ArrayList<>();
    	
    	for(int i = 0; i < contours.size(); i++) {
            // IF contour is a rectangle
            if(isContourRect(contours.get(i))) 
                rectContours.add(contours.get(i));
        }
    
    	return rectContours;
    }
    
    public ArrayList<MatOfPoint> getCircleContours(ArrayList<MatOfPoint> contours) {
    	ArrayList<MatOfPoint> circleContours = new ArrayList<>();
    	
    	for(int i = 0; i < contours.size(); i++) {
            // IF contour is a circle
            if(isContourCircle(contours.get(i))) 
                circleContours.add(contours.get(i));
        }
    
    	return circleContours;
    }
    
    public boolean isContourRect(MatOfPoint thisContour) {
//        Rect rect = null;
        boolean isRect = false;

        MatOfPoint2f thisContour2f = new MatOfPoint2f();
        MatOfPoint approxContour = new MatOfPoint();
        MatOfPoint2f approxContour2f = new MatOfPoint2f();

        thisContour.convertTo(thisContour2f, CvType.CV_32FC2);

        Imgproc.approxPolyDP(thisContour2f, approxContour2f, 2, true);

        approxContour2f.convertTo(approxContour, CvType.CV_32S);

        if (approxContour.size().height == 4) {
            isRect = true;
        }

        return isRect;
    }
    
    public boolean isContourCircle(MatOfPoint thisContour) {
//        Rect rect = null;
        boolean isCircle = false;

        MatOfPoint2f thisContour2f = new MatOfPoint2f();
        MatOfPoint approxContour = new MatOfPoint();
        MatOfPoint2f approxContour2f = new MatOfPoint2f();

        thisContour.convertTo(thisContour2f, CvType.CV_32FC2);

        Imgproc.approxPolyDP(thisContour2f, approxContour2f, 2, true);

        approxContour2f.convertTo(approxContour, CvType.CV_32S);

        if (approxContour.size().height > 15) {
            isCircle = true;
        }

        return isCircle;
    }
    
    public boolean isMinRect(double area, int minRectArea) {
    	boolean isRect = false;
    	
    	if(area >= minRectArea)
    		isRect = true;
    	
    	return isRect;
    }

}
