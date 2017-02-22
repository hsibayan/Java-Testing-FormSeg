
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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Created by Hannah on 1/27/2017.
 */
public class ComputerVision3 {

    private static ComputerVision3 cv = new ComputerVision3();
    public static ComputerVision3 getInstance() { return cv; }
    private ComputerVision3() { }

    public void grayscale(Mat image) {
        Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2GRAY);
    }

    public void threshold(Mat image, boolean isInverted) {
//        Imgproc.threshold(fromMat, toMat, 0, 255, Imgproc.THRESH_OTSU);
//        Imgproc.threshold(fromMat, toMat, 130, 255, Imgproc.THRESH_BINARY);
//        Imgproc.threshold(fromMat, toMat, 130, 255, Imgproc.THRESH_BINARY);
        Imgproc.adaptiveThreshold(image, image, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 15, 6);
        if(isInverted)
        	Imgproc.threshold(image, image, 100, 255, Imgproc.THRESH_BINARY_INV);
    }
    
    public void invert(Mat image) {
        Imgproc.threshold(image, image, 0, 255, Imgproc.THRESH_BINARY_INV);
    }
    
    public Mat canny(Mat fromMat) {
        Mat toMat = new Mat();
        Imgproc.Canny(fromMat, toMat, 100, 255);
        return toMat;
    }

    public void morph(Mat fromMat, int op, int shape, int size) {
        Imgproc.morphologyEx(fromMat, fromMat, op, Imgproc.getStructuringElement(shape, new Size(size,size)));
    }
    
    public void preprocess(Mat mat) {
        grayscale(mat);
        threshold(mat, true);
    }
    
    public ArrayList<MatOfPoint> findContours(Mat img, int heirarchy) {
        ArrayList<MatOfPoint> contoursFound = new ArrayList<>();
//        Imgproc.findContours(img, contoursFound, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        Imgproc.findContours(img, contoursFound, new Mat(), heirarchy, Imgproc.CHAIN_APPROX_NONE);
        Collections.reverse(contoursFound);
        return contoursFound;
    }
    
    /* may or may not use 
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
    */
}
