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
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Created by Hannah on 1/27/2017.
 */
public class ComputerVision {

    private static ComputerVision cv = new ComputerVision();

    public static ComputerVision getInstance() { return cv; }

    private ComputerVision() {
    }

    public Mat grayscale(Mat fromMat) {
        Mat toMat = new Mat();
        Imgproc.cvtColor(fromMat, toMat, Imgproc.COLOR_BGR2GRAY);
        return toMat;
    }

    public Mat threshold(Mat fromMat, boolean isInverted) {
        Mat toMat = new Mat();
        Imgproc.threshold(fromMat, toMat, 100, 255, Imgproc.THRESH_OTSU);
        if(isInverted)
        	Imgproc.threshold(toMat, toMat, 100, 255, Imgproc.THRESH_BINARY_INV);
        return toMat;
    }

    public Mat canny(Mat fromMat, int t1, int t2) {
        Mat toMat = new Mat();
        Imgproc.Canny(fromMat, toMat, t1, t2);
        return toMat;
    }
    
    public ArrayList<Mat> contour2(Mat img, boolean removeBorder, int borderWidth, int minRectArea) {
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
            if(isContourRect(contours.get(i)) && isMinRect(temp.area(), minRectArea)) {
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
    
    public boolean isContourRect(MatOfPoint thisContour) {
        Rect ret = null;

        MatOfPoint2f thisContour2f = new MatOfPoint2f();
        MatOfPoint approxContour = new MatOfPoint();
        MatOfPoint2f approxContour2f = new MatOfPoint2f();

        thisContour.convertTo(thisContour2f, CvType.CV_32FC2);

        Imgproc.approxPolyDP(thisContour2f, approxContour2f, 2, true);

        approxContour2f.convertTo(approxContour, CvType.CV_32S);

        if (approxContour.size().height == 4) {
            ret = Imgproc.boundingRect(approxContour);
        }

        return (ret != null);
    }
    
    public boolean isMinRect(double area, int minRectArea) {
    	boolean isRect = false;
    	
    	if(area >= minRectArea)
    		isRect = true;
    	
    	return isRect;
    }

}
