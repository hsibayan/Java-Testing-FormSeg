package helpers;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;

/**
 * Created by Hannah on 1/27/2017.
 */
public class ComputerVision {

    private static ComputerVision cv = new ComputerVision();
    public static ComputerVision getInstance() { return cv; }
    private ComputerVision() { }

    public void grayscale(Mat image) {
        Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2GRAY);
    }

    public void threshold(Mat image, boolean isInverted) {
        Imgproc.adaptiveThreshold(image, image, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 15, 6);
        if(isInverted)
        	Imgproc.threshold(image, image, 100, 255, Imgproc.THRESH_BINARY_INV);
    }
    
    public void invert(Mat image) {
        Imgproc.threshold(image, image, 0, 255, Imgproc.THRESH_BINARY_INV);
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
}
