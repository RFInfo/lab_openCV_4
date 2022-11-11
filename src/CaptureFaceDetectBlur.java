import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

public class CaptureFaceDetectBlur {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Mat src = new Mat();
        Mat grayFrame = new Mat();
        Mat grayHistoEqualizedFrame = new Mat();
        Mat dst = new Mat();

        String faceCascadeFile = "./models/haarcascades/haarcascade_frontalface_alt.xml";

        CascadeClassifier faceClassifier = new CascadeClassifier();
        faceClassifier.load(faceCascadeFile);

//        VideoCapture capture = new VideoCapture("./test_data/head-pose-face-detection-female-and-male.mp4");
//        VideoCapture capture = new VideoCapture("http://192.168.100.10:8080/video");
        VideoCapture capture = new VideoCapture(0);
        if(!capture.isOpened()) return;

        int key = 0;
        while (key != 27){

            if(!capture.read(src)) break;

            Imgproc.cvtColor(src, grayFrame, Imgproc.COLOR_BGR2GRAY);
            Imgproc.equalizeHist(grayFrame, grayHistoEqualizedFrame);

            MatOfRect faces = new MatOfRect();
            faceClassifier.detectMultiScale(grayHistoEqualizedFrame,faces);

            Rect[] facesArray = faces.toArray();
            for (int i = 0; i < facesArray.length; i++) {
                Mat faceROI = src.submat(facesArray[i]);
//                HighGui.imshow("ROI", faceROI);
                Imgproc.blur(faceROI,faceROI, new Size(25,25));
                Imgproc.rectangle(src, facesArray[i], new Scalar(0,128,0), 2);
            }

            HighGui.imshow("Src", src);
            HighGui.imshow("Gray", grayFrame);
            HighGui.imshow("GrayHistoEqualized", grayHistoEqualizedFrame);
            key = HighGui.waitKey(20);
        }

        HighGui.destroyAllWindows();
        System.exit(0);
    }
}
