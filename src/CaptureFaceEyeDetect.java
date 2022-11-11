import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.videoio.VideoCapture;

public class CaptureFaceEyeDetect {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Mat src = new Mat();
        Mat grayFrame = new Mat();
        Mat grayHistoEqualizedFrame = new Mat();
        Mat dst = new Mat();

        String faceCascadeFile = "./models/haarcascades/haarcascade_frontalface_alt.xml";
        String eyeCascadeFile = "./models/haarcascades/haarcascade_eye.xml";

        CascadeClassifier faceClassifier = new CascadeClassifier();
        faceClassifier.load(faceCascadeFile);

        CascadeClassifier eyeClassifier = new CascadeClassifier();
        eyeClassifier.load(eyeCascadeFile);

//        VideoCapture capture = new VideoCapture("./test_data/head-pose-face-detection-female-and-male.mp4");
        VideoCapture capture = new VideoCapture(0);
        if (!capture.isOpened()) return;

        int key = 0;
        while (key != 27) {

            if (!capture.read(src)) break;

            Imgproc.cvtColor(src, grayFrame, Imgproc.COLOR_BGR2GRAY);
            Imgproc.equalizeHist(grayFrame, grayHistoEqualizedFrame);

            MatOfRect faces = new MatOfRect();
//            faceClassifier.detectMultiScale(grayHistoEqualizedFrame,faces);
//            faceClassifier.detectMultiScale(grayHistoEqualizedFrame, faces, 1.1, 4, Objdetect.CASCADE_SCALE_IMAGE, new Size(100, 100));
            faceClassifier.detectMultiScale(grayFrame, faces, 1.1, 4, Objdetect.CASCADE_SCALE_IMAGE, new Size(100, 100));

            Rect[] facesArray = faces.toArray();
            for (int i = 0; i < facesArray.length; i++) {

//                Mat faceROI = grayHistoEqualizedFrame.submat(facesArray[i]);
                Mat faceROI = grayFrame.submat(facesArray[i]);

                MatOfRect eyes = new MatOfRect();
                eyeClassifier.detectMultiScale(faceROI, eyes);

                Rect[] eyesArray = eyes.toArray();
                for (int j = 0; j < eyesArray.length && j < 2; j++) {
                    Point center = new Point(facesArray[i].x + eyesArray[j].x + eyesArray[j].width / 2,
                            facesArray[i].y + eyesArray[j].y + eyesArray[j].height / 2);
                    int radius = (int) Math.round((eyesArray[j].width + eyesArray[j].height) * 0.25);
                    Imgproc.circle(src, center, radius, new Scalar(0, 0, 128), 2);
                }

                Imgproc.rectangle(src, facesArray[i], new Scalar(0, 128, 0), 2);
            }

            HighGui.imshow("Src", src);
            HighGui.imshow("Gray", grayFrame);
            HighGui.imshow("GrayHisto", grayHistoEqualizedFrame);
            key = HighGui.waitKey(20);
        }

        HighGui.destroyAllWindows();
        System.exit(0);
    }
}
