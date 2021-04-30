package com.example.imagepro;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

public class facialDetection {
    // define Interpreter
    private Interpreter interpreter;
    // now define input size and pixel size
    private int INPUT_SIZE;
    private int PIXEL_SIZE=1;
    // it is use to divide image by 255 to scale it from 0-1
    private float IMAGE_STD=255.0f;
    private float IMAGE_MEAN=0;
    // it is used to initial GPU on your app
    private GpuDelegate gpuDelegate=null;

    // define height and weight
    private  int height=0;
    private int width=0;
    private CascadeClassifier cascadeClassifier;

    // on start
    facialDetection(AssetManager assetManager, Context context, String modelPath, int inputSize) throws IOException{
        INPUT_SIZE=inputSize;

        // define GPU and number of thread to Interpreter
        Interpreter.Options options=new Interpreter.Options();
        gpuDelegate=new GpuDelegate();
        options.addDelegate(gpuDelegate);
        options.setNumThreads(4); // change number of thread according to your phone

        // load CNN model
        interpreter=new Interpreter(loadModelFile(assetManager,modelPath),options);
        Log.d("FacialDetector","CNN model is loaded");
        // Now load haar cascade classifier
        try{
            // define input stream
            InputStream is=context.getResources().openRawResource(R.raw.haarcascade_frontalface_alt);
            // define folder path
            File cascadeDir=context.getDir("cascade",Context.MODE_PRIVATE);
            File mCascadeFile=new File(cascadeDir,"haarcascade_frontalface_alt.xml");
            // define output stream
            FileOutputStream os=new FileOutputStream(mCascadeFile);

            // copy classifier to that folder
            byte[] buffer =new byte[4096];
            int byteRead;
            while ((byteRead=is.read(buffer)) !=-1){
                os.write(buffer,0,byteRead);
            }
            // close input and output stream
            is.close();
            os.close();
            // define cascade classifier
            cascadeClassifier=new CascadeClassifier(mCascadeFile.getAbsolutePath());

            Log.d("FacialDetector","Classifier is loaded");


            // Before watching this video please watch my previous video :
            //Facial Landmark Detection Android App Using TFLite(GPU) and OpenCV: Load CNN Model Part 2
            // You will end up with this code

            // In this video, we will do two things:
            // 1. Detect face on frame
            // 2. Pass cropped face to Interpreter which will give x, y co-ordinate of 15 keypoints on face
            // Let's start

        }
        catch (IOException e){
            e.printStackTrace();
        }

    }

    // Creata a new function input as Mat and output is also Mat format
    public Mat recognizeImage(Mat mat_image){
        // mat_image is not properly align it is 90 degree off
        // rotate mat_image by 90 degree
        Core.flip(mat_image.t(),mat_image,1);

        // do all process here
        // face detection
        // Convert mat_image to grayscale image
        Mat grayscaleImage=new Mat();
        Imgproc.cvtColor(mat_image,grayscaleImage,Imgproc.COLOR_RGBA2GRAY);
        // define height, width of grayscaleImage
        int height =grayscaleImage.height();
        int width=grayscaleImage.width();

        // define minimum height of face in original frame below this height no face will detected
        int absoluteFaceSize=(int) (height*0.1); // you can change this number to get better result

        // check if cascadeClassifier is loaded or not
        // define MatOfRect of faces
        MatOfRect faces=new MatOfRect();

        if(cascadeClassifier !=null){
            // detect face                        input       output
            cascadeClassifier.detectMultiScale(grayscaleImage,faces,1.1,2,2,
                    new Size(absoluteFaceSize,absoluteFaceSize),new Size());
                //      minimum size
        }

        // create faceArray
        Rect[] faceArray=faces.toArray();
        // loop through each face in faceArray

        for(int i=0;i<faceArray.length;i++){
            // if you want to draw face on frame
            //                image      // starting point  ending point        green Color         thickness
            Imgproc.rectangle(mat_image,faceArray[i].tl(),faceArray[i].br(),new Scalar(0,255,0,255),2);
            // Crop face from mat_imag
            //                  starting x coordinate    y-coordinate            width of face
            Rect roi = new Rect((int)faceArray[i].tl().x, (int)faceArray[i].tl().y, (int)(faceArray[i].br().x-(int)faceArray[i].tl().x), (int)(faceArray[i].br().y-(int)faceArray[i].tl().y));
            // this was important for face cropping so check
            // cropped grayscale image
            Mat cropped=new Mat(grayscaleImage,roi);
            // cropped rgba image
            Mat cropped_rgba=new Mat(mat_image,roi);

            // now convert cropped gray scale face image to bitmap
            Bitmap bitmap=null;
            bitmap=Bitmap.createBitmap(cropped.cols(),cropped.rows(),Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(cropped,bitmap);
            // define height and width of cropped bitmap
            int c_height=bitmap.getHeight();
            int c_width=bitmap.getWidth();
            // now convert cropped grayscale bitmap to buffer byte
            // before that scale it to (96,96)
            // input size of interpreter is 96
            Bitmap scaledBitmap=Bitmap.createScaledBitmap(bitmap,96,96,false);
            ByteBuffer byteBuffer=convertBitmapToByteBuffer(scaledBitmap);
            // now define output
            float[][] result=new float[1][30];// total 30 coordinate
            // predict
            interpreter.run(byteBuffer,result);

          //Before watching this video please watch Facial Landmark Detection Android App Using TFLite(GPU) and OpenCV: Predict On Frame Part 3
            // You will end up with this code
            // In this video, we will draw circle around key point

            // height,width of cropped face is different from input size of Interpreter
            // we have to scale each key point co-ordinate for cropped face
            float x_scale=((float)c_width)/((float)INPUT_SIZE);
            float y_scale=((float)c_height)/((float)INPUT_SIZE); // or you can divide it with INPUT_SIZE

            // loop through each key point
            for (int j=0;j<30;j=j+2){
                // now define x,y co-ordinate
               // every even value is x co-ordinate
                // every odd value is y co-ordinate
                float x_val=(float)Array.get(Array.get(result,0),j);
                float y_val=(float)Array.get(Array.get(result,0),j+1);

                // draw circle around x,y
                // draw on cropped_rgb not on cropped
                //              input/output     center                                  radius        color                fill circle
                Imgproc.circle(cropped_rgba,new Point(x_val*x_scale,y_val*y_scale),3,new Scalar(0,255,0,255),-1);

            }
            // replace cropped_rgba with original face on mat_image
            cropped_rgba.copyTo(new Mat(mat_image,roi));
            // select device and run
            // If you want me to train on more key point or increase accuracy of model please comment below
            // Thank you for watching this series of tutorial
            //If you gets any error comment below
            // whole code  github link will be in the description
            
        }


        // but returned mat_image should be same as passing mat
        // rotate back it -90 degree
        Core.flip(mat_image.t(),mat_image,0);
        return mat_image;
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap scaledBitmap) {
        ByteBuffer byteBuffer;
        int inputSize=INPUT_SIZE;// 96
        byteBuffer=ByteBuffer.allocateDirect(4*1*inputSize*inputSize);
        byteBuffer.order(ByteOrder.nativeOrder());
        int pixel=0;
        int [] intValues=new int [inputSize*inputSize];
        scaledBitmap.getPixels(intValues,0,scaledBitmap.getWidth(),0,0,scaledBitmap.getWidth(),scaledBitmap.getHeight());

        for (int i=0;i<inputSize;++i){
            for(int j=0;j<inputSize;++j){
                final int val= intValues[pixel++];
                byteBuffer.putFloat((float) val/255.0f);// scaling it from 0-255 to 0-1

            }
        }
        return  byteBuffer;

    }
    // now call this function in CameraActivity

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException{
        // description of file
        AssetFileDescriptor assetFileDescriptor=assetManager.openFd(modelPath);
        FileInputStream inputStream=new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startOffset=assetFileDescriptor.getStartOffset();
        long declaredLength=assetFileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
    }

}
