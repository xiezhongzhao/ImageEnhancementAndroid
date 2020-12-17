package com.example.imageenhancementandroid;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'native-lib' library on application startup.
    static {
        if(!OpenCVLoader.initDebug()){
            Log.d("opencv", "初始化失败");
        }else{
            Log.d("opencv", "初始化成功");
        }
        System.loadLibrary("native-lib");
    }

    private double maxSize = 1024;
    private int PICK_IMAGE_REQUEST = 1;
    private ImageView myImageView;
    private TextView textTimeView;
    private Bitmap selectbp;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Example of a call to a native method
        //TextView tv = findViewById(R.id.sample_text);
        //tv.setText(stringFromJNI());

        myImageView = (ImageView)findViewById(R.id.imageView);
        myImageView.setScaleType(ImageView.ScaleType.FIT_CENTER);
        Button selectImageBtn = (Button)findViewById(R.id.select_btn);

        selectImageBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                selectImage();
            }

            private void selectImage(){
                Intent intent = new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(Intent.createChooser(intent, "选择图像"), PICK_IMAGE_REQUEST);
            }
        });

        Button grayBtn = (Button)findViewById(R.id.gray_btn);
        grayBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                convertGray();
            }
        });

        Button seceBtn = (Button)findViewById(R.id.sece_btn);
        seceBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                seceEnhance();
            }
        });

        Button baiduBtn = (Button)findViewById(R.id.baidu_btn);
        baiduBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                baiduEnhance();
            }
        });

        textTimeView = (TextView)findViewById(R.id.time_label);
//        textTime.setText(R.string.time);
    }

    private void costTime(long startTime, long endTime){
        float usedTime = (float) ((endTime-startTime));
        String str = Float.toString(usedTime);
        textTimeView.setText("The algorithm costs: " + str + " ms");
    }

    private void convertGray(){

        Mat src = new Mat();
        Mat temp = new Mat();
        Mat dst = new Mat();

        Utils.bitmapToMat(selectbp, src);
        Imgproc.cvtColor(src,temp,Imgproc.COLOR_BGRA2BGR);
        Log.i("CV", "image type:" + (temp.type() == CvType.CV_8UC3));

        long startTime =  System.currentTimeMillis();
        Imgproc.cvtColor(temp, dst, Imgproc.COLOR_BGR2GRAY);
        long endTime =  System.currentTimeMillis();
        costTime(startTime, endTime);

        Utils.matToBitmap(dst, selectbp);
        myImageView.setImageBitmap(selectbp);
    }

    private void seceEnhance(){

        int w = selectbp.getWidth();
        int h = selectbp.getHeight();
        Log.d("width：", String.valueOf(w));
        Log.d("height：", String.valueOf(h));

        int[] pix = new int[w * h];
        selectbp.getPixels(pix, 0, w, 0, 0, w, h);

        //调用native函数，图像对比度增强
        int[] resultInt = seceFromJNI(pix, w, h);
        String seceTime = timeFromJNI();
        textTimeView.setText("The algorithm costs: " + seceTime + " ms");

        Bitmap resultImg = Bitmap.createBitmap(w, h, Bitmap.Config.RGB_565);
        Log.d("width：", String.valueOf(resultImg.getWidth()));
        Log.d("height：", String.valueOf(resultImg.getHeight()));
        resultImg.setPixels(resultInt, 0, w, 0, 0,w, h);
        myImageView.setDrawingCacheEnabled(true);
        myImageView.setImageBitmap(resultImg);

//        //从imageview上获取bitmap图片
//        myImageView.setDrawingCacheEnabled(true);
//        Bitmap bitmap = myImageView.getDrawingCache();
//
//        int w=bitmap.getWidth();
//        int h=bitmap.getHeight();
//        int[] pix = new int[w * h];
//        bitmap.getPixels(pix, 0, w, 0, 0, w, h);
//
//        //调用native函数，模糊图像
//        int[] resultInt = seceFromJNI(pix, w, h);
//        String seceTime = timeFromJNI();
////        String seceTime = timeFromJNI();
//        textTimeView.setText("The algorithm costs: " + seceTime + " ms");
//
//        Bitmap resultImg = Bitmap.createBitmap(w, h, Bitmap.Config.RGB_565);
//        resultImg.setPixels(resultInt, 0, w, 0, 0,w, h);
//        myImageView.setDrawingCacheEnabled(false);
//
//        myImageView.setImageBitmap(resultImg);
    }

    private void baiduEnhance(){


    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK && data != null && data.getData() != null) {
            Uri uri = data.getData();
            try {
                Log.d("image-tag", "start to decode selected image now...");
                InputStream input = getContentResolver().openInputStream(uri);
                BitmapFactory.Options options = new BitmapFactory.Options();
                options.inJustDecodeBounds = true;
                BitmapFactory.decodeStream(input, null, options);
                int raw_width = options.outWidth;
                int raw_height = options.outHeight;
                int max = Math.max(raw_width, raw_height);
                int newWidth = raw_width;
                int newHeight = raw_height;
                int inSampleSize = 1;
                if(max > maxSize) {
                    newWidth = raw_width / 2;
                    newHeight = raw_height / 2;
                    while((newWidth/inSampleSize) > maxSize || (newHeight/inSampleSize) > maxSize) {
                        inSampleSize *=2;
                    }
                }

                options.inSampleSize = inSampleSize;
                options.inJustDecodeBounds = false;
                options.inPreferredConfig = Bitmap.Config.ARGB_8888;
                selectbp = BitmapFactory.decodeStream(getContentResolver().openInputStream(uri), null, options);

                myImageView.setImageBitmap(selectbp);

            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();

    public native int[] seceFromJNI(int[] rawImg, int w, int h);

    public native String timeFromJNI();
}










