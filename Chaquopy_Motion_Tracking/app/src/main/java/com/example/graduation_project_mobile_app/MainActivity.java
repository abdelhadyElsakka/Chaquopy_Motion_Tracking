package com.example.graduation_project_mobile_app;

import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.Path;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.Button;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;


public class MainActivity extends AppCompatActivity {

    public Button how;
    public Button camera;
    public Button upload;
    public int REQUEST_TAKE_GALLERY_VIDEO = 1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        how = (Button) findViewById(R.id.howTo);
        camera = (Button) findViewById(R.id.camera);
        upload = (Button) findViewById(R.id.upload);

        how.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, How.class);
                startActivity(intent);
            }
        });
        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MediaStore.ACTION_VIDEO_CAPTURE);
                startActivityForResult(intent, 1);
            }
        });
        upload.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent videoIntent = new Intent();
                videoIntent.setType("video/*");
                videoIntent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(Intent.createChooser(videoIntent, "Select Video"), REQUEST_TAKE_GALLERY_VIDEO);


            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            if (requestCode == REQUEST_TAKE_GALLERY_VIDEO) {
                String videoPath = data.getData().getPath();
                Uri selectedVideoUri = data.getData();
                String[] projection = {MediaStore.Video.Media.DATA, MediaStore.Video.Media.SIZE, MediaStore.Video.Media.DURATION};
                Cursor cursor = managedQuery(selectedVideoUri, projection, null, null, null);

                cursor.moveToFirst();
                String filePath = cursor.getString(cursor.getColumnIndexOrThrow(MediaStore.Video.Media.DATA));
                Log.d("File Name:",videoPath);

                // Setting the thumbnail of the video in to the image view
                InputStream inputStream = null;
                // Converting the video in to the bytes
                try
                {
                    inputStream = getContentResolver().openInputStream(selectedVideoUri);
                }
                catch (FileNotFoundException e)
                {
                    e.printStackTrace();
                }
                int bufferSize = 1024;
                byte[] buffer = new byte[bufferSize];
                ByteArrayOutputStream byteBuffer = new ByteArrayOutputStream();
                int len = 0;
                try
                {
                    while ((len = inputStream.read(buffer)) != -1)
                    {
                        byteBuffer.write(buffer, 0, len);
                    }
                }
                catch (IOException e)
                {
                    e.printStackTrace();
                }
                System.out.println("converted!");

                String videoData="";
                //Converting bytes into base64
                videoData = Base64.encodeToString(byteBuffer.toByteArray(), Base64.DEFAULT);
                Log.d("VideoData**>  " , videoData);

                String sinSaltoFinal2 = videoData.trim();
                String sinsinSalto2 = sinSaltoFinal2.replaceAll("\n", "");
                Log.d("VideoData**>  " , sinsinSalto2);

//                baseVideo = sinsinSalto2;


                if(!Python.isStarted())
                    Python.start(new AndroidPlatform(this));
                final Python py = Python.getInstance();
                PyObject pyo = py.getModule("myscript");
                PyObject obj = pyo.callAttr("make_video",videoData);
//                String str = obj.toString();
                Intent intent = new Intent(MainActivity.this, Upload.class);
                //TODO: putExtra (Key, value)
                startActivity(intent);
            }
        }
    }
}