package com.example.graduation_project_mobile_app;

import androidx.appcompat.app.AppCompatActivity;

import android.net.Uri;
import android.os.Bundle;
import android.widget.Button;
import android.content.Intent;
import android.view.View;

public class How extends AppCompatActivity {

    public Button getAruco;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_how);
        getAruco = (Button) findViewById(R.id.getAruco);
        getAruco.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse("https://tn1ck.github.io/aruco-print/"));
                startActivity(intent);
            }
        });

    }

}