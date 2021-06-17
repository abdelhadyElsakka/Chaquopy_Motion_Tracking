package com.example.graduation_project_mobile_app;


import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.jjoe64.graphview.GraphView;
import com.jjoe64.graphview.series.DataPoint;
import com.jjoe64.graphview.series.DataPointInterface;
import com.jjoe64.graphview.series.LineGraphSeries;
import com.jjoe64.graphview.series.OnDataPointTapListener;
import com.jjoe64.graphview.series.Series;

public class Upload extends AppCompatActivity {

    public Button toggle;
    public EditText cutoff;
    public Spinner spinner;
    public String[] list = {"Displacement", "Velocity", "Acceleration"};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_upload);

        // ------ Definitions ------ //

        toggle = (Button) findViewById(R.id.toggle);
        cutoff = (EditText) findViewById(R.id.cutoffFrequency);
        spinner = (Spinner) findViewById(R.id.graph_spinner);
        ArrayAdapter<CharSequence> adapter = ArrayAdapter.createFromResource(this,
                R.array.graph_spinner, R.layout.spinner_item);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(adapter);
        GraphView graph = (GraphView) findViewById(R.id.graph);
        LineGraphSeries<DataPoint> series_d = new LineGraphSeries<DataPoint>(new DataPoint[]{
                new DataPoint(0, 1),
                new DataPoint(1, 5),
                new DataPoint(2, 3),
                new DataPoint(3, 2),
                new DataPoint(4, 6)
        });
        series_d.setDrawDataPoints(true);

        LineGraphSeries<DataPoint> series_v = new LineGraphSeries<DataPoint>(new DataPoint[]{
                new DataPoint(0, 1),
                new DataPoint(1, 5),
                new DataPoint(2, 3),
                new DataPoint(3, 2),
                new DataPoint(4, 6)
        });
        series_v.setDrawDataPoints(true);

        LineGraphSeries<DataPoint> series_a = new LineGraphSeries<DataPoint>(new DataPoint[]{
                new DataPoint(0, 1),
                new DataPoint(1, 5),
                new DataPoint(2, 3),
                new DataPoint(3, 2),
                new DataPoint(4, 6)
        });
        series_a.setDrawDataPoints(true);

        // ------ Event Listeners ------ //

        series_d.setOnDataPointTapListener(new OnDataPointTapListener() {
            @Override
            public void onTap(Series series, DataPointInterface dataPoint) {
                Log.d("datapoint", "datapoint: " + dataPoint.toString());
                Toast.makeText(Upload.this, "X = " + dataPoint.getX() + " Y = " + dataPoint.getY(), Toast.LENGTH_SHORT).show();
            }
        });

        series_v.setOnDataPointTapListener(new OnDataPointTapListener() {
            @Override
            public void onTap(Series series, DataPointInterface dataPoint) {
                Log.d("datapoint", "datapoint: " + dataPoint.toString());
                Toast.makeText(Upload.this, "X = " + dataPoint.getX() + " Y = " + dataPoint.getY(), Toast.LENGTH_SHORT).show();
            }
        });

        series_a.setOnDataPointTapListener(new OnDataPointTapListener() {
            @Override
            public void onTap(Series series, DataPointInterface dataPoint) {
                Log.d("datapoint", "datapoint: " + dataPoint.toString());
                Toast.makeText(Upload.this, "X = " + dataPoint.getX() + " Y = " + dataPoint.getY(), Toast.LENGTH_SHORT).show();
            }
        });

        toggle.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (toggle.getText() == "Frequency domain") {
                    toggle.setText("Time domain");
                    cutoff.setVisibility(View.VISIBLE);
                } else {
                    toggle.setText("Frequency domain");
                    cutoff.setVisibility(View.INVISIBLE);
                }
            }
        });

        // ------ Graph Settings ------ //

        graph.getGridLabelRenderer().setGridColor(Color.WHITE);
        graph.getGridLabelRenderer().setVerticalLabelsColor(Color.WHITE);
        graph.getGridLabelRenderer().setHorizontalLabelsColor(Color.WHITE);
        graph.getViewport().setScrollable(true);
        graph.getViewport().setScalable(true);
        graph.addSeries(series_d);
        graph.getGridLabelRenderer().reloadStyles();

        // ------ Spinner Settings ------ //

        spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                graph.removeSeries(series_d);
                if (list[position] == "Velocity") {
                    graph.removeSeries(series_a);
                    graph.addSeries(series_v);
                } else if (list[position] == "Acceleration") {
                    graph.removeSeries(series_v);
                    graph.addSeries(series_a);

                } else {
                    graph.removeSeries(series_a);
                    graph.removeSeries(series_v);
                    graph.addSeries(series_d);

                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });

//      TODO: Set onCharge for cutoff frequency to change the graph
//      TODO: get graphview by id
//      TODO: pass data from opencv to graphview
    }
}


