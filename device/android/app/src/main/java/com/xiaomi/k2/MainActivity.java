package com.xiaomi.k2;

import static java.lang.Math.abs;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Process;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {
  private final int MY_PERMISSIONS_RECORD_AUDIO = 1;
  private static final String LOG_TAG = "k2";
  private static final int SAMPLE_RATE = 16000; // The sampling rate

  private boolean startRecord = false;
  private AudioRecord record = null;

  public static String assetFilePath(Context context, String assetName) {
    File file = new File(context.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (InputStream is = context.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    } catch (IOException e) {
      Log.e(LOG_TAG, "Error process asset " + assetName + " to file path");
    }
    return null;
  }

  @Override
  public void onRequestPermissionsResult(
      int requestCode, String[] permissions, int[] grantResults) {
    if (requestCode == MY_PERMISSIONS_RECORD_AUDIO) {
      if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
        Log.i(LOG_TAG, "record permission is granted");
        initRecoder();
      } else {
        Toast.makeText(this, "Permissions denied to record audio", Toast.LENGTH_LONG).show();
        Button button = findViewById(R.id.button);
        button.setEnabled(false);
      }
    }
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    requestAudioPermissions();

    InitRecognizer();

    Button button = findViewById(R.id.button);
    button.setText("Start Record");
    button.setEnabled(false);

    button.setOnClickListener(
        view -> {
          if (!startRecord) {
            startRecord = true;
            startRecordThread();
            startAsrThread();
            button.setText("Stop Record");
          } else {
            startRecord = false;
            button.setText("Start Record");
          }
          button.setEnabled(false);
        });
  }

  private void InitRecognizer() {
    new Thread(
            () -> {
              final String modelPath = new File(assetFilePath(this, "jit.pt")).getAbsolutePath();
              final String tokenPath =
                  new File(assetFilePath(this, "tokens.txt")).getAbsolutePath();
              Recognizer.init(modelPath, tokenPath);

              runOnUiThread(
                  () -> {
                    Button button = findViewById(R.id.button);
                    button.setEnabled(true);
                  });
            })
        .start();
  }

  private void requestAudioPermissions() {
    if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
        != PackageManager.PERMISSION_GRANTED) {
      ActivityCompat.requestPermissions(
          this, new String[] {Manifest.permission.RECORD_AUDIO}, MY_PERMISSIONS_RECORD_AUDIO);
    } else {
      initRecoder();
    }
  }

  private void initRecoder() {
    // buffer size in bytes 1280 (40ms)
    int miniBufferSize =
        AudioRecord.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
    if (miniBufferSize == AudioRecord.ERROR || miniBufferSize == AudioRecord.ERROR_BAD_VALUE) {
      Log.e(LOG_TAG, "Audio buffer can't initialize!");
      return;
    }
    // Set the buffer size twice as much as the miniBufferSize, in case the
    // decoder can not consume data in time.
    record =
        new AudioRecord(
            MediaRecorder.AudioSource.DEFAULT,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            miniBufferSize * 2);
    if (record.getState() != AudioRecord.STATE_INITIALIZED) {
      Log.e(LOG_TAG, "Audio Record can't initialize!");
      return;
    }
    Log.i(LOG_TAG, "Record init okay");
  }

  private void startRecordThread() {
    new Thread(
            () -> {
              Recognizer.initDecodeStream();
              WaveView waveView = findViewById(R.id.waveView);
              record.startRecording();
              Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO);
              while (startRecord) {
                int bufferSize = 160; // 10ms
                short[] buffer = new short[bufferSize];
                int status = record.read(buffer, 0, buffer.length);
                if (status >= 0) {
                  float[] data = new float[bufferSize];
                  int index = 0;
                  for (short value : buffer) {
                    data[index++] = value * 1.0f / 32768;
                  }
                  Recognizer.acceptWaveform(data);
                  waveView.addData(calculateScale(buffer));
                }
                Button button = findViewById(R.id.button);
                if (!button.isEnabled() && startRecord) {
                  runOnUiThread(() -> button.setEnabled(true));
                }
              }
              Recognizer.inputFinished();
              record.stop();
              waveView.resetView();
            })
        .start();
  }

  private double calculateScale(short[] buffer) {
    int amplitude = 0;
    for (short value : buffer) {
      amplitude += abs(value);
    }
    double scale = amplitude * 1.0 / buffer.length / 32768 * 5;
    scale = Math.min(scale, 1.0);
    return scale;
  }

  private void startAsrThread() {
    new Thread(
            () -> {
              while (startRecord) {
                // Decode one chunk and get partial result
                String result = Recognizer.decode();
                runOnUiThread(
                    () -> {
                      TextView textView = findViewById(R.id.textView);
                      textView.setText(result);
                    });
              }

              // Wait for final result
              while (true) {
                if (!Recognizer.isFinished()) {
                  String result = Recognizer.decode();
                  runOnUiThread(
                      () -> {
                        TextView textView = findViewById(R.id.textView);
                        textView.setText(result);
                      });
                } else {
                  runOnUiThread(
                      () -> {
                        Button button = findViewById(R.id.button);
                        button.setEnabled(true);
                      });
                  break;
                }
              }
            })
        .start();
  }
}
