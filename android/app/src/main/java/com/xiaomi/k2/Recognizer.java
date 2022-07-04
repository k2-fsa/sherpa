package com.xiaomi.k2;

public class Recognizer {
  static {
    System.loadLibrary("k2_device");
  }

  public static native void init(String modelPath, String bpePath);

  public static native void initDecodeStream();

  public static native void acceptWaveform(float[] waveform);

  public static native String decode();

  public static native void inputFinished();

  public static native boolean isFinished();
}
