package com.xiaomi.k2;

import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;
import androidx.core.content.ContextCompat;
import java.util.Arrays;

public class WaveView extends View {
  private Paint mPaint;
  private int mBarNum;
  private int mBarWidth;
  private int mHeight;
  private int mBarOffset;
  private int mDelayTime;
  private double[] mValue = null;

  private int mCurrentBarIndex;

  public WaveView(Context context) {
    this(context, null);
  }

  public WaveView(Context context, AttributeSet attrs) {
    this(context, attrs, 0);
  }

  public WaveView(Context context, AttributeSet attrs, int defStyleAttr) {
    super(context, attrs, defStyleAttr);
    Init(context, attrs);
  }

  public void Init(Context context, AttributeSet attrs) {
    TypedArray ta = context.obtainStyledAttributes(attrs, R.styleable.WaveView);
    mPaint = new Paint();
    mPaint.setColor(
        ta.getColor(
            R.styleable.WaveView_BarColor, ContextCompat.getColor(context, R.color.bar_color)));
    mBarNum = ta.getInt(R.styleable.WaveView_BarNum, 10);
    mValue = new double[mBarNum];
    mDelayTime = ta.getInt(R.styleable.WaveView_DelayTime, 300);
    mBarOffset = ta.getInt(R.styleable.WaveView_BarOffset, 0);
    ta.recycle();
    mCurrentBarIndex = 0;
  }

  @Override
  protected void onSizeChanged(int w, int h, int oldW, int oldH) {
    super.onSizeChanged(w, h, oldW, oldH);
    int width = getWidth();
    int height = getHeight();
    mBarWidth = (width - mBarOffset) / mBarNum;
    mHeight = height;
  }

  public void addData(double amplitude) {
    mValue[mCurrentBarIndex % mBarNum] = amplitude;
    mCurrentBarIndex++;
  }

  public void resetView() {
    Arrays.fill(mValue, 0);
    mCurrentBarIndex = 0;
  }

  @Override
  protected void onDraw(Canvas canvas) {
    super.onDraw(canvas);
    int currentIndex = mCurrentBarIndex;
    for (int i = 0; i < mBarNum; i++) {
      float currentHeight = (float) (mHeight * mValue[(currentIndex + i) % mBarNum]);
      canvas.drawRect(
          (float) ((mBarWidth + mBarOffset) * i),
          (mHeight + currentHeight) / 2,
          (float) (mBarWidth * (i + 1) + mBarOffset * i),
          (mHeight - currentHeight) / 2,
          mPaint);
    }
    postInvalidateDelayed(mDelayTime);
  }
}
