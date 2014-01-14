package prea.util;


public class IREvaluator
{
  private int[] testTable = { 0, 0, 0, 0 };
  private int[] predictValueCount = { 0, 0 };
  private int[] actualValueCount = { 0, 0 };
  
  public void addInstance(boolean predictValue, boolean actualValue)
  {
    if (predictValue)
      this.predictValueCount[1]++;
    else
      this.predictValueCount[0]++;
    if (actualValue)
      this.actualValueCount[1]++;
    else
      this.actualValueCount[0]++;
    
    if (predictValue && actualValue)
      this.testTable[0]++;
    if (predictValue && !actualValue)
      this.testTable[1]++;
    if (!predictValue && actualValue)
      this.testTable[2]++;
    if (!predictValue && !actualValue)
      this.testTable[3]++;
  }
  
  public double getPrecision()
  {
    return (float) (1.0f * testTable[0] / (testTable[0] + testTable[1]));
  }

  public double getRecall()
  {
    return (float) (1.0f * testTable[0] / (testTable[0] + testTable[2]));
  }

  public double getF1Measure()
  {
    return getFNMeasure(1.0);
  }
  
  public double getFNMeasure(double b) 
  {
    double b2 = b * b;
    double sum = b2 * getPrecision() + getRecall();
    return sum == 0.0 ? Double.NaN : (1.0 + b2) * getPrecision() * getRecall() / sum;
  }
  
  public String getHeader()
  {
    return "Precision\tRecall\tF1";
  }
  
  public String getResult()
  {
    return getPrecision() + "\t" + getRecall() + "\t" + getF1Measure();
  }
  
}
