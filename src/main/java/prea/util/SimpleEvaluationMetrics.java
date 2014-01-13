package prea.util;
import prea.data.structure.SparseMatrix;
import prea.data.structure.SparseVector;

/**
 * A simplified version of evaluation metrics class.
 * It deals only with MAE and RMSE, and ignores unexisting items.
 * 
 * @author Joonseok Lee
 * @since 2012. 5. 16
 * @version 1.1
 */
public class SimpleEvaluationMetrics {
	/** Real ratings for test items. */
	private SparseMatrix testMatrix;
	/** Predicted ratings by CF algorithms for test items. */
	private SparseMatrix predicted;
	/** Maximum value of rating, existing in the dataset. */
	private double maxValue;
	/** Minimum value of rating, existing in the dataset. */
	private double minValue;
 

    /** Mean Absoulte Error (MAE) */
    private double mae;
    /** Mean Squared Error (MSE) */
    private double mse;
    /** Asymmetric Loss */
    private double asymmetricLoss;
	
	/**
	 * Standard constructor for EvaluationMetrics class.
	 * 
	 * @param tm Real ratings of test items.
	 * @param p Predicted ratings of test items.
	 * @param max Maximum value of rating, existing in the dataset.
	 * @param min Minimum value of rating, existing in the dataset.
	 *
	 */
	public SimpleEvaluationMetrics(SparseMatrix tm, SparseMatrix p, double max, double min) {
		testMatrix = tm;
		predicted = p;
		maxValue = max;
		minValue = min;
		
		build();
	}
	
	public SparseMatrix getPrediction() {
		return predicted;
	}
	
	/**
	 * Getter method for Mean Absolute Error (MAE)
	 * 
	 * @return Mean Absolute Error (MAE)
	 */
	public double getMAE() {
		return mae;
	}
	
	/**
	 * Getter method for Normalized Mean Absolute Error (NMAE)
	 * 
	 * @return Normalized Mean Absolute Error (NMAE)
	 */
	public double getNMAE() {
		return mae / (maxValue - minValue);
	}
	
	/**
	 * Getter method for Mean Squared Error (MSE)
	 * 
	 * @return Mean Squared Error (MSE)
	 */
	public double getMSE() {
		return mse;
	}
	
	/**
	 * Getter method for Root of Mean Squared Error (RMSE)
	 * 
	 * @return Root of Mean Squared Error (RMSE)
	 */
	public double getRMSE() {
		return Math.sqrt(mse);
	}

	/**
	 * Getter method for Asymmetric loss
	 * 
	 * @return Asymmetric loss
	 */
	public double getAsymmetricLoss() {
		return asymmetricLoss;
	}
		
	/** Calculate all evaluation metrics with given real and predicted rating matrices. */
	private void build() {
		int userCount = (testMatrix.length())[0] - 1;
		int testItemCount = 0;
		
		for (int u = 1; u <= userCount; u++) {
			SparseVector predictedRateList = predicted.getRowRef(u);
			
			if (predictedRateList.itemCount() > 0) {
				int[] predictedRateIndex = predictedRateList.indexList();
				
				for (int i : predictedRateIndex) {
					double realRate = testMatrix.getValue(u, i);
					double predictedRate = predicted.getValue(u, i);
					
					// Ignore the item without prediction
					if (predictedRate != 0.0 && realRate != 0.0) {
						// Accuracy calculation:
						mae += Math.abs(realRate - predictedRate);
						mse += Math.pow(realRate - predictedRate, 2);
						asymmetricLoss += Loss.asymmetricLoss(realRate, predictedRate, minValue, maxValue);
						testItemCount++;
					}
				}
			}
		}
		
		mae /= (double) testItemCount;
		mse /= (double) testItemCount;
		asymmetricLoss /= (double) testItemCount;
	}

	
	public String printMultiLine() {
		return	"MAE\t" + this.getMAE() + "\r\n" +
				"RMSE\t" + this.getRMSE() + "\r\n" +
				"Asymm\t" + this.getAsymmetricLoss() + "\r\n" +
				"HLU\tN/A\r\n" +
				"NDCG\tN/A\r\n" +
				"Kendall\tN/A\r\n" +
				"Spear\tN/A";
	}
	
	public String printOneLine() {
		return String.format("%.4f\t%.4f\t%.4f\tN/A\tN/A\tN/A\tN/A",
				this.getMAE(), this.getRMSE(), this.getAsymmetricLoss());
	}
	
	public static String printTitle() {
		return "==============================================================================================\r\nName\tMAE\tRMSE\tAsymm\tHLU\tNDCG\tKendall\tSpear";
	}
}