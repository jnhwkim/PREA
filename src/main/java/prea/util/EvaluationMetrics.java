package prea.util;
import prea.data.structure.SparseMatrix;
import prea.data.structure.SparseVector;

/**
 * This is a unified class providing evaluation metrics,
 * including comparison of predicted ratings and rank-based metrics, etc.
 * 
 * @author Joonseok Lee
 * @author Mingxuan Sun
 * @since 2012. 4. 20
 * @version 1.1
 */
public class EvaluationMetrics {
	/** Real ratings for test items. */
	private SparseMatrix testMatrix;
	/** Predicted ratings by CF algorithms for test items. */
	private SparseMatrix predicted;
	/** Maximum value of rating, existing in the dataset. */
	private double maxValue;
	/** Minimum value of rating, existing in the dataset. */
	private double minValue;
	/** The number of items to recommend, in rank-based metrics */
	private int recommendCount;
	/** Half-life in rank-based metrics */
	private int halflife;
	/** Rank-based logistic loss */
	private double rankLogisticError;
	/** Rank-based log-loss I */
	private double rankLogError1;
	/** Rank-based log-loss II */
	private double rankLogError2;
	/** Rank-based absolute error */
	private double rankAbsError;
	/** Rank-based squared error */
	private double rankSqrError;
	/** Rank-based exponential regression */
	private double rankExpRegError;
	/** Rank-based smooth L1 regression */
	private double rankSmoothL1Error;
	/** Rank-based Hinge loss I */
	private double rankHingeError1;
	/** Rank-based Hinge loss II */
	private double rankHingeError2;
	/** Rank-based 0/1-loss */
	private double rankZeroOneError;
 
	  /** IREvaluator (Precision/Recall/F1Measure) */
	  private IREvaluator irEval;
    /** Mean Absoulte Error (MAE) */
    private double mae;
    /** Mean Squared Error (MSE) */
    private double mse;
    /** Rank-based Half-Life Utility (HLU) */
    private double hlu;
    /** Rank-based Normalized Discounted Cumulative Gain (NDCG) */
    private double ndcg;
    /** Rank-based Kendall's Tau */
    private double kendallsTau;
    /** Rank-based Spear */
    private double spearman;
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
	public EvaluationMetrics(SparseMatrix tm, SparseMatrix p, double max, double min) {
		testMatrix = tm;
		predicted = p;
		maxValue = max;
		minValue = min;
		recommendCount = 5;
		halflife = 5;
		
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
	 * Getter method for Rank-based Half-life score
	 * 
	 * @return Rank-based Half-life score
	 */
	public double getHLU() {
		return hlu;
	}
	/**
	 * Getter method for Rank-based NDCG
	 * 
	 * @return Rank-based NDCG score
	 */
	public double getNDCG() {
		return ndcg;
	}
	/**
	 * Getter method for Rank-based Kendall's Tau
	 * 
	 * @return Rank-based Kendall's Tau score
	 */
	public double getKendall() {
		return kendallsTau;
	}
	/**
	 * Getter method for Rank-based Spearman score
	 * 
	 * @return Rank-based Spearman score
	 */
	public double getSpearman() {
		return spearman;
	}

	/**
	 * Getter method for Asymmetric loss
	 * 
	 * @return Asymmetric loss
	 */
	public double getAsymmetricLoss() {
		return asymmetricLoss;
	}
	
	/**
	 * Getter method for Rank-based Log-Loss
	 * 
	 * @return Log-loss
	 */
	public double getRankLoss(int errorCode) {
		double rankError = 0.0;
		
		switch (errorCode) {
		case RankEvaluator.LOGISTIC_LOSS:
			rankError = rankLogisticError;
			break;
		case RankEvaluator.LOG_LOSS_1:
			rankError = rankLogError1;
			break;
		case RankEvaluator.LOG_LOSS_2:
			rankError = rankLogError2;
			break;
		case RankEvaluator.ABSOLUTE_LOSS:
			rankError = rankAbsError;
			break;
		case RankEvaluator.SQUARED_LOSS:
			rankError = rankSqrError;
			break;
		case RankEvaluator.EXP_REGRESSION:
			rankError = rankExpRegError;
			break;
		case RankEvaluator.SMOOTH_L1_REGRESSION:
			rankError = rankSmoothL1Error;
			break;
		case RankEvaluator.HINGE_LOSS_1:
			rankError = rankHingeError1;
			break;
		case RankEvaluator.HINGE_LOSS_2:
			rankError = rankHingeError2;
			break;
		}
		
		return rankError / (double) ((testMatrix.length())[0] - 1);
	}
	
	/**
	 * Getter method for Rank-based 0/1-Loss
	 * 
	 * @return 0/1-loss
	 */
	public double getZeroOneLoss() {
		return rankZeroOneError / (double) ((testMatrix.length())[0] - 1);
	}
		
	/** Calculate all evaluation metrics with given real and predicted rating matrices. */
	private void build() {
		int userCount = (testMatrix.length())[0] - 1;
		int testUserCount = 0;
		int testItemCount = 0;
		double rScoreSum = 0.0;
		double rMaxSum = 0;
		irEval = new IREvaluator();
		
		for (int u = 1; u <= userCount; u++) {
			testUserCount++;
			
			SparseVector realRateList = testMatrix.getRowRef(u);
			SparseVector predictedRateList = predicted.getRowRef(u);
			
			if (realRateList.itemCount() != predictedRateList.itemCount()) {
				System.out.println("Error: The number of test items and predicted items does not match! (" + 
				    realRateList.itemCount() + "/" + predictedRateList.itemCount() + ")");
				continue;
			}
			
			if (realRateList.itemCount() > 0) {
			  int[] realRateIndex = realRateList.indexList();
				double[] realRateValue = realRateList.valueList();
				int[] predictedRateIndex = predictedRateList.indexList();
				double[] predictedRateValue = predictedRateList.valueList();

				// k-largest rating value arrays are sorted here:
			  Sort.kLargest(predictedRateValue, predictedRateIndex, 0, predictedRateIndex.length-1, recommendCount);
			  Sort.kLargest(realRateValue, realRateIndex, 0, predictedRateIndex.length-1, recommendCount);
			  
				// Top-n
				realRateList = new SparseVector(realRateList.length());
				predictedRateList = new SparseVector(predictedRateList.length());
				
				for (int i = 0; i < recommendCount; i++) {
				  if (i < predictedRateIndex.length && 
				      i < realRateIndex.length) {
				    realRateList.setValue(realRateIndex[i], realRateValue[i]);
				    predictedRateList.setValue(predictedRateIndex[i], predictedRateValue[i]);
				  }
				}
				
				// re-get the lists
				realRateIndex = realRateList.indexList();
				realRateValue = realRateList.valueList();
				predictedRateIndex = predictedRateList.indexList();
				predictedRateValue = predictedRateList.valueList();
        
        // k-largest rating value arrays are sorted here:
        Sort.kLargest(predictedRateValue, predictedRateIndex, 0, predictedRateIndex.length-1, recommendCount);
        Sort.kLargest(realRateValue, realRateIndex, 0, predictedRateIndex.length-1, recommendCount);

				int r = 1;
				double rScore = 0.0;
				for (int i : predictedRateIndex) {
					double realRate = testMatrix.getValue(u, i);
					double predictedRate = predicted.getValue(u, i);
					
					// Accuracy calculation:
					mae += Math.abs(realRate - predictedRate);
					mse += Math.pow(realRate - predictedRate, 2);
					asymmetricLoss += Loss.asymmetricLoss(realRate, predictedRate, minValue, maxValue);
					testItemCount++;
					
					// Precision / Recall / F1Measure
					boolean predictValue = predictedRate > 3 ? true : false;
					boolean actualValue = realRate > 3 ? true : false;
					irEval.addInstance(predictValue, actualValue);
					
					// Half-life evaluation:
					if (r <= recommendCount) {
						rScore += Math.max(realRate - (double) (maxValue + minValue) / 2.0, 0.0) 
									/ Math.pow(2.0, (double) (r-1) / (double) (halflife-1));
						
						r++;
					}
				}
				
				// Rank-based metrics
				double rankLogisticLoss = 0.0;
				double rankLogLoss1 = 0.0;
				double rankLogLoss2 = 0.0;
				double rankAbsLoss = 0.0;
				double rankSqrLoss = 0.0;
				double rankExpRegLoss = 0.0;
				double rankSmoothL1Loss = 0.0;
				double rankHingeLoss1 = 0.0;
				double rankHingeLoss2 = 0.0;
				double zeroOneLoss = 0.0;
				int pairCount = 0;
				for (int i : predictedRateIndex) {
					for (int j : predictedRateIndex) {
						double realRate_i = testMatrix.getValue(u, i);
						double predictedRate_i = predicted.getValue(u, i);
						double realRate_j = testMatrix.getValue(u, j);
						double predictedRate_j = predicted.getValue(u, j);
						
						if (realRate_i > realRate_j) {
							rankLogisticLoss += 1 / (1 + Math.exp((realRate_i - realRate_j) * (predictedRate_i - predictedRate_j)));
							rankLogLoss1 += (realRate_i - realRate_j) * Math.log(1 + Math.exp(predictedRate_j - predictedRate_i));
							rankLogLoss2 += Math.log(1 + Math.exp(realRate_i - realRate_j - predictedRate_i + predictedRate_j));
							rankAbsLoss += Math.abs(realRate_i - realRate_j - predictedRate_i + predictedRate_j);
							rankSqrLoss += Math.pow(realRate_i - realRate_j - predictedRate_i + predictedRate_j, 2);
							rankExpRegLoss += Math.exp(realRate_i - realRate_j - predictedRate_i + predictedRate_j)
											+ Math.exp(predictedRate_i - predictedRate_j - realRate_i + realRate_j);
							rankSmoothL1Loss += Math.log(1 + Math.exp(realRate_i - realRate_j - predictedRate_i + predictedRate_j))
											  + Math.log(1 + Math.exp(predictedRate_i - predictedRate_j - realRate_i + realRate_j));
							rankHingeLoss1 += Math.max(realRate_i - realRate_j - predictedRate_i + predictedRate_j, 0);
							rankHingeLoss2 += (realRate_i - realRate_j) * Math.max(1 - predictedRate_i + predictedRate_j, 0);
							zeroOneLoss += (realRate_i - realRate_j) * (predictedRate_i - predictedRate_j) < 0 ? 1 : 0;
							pairCount++;
						}
					}
				}
				
				if (pairCount > 0) {
					rankLogisticLoss = rankLogisticLoss / (double) pairCount;
					rankLogLoss1 = rankLogLoss1 / (double) pairCount;
					rankLogLoss2 = rankLogLoss2 / (double) pairCount;
					rankAbsLoss = rankAbsLoss / (double) pairCount;
					rankSqrLoss = rankSqrLoss / (double) pairCount;
					rankExpRegLoss = rankExpRegLoss / (double) pairCount;
					rankSmoothL1Loss = rankSmoothL1Loss / (double) pairCount;
					rankHingeLoss1 = rankHingeLoss1 / (double) pairCount;
					rankHingeLoss2 = rankHingeLoss2 / (double) pairCount;
					zeroOneLoss = zeroOneLoss / (double) pairCount;
				}
				
				rankLogisticError += rankLogisticLoss;
				rankLogError1 += rankLogLoss1;
				rankLogError2 += rankLogLoss2;
				rankAbsError += rankAbsLoss;
				rankSqrError += rankSqrLoss;
				rankExpRegError += rankExpRegLoss;
				rankSmoothL1Error += rankSmoothL1Loss;
				rankHingeError1 += rankHingeLoss1;
				rankHingeError2 += rankHingeLoss2;
				rankZeroOneError += zeroOneLoss;

				// calculate R_Max here, and divide rScore by it.
				int rr = 1;
				double rMax = 0.0;
				for (int i : realRateIndex) {
					if (rr < r) {
						double realRate = testMatrix.getValue(u, i);
						rMax += Math.max(realRate - (double) (maxValue + minValue) / 2.0, 0.0) 
								/ Math.pow(2.0, (double) (rr-1) / (double) (halflife-1));
						
						rr++;
					}
				}
				
				rScoreSum += rScore * Math.min(realRateIndex.length, recommendCount);
				rMaxSum += rMax * Math.min(realRateIndex.length, recommendCount);
				
				// Rank-based metrics:
				ndcg += Distance.distanceNDCG(realRateList.indexList(), realRateList.valueList(), predictedRateList.indexList(), predictedRateList.valueList());
				kendallsTau += Distance.distanceKendall(realRateList.indexList(), realRateList.valueList(), predictedRateList.indexList(), predictedRateList.valueList(), realRateList.itemCount());
				spearman += Distance.distanceSpearman(realRateList.indexList(), realRateList.valueList(), predictedRateList.indexList(), predictedRateList.valueList(), realRateList.itemCount());
			}
		}
		
		mae /= (double) testItemCount;
		mse /= (double) testItemCount;
		hlu = rScoreSum / rMaxSum;
		ndcg /= (double) testUserCount;
		kendallsTau /= (double) testUserCount;
		spearman /= (double) testUserCount;
		asymmetricLoss /= (double) testItemCount;
	}

	
	public String printMultiLine() {
		return	"MAE\t" + this.getMAE() + "\r\n" +
				"RMSE\t" + this.getRMSE() + "\r\n" +
				"Asymm\t" + this.getAsymmetricLoss() + "\r\n" +
				"HLU\t" + this.getHLU() + "\r\n" +
				"NDCG\t" + this.getNDCG() + "\r\n" +
				"Kendall\t" + this.getKendall() + "\r\n" +
				"Spear\t" + this.getSpearman() + "\r\n";
	}
	
	public String printOneLine() {
		return String.format("%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f",
		    this.irEval.getPrecision(),
        this.irEval.getRecall(),
        this.irEval.getF1Measure(),
				this.getMAE(),
				this.getRMSE(),
				this.getAsymmetricLoss(),
				this.getHLU(),
				this.getNDCG(),
				this.getKendall(),
				this.getSpearman());
	}
	
	public String printRankErrors() {
		return String.format("%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f",
				this.getRankLoss(RankEvaluator.LOGISTIC_LOSS),
				this.getRankLoss(RankEvaluator.LOG_LOSS_1),
				this.getRankLoss(RankEvaluator.LOG_LOSS_2),
				this.getRankLoss(RankEvaluator.ABSOLUTE_LOSS),
				this.getRankLoss(RankEvaluator.SQUARED_LOSS),
				this.getRankLoss(RankEvaluator.EXP_REGRESSION),
				this.getRankLoss(RankEvaluator.SMOOTH_L1_REGRESSION),
				this.getRankLoss(RankEvaluator.HINGE_LOSS_1),
				this.getRankLoss(RankEvaluator.HINGE_LOSS_2),
				this.getZeroOneLoss());
	}
	
	public static String printTitle() {
		return "=============================================================================================================\r\nName\tPreci.\tRecall\tF1 \tMAE\tRMSE\tAsymm\tHLU\tNDCG\tKendall\tSpear";
	}
	
	public static String printTitleWithLongName() {
    return "=============================================================================================================\r\nName\t\tPreci.\tRecall\tF1 \tMAE\tRMSE\tAsymm\tHLU\tNDCG\tKendall\tSpear";
  }
	
	public static String printRankTitle() {
		return "===================================================================================================================\r\nName\tLogs\tLog1\tLog2\tAbs\tSqr\tExpReg\tSmL1\tHinge1\tHinge2\t0/1";
	}
}