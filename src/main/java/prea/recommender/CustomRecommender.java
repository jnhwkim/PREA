package prea.recommender;

import prea.data.structure.SparseMatrix;
import prea.data.structure.SparseVector;
import prea.util.EvaluationMetrics;

/**
 * This is a skeleton class for user-defined custom recommenders.
 * You may copy this class if you want to implement more than one class.
 * Please look at the "COMMENT FOR AUTHORS" in each method carefully before coding your algorithm.
 * 
 * @author Joonseok Lee
 * @since 2012. 4. 20
 * @version 1.1
 */
public class CustomRecommender implements Recommender {
	/*========================================
	 * Common Variables
	 *========================================*/
	/** Rating matrix for each user (row) and item (column) */
	public SparseMatrix rateMatrix;
	
	/* =====================
	 *  COMMENT FOR AUTHORS
	 * =====================
	 * These variables may be commonly used in most recommendation systems.
	 * You can freely add new variables if needed.
	 * Note that do not delete these since they are used in evaluation method.
	 */
	/** The number of users. */
	public int userCount;
	/** The number of items. */
	public int itemCount;
	/** Maximum value of rating, existing in the dataset. */
	public double maxValue;
	/** Minimum value of rating, existing in the dataset. */
	public double minValue;
	
	/*========================================
	 * Constructors
	 *========================================*/
	/**
	 * Construct a customized recommender model with the given data.
	 * 
	 * @param uc The number of users in the dataset.
	 * @param ic The number of items in the dataset.
	 * @param max The maximum rating value in the dataset.
	 * @param min The minimum rating value in the dataset.
	 */
	public CustomRecommender(int uc, int ic, double max, double min) {
		/* =====================
		 *  COMMENT FOR AUTHORS
		 * =====================
		 * Please make sure that all your custom member variables
		 * are correctly initialized in this stage.
		 * If you added new variables in "Common Variables" section above,
		 * you should initialize them properly here.
		 * You may add parameters as well.
		 */
		userCount = uc;
		itemCount = ic;
		maxValue = max;
		minValue = min;
	}
	
	/*========================================
	 * Model Builder
	 *========================================*/
	/** 
	 * Build a model with the given data and algorithm.
	 * 
	 * @param rm The rating matrix with train data.
	 */
	@Override
	public void buildModel(SparseMatrix rm) {
		rateMatrix = rm;
		
		/* =====================
		 *  COMMENT FOR AUTHORS
		 * =====================
		 * Using the training data in "rm", you are supposed to write codes to learn your model here.
		 * If your method is memory-based one, you may leave the model as rateMatrix itself, simply by "rateMatrix = rm;".
		 * If your method is model-based algorithm, you may not need a reference to rateMatrix.
		 * (In this case, you may remove the variable "rateMatrix", just as matrix-factorization-based methods do in this toolkit.)
		 * Note that in any case train data in "rateMatrix" are read-only. You should not alter any value in it to guarantee proper operation henceforth.
		 */
	}
	
	/*========================================
	 * Prediction
	 *========================================*/
	/**
	 * Evaluate the designated algorithm with the given test data.
	 * 
	 * @param testMatrix The rating matrix with test data.
	 * 
	 * @return The result of evaluation, such as MAE, RMSE, and rank-score.
	 */
	@Override
	public EvaluationMetrics evaluate(SparseMatrix testMatrix) {
		SparseMatrix predicted = new SparseMatrix(userCount+1, itemCount+1);
		for (int u = 1; u <= userCount; u++) {
			int[] testItems = testMatrix.getRowRef(u).indexList();
			
			if (testItems != null) {
				SparseVector predictedForUser = new SparseVector(itemCount);
				
				for (int i : testItems) {
					/* =====================
					 *  COMMENT FOR AUTHORS
					 * =====================
					 * From your model (model-based) or with your estimation method (memory-based) from rating matrix,
					 * you are supposed to estimate an unseen rating for an item "i" by a user "u" this point.
					 * Please store your estimation for this (u, i) pair in the variable "estimate" below.
					 * If the estimation is not simple, you may make private methods to help the decision.
					 * Obviously again, you should not alter/add/remove any value in testMatrix during the evaluation process.
					 */
					double estimate = 0.0;
										
					/* =====================
					 *  COMMENT FOR AUTHORS
					 * =====================
					 * This part ensures that your algorithm always produces a valid estimation.
					 * You may be freely remove this part under your judge, but you should make sure that
					 * your algorithm does not estimate ratings outside of legal range in the domain.
					 */
					if (estimate < minValue)
						estimate = minValue;
					else if (estimate > maxValue)
						estimate = maxValue;
					
					predictedForUser.setValue(i, estimate);
				}
				
				if (predictedForUser != null) {
					for (int i : predictedForUser.indexList()) {
						predicted.setValue(u, i, predictedForUser.getValue(i));
					}
				}
			}
		}
		
		return new EvaluationMetrics(testMatrix, predicted, maxValue, minValue);
	}
}

/* =====================
 *  COMMENT FOR AUTHORS
 * =====================
 * How to run your algorithm:
 * In the main method, you can easily test your algorithm by the following two steps.
 * 
 * First, make an instance of your recommender with a constructor you implemented.
 *  Ex) CustomRecommender myRecommender = new CustomRecommender(2000, 1000, 5.0, 1.0);
 * 
 * Second, call "testRecommender" method in main method.
 * This returns a String which contains evaluation results.
 * The first argument is the name to be printed, and the second one is the instance you created previously.
 *  Ex) System.out.println(testRecommender("MyRec", myRecommender));
 * 
 * 
 * We provide a unit test module to help you verifying whether your implementation is correct.
 * In the main method, you can make an instance of unit test module with your recommender by
 *  Ex) UnitTest u = new UnitTest(myRecommender, rateMatrix, testMatrix);
 * 
 * After you make the instance, simply call "check" method by
 *  Ex) u.check();
 * 
 * The unit test module may print some warnings or errors based on verification result.
 * If you get some errors, they should be fixed since they imply your implementation is illegal or incorrect.
 * If you get some warnings, you may concern them and we recommend to investigate your code.
 * If the unit test module does not find any problem, it will say so.
 * We recommend to rerun with various parameters since some problems may occur occasionally. 
 */


/* How to call in main method:
 * 
 * 	CustomRecommender myRecommender = new CustomRecommender(userCount, itemCount, maxValue, minValue);
 *	UnitTest u = new UnitTest(myRecommender, rateMatrix, testMatrix);
 *	u.check();
 */
