package prea.recommender.baseline;
import prea.data.structure.SparseMatrix;

/**
 * The class implementing a baseline, predicting by overall average of training set ratings.
 * 
 * @author Joonseok Lee
 * @since 2012. 4. 20
 * @version 1.1
 */
public class Average extends BaselineRecommender {
	/** The value which will be used for predicting all ratings. */
	private double constantValue;
	
	/*========================================
	 * Constructors
	 *========================================*/
	/**
	 * Construct a constant model with the given data.
	 * 
	 * @param uc The number of users in the dataset.
	 * @param ic The number of items in the dataset.
	 * @param max The maximum rating value in the dataset.
	 * @param min The minimum rating value in the dataset.
	 */
	public Average(int uc, int ic, double max, double min) {
		super(uc, ic, max, min);
	}
	
	/*========================================
	 * Model Builder
	 *========================================*/
	/**
	 * Build a model with given training set.
	 * 
	 * @param rm Training data set.
	 */
	@Override
	public void buildModel(SparseMatrix rm) {
		super.buildModel(rm);
		constantValue = rm.average();
	}
	
	/*========================================
	 * Prediction
	 *========================================*/
	/**
	 * Predict a rating for the given user and item.
	 * 
	 * @param userId The target user.
	 * @param itemId The target item.
	 * 
	 * @return predicted rating.
	 */
	@Override
	public double predict(int userId, int itemId) {
		return constantValue;
	}
}
