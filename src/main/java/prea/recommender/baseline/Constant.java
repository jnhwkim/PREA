package prea.recommender.baseline;

/**
 * The class implementing a baseline, always predicting with the given constant.
 * 
 * @author Joonseok Lee
 * @since 2012. 4. 20
 * @version 1.1
 */
public class Constant extends BaselineRecommender {
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
	 * @param val The target value which will be used as constant prediction.
	 */
	public Constant(int uc, int ic, double max, double min, double val) {
		super(uc, ic, max, min);
		constantValue = val;
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
