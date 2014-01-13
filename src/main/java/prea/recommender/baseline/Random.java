package prea.recommender.baseline;

/**
 * The class implementing a baseline, predicting uniformly randomly from the score range.
 * 
 * @author Joonseok Lee
 * @since 2012. 4. 20
 * @version 1.1
 */
public class Random extends BaselineRecommender {
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
	public Random(int uc, int ic, double max, double min) {
		super(uc, ic, max, min);
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
		return Math.random() * (maxValue - minValue) + minValue;
	}
}
