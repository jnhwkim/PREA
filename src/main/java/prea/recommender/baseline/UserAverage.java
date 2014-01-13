package prea.recommender.baseline;
import prea.data.structure.SparseVector;

/**
 * The class implementing a baseline, predicting by the average of target user ratings.
 * 
 * @author Joonseok Lee
 * @since 2012. 4. 20
 * @version 1.1
 */
public class UserAverage extends BaselineRecommender {
	/** Average of ratings for each user. */
	public SparseVector userRateAverage;
	
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
	 * @param ura The average of ratings for each user.
	 */
	public UserAverage(int uc, int ic, double max, double min, SparseVector ura) {
		super(uc, ic, max, min);
		userRateAverage = ura;
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
		return userRateAverage.getValue(userId);
	}
}
