package prea.recommender.baseline;
import prea.data.structure.SparseVector;

/**
 * The class implementing a baseline, predicting by the average of target item ratings.
 * 
 * @author Joonseok Lee
 * @since 2012. 4. 20
 * @version 1.1
 */
public class ItemAverage extends BaselineRecommender {
	/** Average of ratings for each item. */
	public SparseVector itemRateAverage;
	
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
	 * @param ira The average of ratings for each item.
	 */
	public ItemAverage(int uc, int ic, double max, double min, SparseVector ira) {
		super(uc, ic, max, min);
		itemRateAverage = ira;
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
		return itemRateAverage.getValue(itemId);
	}
}
